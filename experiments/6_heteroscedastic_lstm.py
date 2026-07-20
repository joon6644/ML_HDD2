import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data, build_sequences
import math
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

class HeteroscedasticRightCensoredLoss(nn.Module):
    def __init__(self, failure_weight=1.0, survival_weight=1.0, warmup_survival_weight=0.05):
        super(HeteroscedasticRightCensoredLoss, self).__init__()
        self.failure_weight = failure_weight
        self.survival_weight = survival_weight
        self.warmup_survival_weight = warmup_survival_weight

    def forward(self, mu, log_var, labels, censored, is_warmup=False):
        mu = mu.view(-1)
        log_var = log_var.view(-1)
        labels = labels.view(-1)
        censored = censored.view(-1)

        u_mask = (censored == 0)
        c_mask = (censored == 1)

        # Fix log_var to 0.0 (variance to 1.0) during warmup to stabilize optimization.
        if is_warmup:
            log_var_bound = torch.zeros_like(log_var)
        else:
            log_var_bound = -1.0 + F.softplus(log_var)

        # We clamp ONLY for the exp() to prevent float32 overflow (inf).
        # We MUST NOT clamp the penalty term `0.5 * log_var_bound` so gradients always flow!
        var_for_div = torch.exp(torch.clamp(log_var_bound, max=80.0))
        sigma_for_cdf = torch.sqrt(var_for_div) + 1e-6

        # 1. Uncensored Loss: Negative Log Likelihood of Gaussian
        loss_uncensored = 0.5 * log_var_bound + 0.5 * ((labels - mu) ** 2) / var_for_div

        # 2. Censored Loss: Penalize the "Area" estimated below the minimum known RUL
        # CDF of Normal distribution
        cdf = 0.5 * (1.0 + torch.erf((labels - mu) / (sigma_for_cdf * math.sqrt(2))))

        # Survival probability = 1 - CDF
        surv_prob = 1.0 - cdf + 1e-7 # Epsilon for numerical stability

        # Penalty (Negative Log Likelihood of survival)
        loss_censored = -torch.log(surv_prob)

        # Calculate class-wise means safely
        if u_mask.sum() > 0:
            mean_loss_unc = loss_uncensored[u_mask].mean()
        else:
            mean_loss_unc = torch.tensor(0.0, device=mu.device)

        if c_mask.sum() > 0:
            mean_loss_cen = loss_censored[c_mask].mean()
        else:
            mean_loss_cen = torch.tensor(0.0, device=mu.device)

        # Weighted combination based on training phase
        if is_warmup:
            loss = self.failure_weight * mean_loss_unc + self.warmup_survival_weight * mean_loss_cen
        else:
            loss = self.failure_weight * mean_loss_unc + self.survival_weight * mean_loss_cen

        return loss, loss_censored, loss_uncensored

class HeteroscedasticLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(HeteroscedasticLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(32, 1)
        self.logvar_head = nn.Linear(32, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        shared_out = self.shared(last_out)
        mu = self.mu_head(shared_out)
        log_var = self.logvar_head(shared_out)
        return mu, log_var, shared_out

def eval_pass(model, criterion, X, y, c, batch_size, device):
    """Run evaluation pass using CPU memory with fast non-blocking transfer to GPU."""
    model.eval()
    val_loss = torch.tensor(0.0, device=device)
    val_mse = torch.tensor(0.0, device=device)
    val_mae = torch.tensor(0.0, device=device)
    val_uncensored = torch.tensor(0.0, device=device)
    val_y_sum = torch.tensor(0.0, device=device)
    val_y_sq_sum = torch.tensor(0.0, device=device)
    val_log_vars = []
    n = len(X)
    with torch.no_grad():
        for i in range(math.ceil(n / batch_size)):
            bx = X[i * batch_size:(i + 1) * batch_size].to(device, non_blocking=True)
            by = y[i * batch_size:(i + 1) * batch_size].to(device, non_blocking=True)
            bc = c[i * batch_size:(i + 1) * batch_size].to(device, non_blocking=True)

            mu, log_var, _ = model(bx)
            loss, _, _ = criterion(mu, log_var, by, bc, is_warmup=False)
            val_log_vars.append(log_var.detach())

            pf = mu.view(-1)
            yf = by.view(-1)
            cf = bc.view(-1)

            val_loss += loss * bx.size(0)

            preds_clipped = torch.clamp(pf, min=-2.0, max=9.0)
            preds_orig = torch.expm1(preds_clipped)
            y_orig = torch.expm1(yf)

            uf = 1.0 - cf
            diff_u = (preds_orig - y_orig) * uf

            val_mse += torch.sum(diff_u ** 2)
            val_mae += torch.sum(torch.abs(diff_u))
            val_y_sum += torch.sum(y_orig * uf)
            val_y_sq_sum += torch.sum((y_orig ** 2) * uf)
            val_uncensored += torch.sum(uf)
    return val_loss, val_mse, val_mae, val_y_sum, val_y_sq_sum, val_uncensored, val_log_vars

def main():
    train_df, val_df, features = get_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    window_size = 7
    batch_size = 16384

    print("Preparing sequence datasets...")
    X_train, y_train, c_train = build_sequences(train_df, features, window_size)
    print(f"Train sequences: {len(X_train)}")
    X_val, y_val, c_val = build_sequences(val_df, features, window_size)
    print(f"Test sequences: {len(X_val)}")
    
    # 텐서 변환 완료 후 대용량 pandas DataFrame들을 즉시 지우고 가비지 컬렉터 강제 구동
    import gc
    del train_df, val_df
    gc.collect()

    # Pin memory logic removed to prevent host OOM crash with large sequence data

    model = HeteroscedasticLSTM(input_dim=len(features)).to(device)
    criterion = HeteroscedasticRightCensoredLoss(failure_weight=1.0, survival_weight=1.0, warmup_survival_weight=0.05).to(device)

    # Use distinct learning rates for stability
    warmup_epochs = 10
    
    # Phase 1: Sigma fixed
    for param in model.logvar_head.parameters():
        param.requires_grad = False
        
    optimizer = optim.Adam([
        {'params': model.lstm.parameters(), 'lr': 1e-3},
        {'params': model.shared.parameters(), 'lr': 1e-3},
        {'params': model.mu_head.parameters(), 'lr': 1e-3}
    ])

    epochs = 100
    history = []
    n_train = len(X_train)
    is_warmup = True

    print("Training Heteroscedastic LSTM model with area-based survival loss...")
    for epoch in range(epochs):
        if epoch == warmup_epochs:
            print("\n>>> Warm-up finished. Switching to joint training of mean and sigma. <<<")
            is_warmup = False
            for param in model.logvar_head.parameters():
                param.requires_grad = True
            optimizer = optim.Adam([
                {'params': model.lstm.parameters(), 'lr': 1e-3},
                {'params': model.shared.parameters(), 'lr': 1e-3},
                {'params': model.mu_head.parameters(), 'lr': 1e-3},
                {'params': model.logvar_head.parameters(), 'lr': 1e-4}
            ])

        model.train()
        total_loss = torch.tensor(0.0, device=device)
        total_mse = torch.tensor(0.0, device=device)
        total_mae = torch.tensor(0.0, device=device)
        total_uncensored = torch.tensor(0.0, device=device)
        total_y_sum = torch.tensor(0.0, device=device)
        total_y_sq_sum = torch.tensor(0.0, device=device)

        # Diagnostics accumulators
        epoch_lc_sum = torch.tensor(0.0, device=device)
        epoch_lu_sum = torch.tensor(0.0, device=device)
        epoch_lc_count = torch.tensor(0.0, device=device)
        epoch_lu_count = torch.tensor(0.0, device=device)
        epoch_mu_c_sum = torch.tensor(0.0, device=device)
        epoch_mu_u_sum = torch.tensor(0.0, device=device)

        indices = torch.randperm(n_train)
        num_batches = math.ceil(n_train / batch_size)

        # Use tqdm for progress bar
        desc_str = f"Epoch {epoch+1}/{epochs} (Warmup)" if is_warmup else f"Epoch {epoch+1}/{epochs} (Joint)"
        train_pbar = tqdm(range(num_batches), desc=desc_str, leave=False)
        for i in train_pbar:
            batch_idx = indices[i * batch_size:(i + 1) * batch_size]
            batch_x = X_train[batch_idx].to(device, non_blocking=True)
            batch_y = y_train[batch_idx].to(device, non_blocking=True)
            batch_c = c_train[batch_idx].to(device, non_blocking=True)

            optimizer.zero_grad()
            mu, log_var, _ = model(batch_x)
            loss, lc, lu = criterion(mu, log_var, batch_y, batch_c, is_warmup=is_warmup)
            loss.backward()
            optimizer.step()

            # Diagnostic Accumulation
            with torch.no_grad():
                c_mask = (batch_c == 1)
                u_mask = (batch_c == 0)

                epoch_lc_sum += lc[c_mask].sum()
                epoch_lc_count += c_mask.sum()

                epoch_lu_sum += lu[u_mask].sum()
                epoch_lu_count += u_mask.sum()

                epoch_mu_c_sum += mu.view(-1)[c_mask].sum()
                epoch_mu_u_sum += mu.view(-1)[u_mask].sum()

            pf = mu.detach().view(-1)
            yf = batch_y.view(-1)
            cf = batch_c.view(-1)

            total_loss += loss.detach() * batch_x.size(0)

            with torch.no_grad():
                preds_clipped = torch.clamp(pf, min=-2.0, max=9.0)
                preds_orig = torch.expm1(preds_clipped)
                y_orig = torch.expm1(yf)

                uf = 1.0 - cf
                diff_u = (preds_orig - y_orig) * uf

                total_mse += torch.sum(diff_u ** 2)
                total_mae += torch.sum(torch.abs(diff_u))
                total_y_sum += torch.sum(y_orig * uf)
                total_y_sq_sum += torch.sum((y_orig ** 2) * uf)
                total_uncensored += torch.sum(uf)

        val_loss, val_mse, val_mae, val_y_sum, val_y_sq_sum, val_uncensored, val_log_vars = eval_pass(
            model, criterion, X_val, y_val, c_val, batch_size, device
        )

        t_stats = torch.stack([
            total_loss, total_mse, total_mae, total_y_sum, total_y_sq_sum, total_uncensored
        ]).cpu().tolist()
        tl, t_mse_sum, t_mae_sum, t_y_sum, t_y_sq_sum, t_unc = t_stats

        v_stats = torch.stack([
            val_loss, val_mse, val_mae, val_y_sum, val_y_sq_sum, val_uncensored
        ]).cpu().tolist()
        vl, v_mse_sum, v_mae_sum, v_y_sum, v_y_sq_sum, v_unc = v_stats

        if t_unc > 0:
            t_mse = t_mse_sum / t_unc
            t_rmse = math.sqrt(t_mse)
            t_mae = t_mae_sum / t_unc
            t_mean_y = t_y_sum / t_unc
            t_ss_tot = t_y_sq_sum - t_unc * (t_mean_y ** 2)
            t_r2 = 1 - (t_mse * t_unc / t_ss_tot) if t_ss_tot > 0 else 0.0
        else:
            t_mse = t_rmse = t_mae = t_r2 = 0.0

        if v_unc > 0:
            v_mse = v_mse_sum / v_unc
            v_rmse = math.sqrt(v_mse)
            v_mae = v_mae_sum / v_unc
            v_mean_y = v_y_sum / v_unc
            v_ss_tot = v_y_sq_sum - v_unc * (v_mean_y ** 2)
            v_r2 = 1 - (v_mse * v_unc / v_ss_tot) if v_ss_tot > 0 else 0.0
        else:
            v_mse = v_rmse = v_mae = v_r2 = 0.0

        phase_str = "[Warm-up]" if is_warmup else "[Joint]"
        print(f"Epoch {epoch+1}/{epochs} {phase_str}")
        print(f"  Train -> Loss: {tl / n_train:.4f} | MSE: {t_mse:.4f} | RMSE: {t_rmse:.4f} | MAE: {t_mae:.4f} | R2: {t_r2:.4f}")
        print(f"  Val   -> Loss: {vl / len(X_val):.4f} | MSE: {v_mse:.4f} | RMSE: {v_rmse:.4f} | MAE: {v_mae:.4f} | R2: {v_r2:.4f}")

        # Diagnostic prints
        lc_mean = (epoch_lc_sum / epoch_lc_count).item() if epoch_lc_count > 0 else 0
        lu_mean = (epoch_lu_sum / epoch_lu_count).item() if epoch_lu_count > 0 else 0
        mu_c_mean = (epoch_mu_c_sum / epoch_lc_count).item() if epoch_lc_count > 0 else 0
        mu_u_mean = (epoch_mu_u_sum / epoch_lu_count).item() if epoch_lu_count > 0 else 0

        print(f"  [Diagnostics] Uncensored Loss Mean: {lu_mean:.4f} | Count: {epoch_lu_count.item():.0f} | mu mean: {mu_u_mean:.4f}")
        print(f"  [Diagnostics] Censored Loss Mean:   {lc_mean:.4f} | Count: {epoch_lc_count.item():.0f} | mu mean: {mu_c_mean:.4f}")

        all_val_log_vars = torch.cat(val_log_vars)
        val_logvar_mean = all_val_log_vars.mean().item()
        val_logvar_std = all_val_log_vars.std().item()
        val_logvar_min = all_val_log_vars.min().item()
        val_logvar_max = all_val_log_vars.max().item()
        print(f"  [Diagnostics] log_var (Val) -> mean: {val_logvar_mean:.4f} | std: {val_logvar_std:.4f} | min: {val_logvar_min:.4f} | max: {val_logvar_max:.4f}")
        print("-" * 80)

        # Record metrics for CSV logging in real-time
        epoch_data = {
            'epoch': epoch + 1,
            'warmup': 1 if is_warmup else 0,
            'train_loss': tl / n_train,
            'train_mse': t_mse,
            'train_rmse': t_rmse,
            'train_mae': t_mae,
            'train_r2': t_r2,
            'val_loss': vl / len(X_val),
            'val_mse': v_mse,
            'val_rmse': v_rmse,
            'val_mae': v_mae,
            'val_r2': v_r2,
            'diag_uncensored_loss_mean': lu_mean,
            'diag_censored_loss_mean': lc_mean,
            'diag_val_logvar_mean': val_logvar_mean,
            'diag_val_logvar_std': val_logvar_std,
            'diag_val_logvar_min': val_logvar_min,
            'diag_val_logvar_max': val_logvar_max
        }
        history.append(epoch_data)
        from data_loader import log_epoch_to_csv
        log_epoch_to_csv("6_heteroscedastic_lstm", epoch_data)

    print("Training finished.")

if __name__ == "__main__":
    main()
