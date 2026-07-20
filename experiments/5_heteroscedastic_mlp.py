import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from data_loader import get_data
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

class HeteroscedasticMLP(nn.Module):
    def __init__(self, input_dim):
        super(HeteroscedasticMLP, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Outputs two values: mu and log_var
        self.mu_head = nn.Linear(32, 1)
        self.logvar_head = nn.Linear(32, 1)
        
    def forward(self, x):
        shared_out = self.shared(x)
        mu = self.mu_head(shared_out)
        log_var = self.logvar_head(shared_out)
        return mu, log_var, shared_out

def main():
    train_df, val_df, features = get_data()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_df['log_RUL'] = np.log1p(train_df['RUL'])
    val_df['log_RUL'] = np.log1p(val_df['RUL'])
    
    import gc
    # Convert to tensors on CPU memory to avoid VRAM exhaustion
    X_train = torch.tensor(train_df[features].values, dtype=torch.float32, device='cpu')
    y_train = torch.tensor(train_df['log_RUL'].values, dtype=torch.float32, device='cpu')
    c_train = torch.tensor(train_df['censored'].values, dtype=torch.float32, device='cpu')
    
    X_val = torch.tensor(val_df[features].values, dtype=torch.float32, device='cpu')
    y_val = torch.tensor(val_df['log_RUL'].values, dtype=torch.float32, device='cpu')
    c_val = torch.tensor(val_df['censored'].values, dtype=torch.float32, device='cpu')

    # 메모리 절약을 위해 metadata만 가볍게 복사해두고, 거대한 DataFrame들은 삭제
    val_meta = val_df[['serial_number', 'date']].values.astype(str)
    
    # 텐서 변환 완료 후 대용량 pandas DataFrame들을 즉시 지우고 가비지 컬렉터 강제 구동
    del train_df, val_df
    gc.collect()
    
    # --- Stratified batch sampling: guarantee fail/censored mix per batch ---
    fail_idx = torch.where(c_train == 0)[0]   # uncensored (failed) indices
    cen_idx  = torch.where(c_train == 1)[0]   # censored (survived) indices
    n_fail_total = len(fail_idx)
    n_cen_total  = len(cen_idx)
    failure_ratio = n_fail_total / (n_fail_total + n_cen_total)
    n_fail_per_batch = max(1, round(batch_size * failure_ratio))
    n_cen_per_batch  = batch_size - n_fail_per_batch
    # Number of complete batches is determined by the censored pool (larger group)
    num_batches_strat = n_cen_total // n_cen_per_batch
    print(f"Stratified batching: {n_fail_total:,} fail | {n_cen_total:,} censored")
    print(f"  Per batch: {n_fail_per_batch} fail + {n_cen_per_batch} censored = {batch_size} total")
    print(f"  Batches per epoch: {num_batches_strat:,}")

    model = HeteroscedasticMLP(input_dim=len(features)).to(device)
    # Adjustable failure weight for experimentation
    criterion = HeteroscedasticRightCensoredLoss(failure_weight=1.0, survival_weight=0.05, warmup_survival_weight=0.05).to(device)
    
    # Use distinct learning rates: slower learning rate for variance head to prevent explosion
    warmup_epochs = 6
    
    # Phase 1: Sigma fixed (logvar_head frozen)
    for param in model.logvar_head.parameters():
        param.requires_grad = False
        
    optimizer = optim.Adam([
        {'params': model.shared.parameters(), 'lr': 1e-3},
        {'params': model.mu_head.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-5)
    
    epochs = 1000
    history = []
    is_warmup = True
    
    print("Training Heteroscedastic MLP model with area-based survival loss...")
    for epoch in range(epochs):
        if epoch == warmup_epochs:
            print("\n>>> Warm-up finished. Switching to joint training of mean and sigma. <<<")
            is_warmup = False
            for param in model.logvar_head.parameters():
                param.requires_grad = True
            optimizer = optim.Adam([
                {'params': model.shared.parameters(), 'lr': 1e-3},
                {'params': model.mu_head.parameters(), 'lr': 1e-3},
                {'params': model.logvar_head.parameters(), 'lr': 1e-4}
            ], weight_decay=1e-5)

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
        # --- Stratified batch construction ---
        # Shuffle both groups each epoch
        fail_perm = fail_idx[torch.randperm(n_fail_total)]
        cen_perm  = cen_idx[torch.randperm(n_cen_total)]
        n_epoch_samples = num_batches_strat * batch_size
        
        # Use tqdm for progress bar
        desc_str = f"Epoch {epoch+1}/{epochs} (Warmup)" if is_warmup else f"Epoch {epoch+1}/{epochs} (Joint)"
        train_pbar = tqdm(range(num_batches_strat), desc=desc_str, leave=False)
        for i in train_pbar:
            # Censored slice (sequential within epoch)
            cen_start = i * n_cen_per_batch
            batch_cen_idx = cen_perm[cen_start : cen_start + n_cen_per_batch]
            # Fail slice (cycle through fail_perm across batches)
            f_start = (i * n_fail_per_batch) % n_fail_total
            f_end   = f_start + n_fail_per_batch
            if f_end <= n_fail_total:
                batch_fail_idx = fail_perm[f_start:f_end]
            else:
                batch_fail_idx = torch.cat([fail_perm[f_start:], fail_perm[:f_end - n_fail_total]])
            # Merge and shuffle within batch
            batch_idx = torch.cat([batch_fail_idx, batch_cen_idx])
            batch_idx = batch_idx[torch.randperm(len(batch_idx))]
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
            
            mu_flat = mu.view(-1)
            y_flat = batch_y.view(-1)
            c_flat = batch_c.view(-1)
            
            total_loss += loss.detach() * batch_x.size(0)
            
            # max log_RUL in data is ~7.9, so 9.0 is a safe evaluation upper bound
            preds_clipped = torch.clamp(mu_flat, min=-2.0, max=9.0)
            preds_orig = torch.expm1(preds_clipped)
            y_orig = torch.expm1(y_flat)
            
            uncensored_float = (1.0 - c_flat)
            diff_u = (preds_orig - y_orig) * uncensored_float
            
            total_mse += torch.sum(diff_u ** 2)
            total_mae += torch.sum(torch.abs(diff_u))
            total_y_sum += torch.sum(y_orig * uncensored_float)
            total_y_sq_sum += torch.sum((y_orig ** 2) * uncensored_float)
            total_uncensored += torch.sum(uncensored_float)
            
        model.eval()
        val_loss = torch.tensor(0.0, device=device)
        val_mse = torch.tensor(0.0, device=device)
        val_mae = torch.tensor(0.0, device=device)
        val_uncensored = torch.tensor(0.0, device=device)
        val_y_sum = torch.tensor(0.0, device=device)
        val_y_sq_sum = torch.tensor(0.0, device=device)
        val_log_vars = []
        
        num_val_batches = math.ceil(len(X_val) / batch_size)
        val_pbar = tqdm(range(num_val_batches), desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for i in val_pbar:
                batch_x = X_val[i * batch_size : (i + 1) * batch_size].to(device, non_blocking=True)
                batch_y = y_val[i * batch_size : (i + 1) * batch_size].to(device, non_blocking=True)
                batch_c = c_val[i * batch_size : (i + 1) * batch_size].to(device, non_blocking=True)
                
                mu, log_var, shared_out = model(batch_x)
                if is_warmup:
                    val_log_vars.append(torch.zeros_like(log_var))
                else:
                    val_log_vars.append(log_var.detach())
                
                # Outlier diagnosis using dataframe information
                extreme_mask = log_var > 1000
                if extreme_mask.any():
                    idx_in_batch = torch.nonzero(extreme_mask, as_tuple=True)[0][0].item()
                    global_idx = i * batch_size + idx_in_batch
                    
                    # Extract original metadata info from cached val_meta
                    disk_id, date = val_meta[global_idx]
                    
                    x_outlier = batch_x[idx_in_batch]
                    
                    print(f"\n[Extreme Outlier Detected]")
                    print(f"  HDD ID: {disk_id} | Date: {date}")
                    print(f"  Input (x) -> norm: {torch.norm(x_outlier).item():.4f} | max(abs): {torch.max(torch.abs(x_outlier)).item():.4f}")
                    print(f"  shared_out -> norm: {torch.norm(shared_out[idx_in_batch]).item():.4f} | max: {shared_out[idx_in_batch].max().item():.4f}")
                    print(f"  Outputs -> mu: {mu[idx_in_batch].item():.4f} | log_var: {log_var[idx_in_batch].item():.4f}")
                
                loss, _, _ = criterion(mu, log_var, batch_y, batch_c, is_warmup=is_warmup)
                
                mu_flat = mu.view(-1)
                y_flat = batch_y.view(-1)
                c_flat = batch_c.view(-1)
                
                val_loss += loss.detach() * batch_x.size(0)
                
                # max log_RUL in data is ~7.9, so 9.0 is a safe evaluation upper bound
                preds_clipped = torch.clamp(mu_flat, min=-2.0, max=9.0)
                preds_orig = torch.expm1(preds_clipped)
                y_orig = torch.expm1(y_flat)
                
                uncensored_float = (1.0 - c_flat)
                diff_u = (preds_orig - y_orig) * uncensored_float
                
                val_mse += torch.sum(diff_u ** 2)
                val_mae += torch.sum(torch.abs(diff_u))
                val_y_sum += torch.sum(y_orig * uncensored_float)
                val_y_sq_sum += torch.sum((y_orig ** 2) * uncensored_float)
                val_uncensored += torch.sum(uncensored_float)
                
        # Move accumulated metrics to CPU once at the end of the epoch
        total_uncensored_val = total_uncensored.item()
        if total_uncensored_val > 0:
            t_mse = (total_mse / total_uncensored_val).item()
            t_rmse = math.sqrt(t_mse)
            t_mae = (total_mae / total_uncensored_val).item()
            t_mean_y = (total_y_sum / total_uncensored_val).item()
            t_ss_tot = (total_y_sq_sum.item() - total_uncensored_val * (t_mean_y ** 2))
            t_r2 = 1 - (t_mse * total_uncensored_val / t_ss_tot) if t_ss_tot > 0 else 0.0
        else:
            t_mse = t_rmse = t_mae = t_r2 = 0.0
            
        val_uncensored_val = val_uncensored.item()
        if val_uncensored_val > 0:
            v_mse = (val_mse / val_uncensored_val).item()
            v_rmse = math.sqrt(v_mse)
            v_mae = (val_mae / val_uncensored_val).item()
            v_mean_y = (val_y_sum / val_uncensored_val).item()
            v_ss_tot = (val_y_sq_sum.item() - val_uncensored_val * (v_mean_y ** 2))
            v_r2 = 1 - (v_mse * val_uncensored_val / v_ss_tot) if v_ss_tot > 0 else 0.0
        else:
            v_mse = v_rmse = v_mae = v_r2 = 0.0
            
        phase_str = "[Warm-up]" if is_warmup else "[Joint]"
        print(f"Epoch {epoch+1}/{epochs} {phase_str}")
        print(f"  Train -> Loss: {(total_loss / n_epoch_samples).item():.4f} | MSE: {t_mse:.4f} | RMSE: {t_rmse:.4f} | MAE: {t_mae:.4f} | R2: {t_r2:.4f}")
        print(f"  Val   -> Loss: {(val_loss / len(X_val)).item():.4f} | MSE: {v_mse:.4f} | RMSE: {v_rmse:.4f} | MAE: {v_mae:.4f} | R2: {v_r2:.4f}")
        
        # Diagnostic prints as requested
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
        logvar_note = " [loss uses fixed var=1.0]" if is_warmup else ""
        print(f"  [Diagnostics] log_var (Val) -> mean: {val_logvar_mean:.4f} | std: {val_logvar_std:.4f} | min: {val_logvar_min:.4f} | max: {val_logvar_max:.4f}{logvar_note}")
        print("-" * 80)

        # Record metrics for CSV logging in real-time
        epoch_data = {
            'epoch': epoch + 1,
            'warmup': 1 if is_warmup else 0,
            'train_loss': (total_loss / n_epoch_samples).item(),
            'train_mse': t_mse,
            'train_rmse': t_rmse,
            'train_mae': t_mae,
            'train_r2': t_r2,
            'val_loss': (val_loss / len(X_val)).item(),
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
        log_epoch_to_csv("5_heteroscedastic_mlp", epoch_data)

    print("Training finished.")

if __name__ == "__main__":
    main()
