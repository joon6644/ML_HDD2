import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data, build_sequences
import math
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

class LogNormalAFTLoss(nn.Module):
    def __init__(self, failure_weight=8.0):
        super(LogNormalAFTLoss, self).__init__()
        self.log_sigma = nn.Parameter(torch.zeros(1))
        self.failure_weight = failure_weight

    def forward(self, preds, labels, censored):
        preds = preds.view(-1)
        labels = labels.view(-1)
        censored = censored.view(-1)

        log_sigma_bound = -1.0 + F.softplus(self.log_sigma)
        sigma_for_div = torch.exp(torch.clamp(log_sigma_bound, max=40.0))
        z = (labels - preds) / sigma_for_div

        loss_uncensored = log_sigma_bound + 0.5 * (z ** 2) + 0.5 * math.log(2 * math.pi)

        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2)))
        surv = 1.0 - cdf + 1e-7
        loss_censored = -torch.log(surv)

        loss = torch.where(censored == 1, loss_censored, loss_uncensored * self.failure_weight)
        return loss.mean()

class GRUReg(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(GRUReg, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        return self.fc(last_out)

def eval_pass(model, criterion, X, y, c, batch_size, device):
    """Run evaluation pass using CPU memory with fast non-blocking transfer to GPU."""
    model.eval()
    val_loss = torch.tensor(0.0, device=device)
    val_mse = torch.tensor(0.0, device=device)
    val_mae = torch.tensor(0.0, device=device)
    val_uncensored = torch.tensor(0.0, device=device)
    val_y_sum = torch.tensor(0.0, device=device)
    val_y_sq_sum = torch.tensor(0.0, device=device)
    n = len(X)
    with torch.no_grad():
        for i in range(math.ceil(n / batch_size)):
            bx = X[i * batch_size:(i + 1) * batch_size].to(device, non_blocking=True)
            by = y[i * batch_size:(i + 1) * batch_size].to(device, non_blocking=True)
            bc = c[i * batch_size:(i + 1) * batch_size].to(device, non_blocking=True)

            preds = model(bx)
            loss = criterion(preds, by, bc)

            pf = preds.view(-1)
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
    return val_loss, val_mse, val_mae, val_y_sum, val_y_sq_sum, val_uncensored

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

    # Pin memory for faster CPU-to-GPU data transfer
    X_train = X_train.pin_memory()
    y_train = y_train.pin_memory()
    c_train = c_train.pin_memory()
    X_val = X_val.pin_memory()
    y_val = y_val.pin_memory()
    c_val = c_val.pin_memory()

    print("Datasets prepared and pinned on CPU.")

    model = GRUReg(input_dim=len(features)).to(device)
    criterion = LogNormalAFTLoss(failure_weight=14.0).to(device)

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 1e-3},
        {'params': criterion.parameters(), 'lr': 1e-4}
    ])

    epochs = 100
    history = []
    n_train = len(X_train)

    print("Training GRU model with custom right-censored loss...")
    for epoch in range(epochs):
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        total_mse = torch.tensor(0.0, device=device)
        total_mae = torch.tensor(0.0, device=device)
        total_uncensored = torch.tensor(0.0, device=device)
        total_y_sum = torch.tensor(0.0, device=device)
        total_y_sq_sum = torch.tensor(0.0, device=device)

        indices = torch.randperm(n_train)
        num_batches = math.ceil(n_train / batch_size)

        train_pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for i in train_pbar:
            batch_idx = indices[i * batch_size:(i + 1) * batch_size]
            batch_x = X_train[batch_idx].to(device, non_blocking=True)
            batch_y = y_train[batch_idx].to(device, non_blocking=True)
            batch_c = c_train[batch_idx].to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y, batch_c)
            loss.backward()
            optimizer.step()

            pf = preds.detach().view(-1)
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

        val_loss, val_mse, val_mae, val_y_sum, val_y_sq_sum, val_uncensored = eval_pass(
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

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train -> Loss: {tl / n_train:.4f} | MSE: {t_mse:.4f} | RMSE: {t_rmse:.4f} | MAE: {t_mae:.4f} | R2: {t_r2:.4f}")
        print(f"  Val   -> Loss: {vl / len(X_val):.4f} | MSE: {v_mse:.4f} | RMSE: {v_rmse:.4f} | MAE: {v_mae:.4f} | R2: {v_r2:.4f}")

        # Record metrics for CSV logging in real-time
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': tl / n_train,
            'train_mse': t_mse,
            'train_rmse': t_rmse,
            'train_mae': t_mae,
            'train_r2': t_r2,
            'val_loss': vl / len(X_val),
            'val_mse': v_mse,
            'val_rmse': v_rmse,
            'val_mae': v_mae,
            'val_r2': v_r2
        }
        history.append(epoch_data)
        from data_loader import log_epoch_to_csv
        log_epoch_to_csv("4_gru_regression", epoch_data)

    print("Training finished.")

if __name__ == "__main__":
    main()
