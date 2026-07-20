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
    def __init__(self, failure_weight=8.0):
        super(HeteroscedasticRightCensoredLoss, self).__init__()
        self.failure_weight = failure_weight
        
    def forward(self, mu, log_var, labels, censored):
        mu = mu.view(-1)
        log_var = log_var.view(-1)
        labels = labels.view(-1)
        censored = censored.view(-1)
        
        # Smoothly bound minimum log_var to -1.0 using softplus.
        log_var_bound = -1.0 + F.softplus(log_var)
        
        # We clamp ONLY for the exp() to prevent float32 overflow (inf).
        # We MUST NOT clamp the penalty term `0.5 * log_var_bound` so gradients always flow!
        var_for_div = torch.exp(torch.clamp(log_var_bound, max=80.0))
        sigma_for_cdf = torch.sqrt(var_for_div) + 1e-6
        
        # 1. Uncensored Loss: Negative Log Likelihood of Gaussian
        # 0.5 * log(var) + 0.5 * (y - mu)^2 / var
        loss_uncensored = 0.5 * log_var_bound + 0.5 * ((labels - mu) ** 2) / var_for_div
        
        # 2. Censored Loss: Penalize the "Area" estimated below the minimum known RUL
        # "예측값의 넓이로 최소 며칠 살았는지와 대조하여 그보다 낮게 추정한 면적만큼을 페널티"
        # The probability area estimated below `labels` is exactly the CDF at `labels`.
        # Standard Survival NLL is -log(1 - CDF). As the "lower area" (CDF) grows, the penalty grows.
        
        # CDF of Normal distribution
        cdf = 0.5 * (1.0 + torch.erf((labels - mu) / (sigma_for_cdf * math.sqrt(2))))
        
        # Survival probability = 1 - CDF
        surv_prob = 1.0 - cdf + 1e-7 # Epsilon for numerical stability
        
        # Penalty (Negative Log Likelihood of survival)
        loss_censored = -torch.log(surv_prob)
        
        # Weighting uncensored loss to balance the dataset distribution.
        loss = torch.where(censored == 1, loss_censored, loss_uncensored * self.failure_weight)
        
        return loss.mean(), loss_censored, loss_uncensored

class HeteroscedasticMLP(nn.Module):
    def __init__(self, input_dim):
        super(HeteroscedasticMLP, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
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
    
    # Only train on failure data
    train_df = train_df[train_df['censored'] == 0].copy()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_df['log_RUL'] = np.log1p(train_df['RUL'])
    val_df['log_RUL'] = np.log1p(val_df['RUL'])
    
    # Convert to tensors and move directly to GPU memory (dataset is small enough)
    X_train = torch.tensor(train_df[features].values, dtype=torch.float32, device=device)
    y_train = torch.tensor(train_df['log_RUL'].values, dtype=torch.float32, device=device)
    c_train = torch.tensor(train_df['censored'].values, dtype=torch.float32, device=device)
    
    # Extremely large batch size to maximize GPU usage for this tiny model
    batch_size = 262144
    
    X_val = torch.tensor(val_df[features].values, dtype=torch.float32, device=device)
    y_val = torch.tensor(val_df['log_RUL'].values, dtype=torch.float32, device=device)
    c_val = torch.tensor(val_df['censored'].values, dtype=torch.float32, device=device)

    model = HeteroscedasticMLP(input_dim=len(features)).to(device)
    # Adjustable failure weight for experimentation
    criterion = HeteroscedasticRightCensoredLoss(failure_weight=14.0)
    
    # Use distinct learning rates: slower learning rate for variance head to prevent explosion
    optimizer = optim.Adam([
        {'params': model.shared.parameters(), 'lr': 1e-3},
        {'params': model.mu_head.parameters(), 'lr': 1e-3},
        {'params': model.logvar_head.parameters(), 'lr': 1e-4}
    ])
    
    epochs = 1000
    
    print("Training Heteroscedastic MLP model with area-based survival loss...")
    for epoch in range(epochs):
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
        # Generate random indices for shuffling natively on GPU
        indices = torch.randperm(len(X_train), device=device)
        num_batches = math.ceil(len(X_train) / batch_size)
        
        # Use tqdm for progress bar
        train_pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for i in train_pbar:
            batch_idx = indices[i * batch_size : (i + 1) * batch_size]
            batch_x = X_train[batch_idx]
            batch_y = y_train[batch_idx]
            batch_c = c_train[batch_idx]
            
            optimizer.zero_grad()
            mu, log_var, _ = model(batch_x)
            loss, lc, lu = criterion(mu, log_var, batch_y, batch_c)
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
                batch_x = X_val[i * batch_size : (i + 1) * batch_size]
                batch_y = y_val[i * batch_size : (i + 1) * batch_size]
                batch_c = c_val[i * batch_size : (i + 1) * batch_size]
                
                mu, log_var, shared_out = model(batch_x)
                val_log_vars.append(log_var.detach())
                
                # Outlier diagnosis using dataframe information
                extreme_mask = log_var > 1000
                if extreme_mask.any():
                    idx_in_batch = torch.nonzero(extreme_mask, as_tuple=True)[0][0].item()
                    global_idx = i * batch_size + idx_in_batch
                    
                    # Extract original row info
                    row = val_df.iloc[global_idx]
                    disk_id = row.get('disk_id', 'Unknown')
                    date = row.get('date', 'Unknown')
                    
                    x_outlier = batch_x[idx_in_batch]
                    
                    print(f"\n[Extreme Outlier Detected]")
                    print(f"  HDD ID: {disk_id} | Date: {date}")
                    print(f"  Input (x) -> norm: {torch.norm(x_outlier).item():.4f} | max(abs): {torch.max(torch.abs(x_outlier)).item():.4f}")
                    print(f"  shared_out -> norm: {torch.norm(shared_out[idx_in_batch]).item():.4f} | max: {shared_out[idx_in_batch].max().item():.4f}")
                    print(f"  Outputs -> mu: {mu[idx_in_batch].item():.4f} | log_var: {log_var[idx_in_batch].item():.4f}")
                
                loss, _, _ = criterion(mu, log_var, batch_y, batch_c)
                
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
            
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train -> Loss: {(total_loss / len(X_train)).item():.4f} | MSE: {t_mse:.4f} | RMSE: {t_rmse:.4f} | MAE: {t_mae:.4f} | R2: {t_r2:.4f}")
        print(f"  Val   -> Loss: {(val_loss / len(X_val)).item():.4f} | MSE: {v_mse:.4f} | RMSE: {v_rmse:.4f} | MAE: {v_mae:.4f} | R2: {v_r2:.4f}")
        
        # Diagnostic prints as requested
        lc_mean = (epoch_lc_sum / epoch_lc_count).item() if epoch_lc_count > 0 else 0
        lu_mean = (epoch_lu_sum / epoch_lu_count).item() if epoch_lu_count > 0 else 0
        mu_c_mean = (epoch_mu_c_sum / epoch_lc_count).item() if epoch_lc_count > 0 else 0
        mu_u_mean = (epoch_mu_u_sum / epoch_lu_count).item() if epoch_lu_count > 0 else 0
        
        print(f"  [Diagnostics] Uncensored Loss Mean: {lu_mean:.4f} | Count: {epoch_lu_count.item():.0f} | mu mean: {mu_u_mean:.4f}")
        print(f"  [Diagnostics] Censored Loss Mean:   {lc_mean:.4f} | Count: {epoch_lc_count.item():.0f} | mu mean: {mu_c_mean:.4f}")
        
        all_val_log_vars = torch.cat(val_log_vars)
        print(f"  [Diagnostics] log_var (Val) -> mean: {all_val_log_vars.mean().item():.4f} | std: {all_val_log_vars.std().item():.4f} | min: {all_val_log_vars.min().item():.4f} | max: {all_val_log_vars.max().item():.4f}")
        print("-" * 80)

    print("Training finished.")

if __name__ == "__main__":
    main()
