import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from data_loader import get_data
import math
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F

class LogNormalAFTLoss(nn.Module):
    def __init__(self, failure_weight=8.0):
        super(LogNormalAFTLoss, self).__init__()
        # Learnable global log_sigma
        self.log_sigma = nn.Parameter(torch.zeros(1))
        self.failure_weight = failure_weight
        
    def forward(self, preds, labels, censored):
        preds = preds.view(-1)
        labels = labels.view(-1)
        censored = censored.view(-1)
        
        log_sigma_bound = -1.0 + F.softplus(self.log_sigma)
        sigma_for_div = torch.exp(torch.clamp(log_sigma_bound, max=40.0))
        z = (labels - preds) / sigma_for_div
        
        # Uncensored: log(sigma * sqrt(2*pi)) + 0.5 * z^2
        loss_uncensored = log_sigma_bound + 0.5 * (z ** 2) + 0.5 * math.log(2 * math.pi)
        
        # Censored: -log(1 - CDF(z))
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2)))
        surv = 1.0 - cdf + 1e-7
        loss_censored = -torch.log(surv)
        
        loss = torch.where(censored == 1, loss_censored, loss_uncensored * self.failure_weight)
        
        return loss.mean()

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def main():
    train_df, val_df, features = get_data()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_df['log_RUL'] = np.log1p(train_df['RUL'])
    val_df['log_RUL'] = np.log1p(val_df['RUL'])
    
    # Convert to tensors and move directly to GPU memory
    X_train = torch.tensor(train_df[features].values, dtype=torch.float32, device=device)
    y_train = torch.tensor(train_df['log_RUL'].values, dtype=torch.float32, device=device)
    c_train = torch.tensor(train_df['censored'].values, dtype=torch.float32, device=device)
    
    batch_size = 262144
    
    X_val = torch.tensor(val_df[features].values, dtype=torch.float32, device=device)
    y_val = torch.tensor(val_df['log_RUL'].values, dtype=torch.float32, device=device)
    c_val = torch.tensor(val_df['censored'].values, dtype=torch.float32, device=device)
    
    model = MLP(input_dim=len(features)).to(device)
    criterion = LogNormalAFTLoss(failure_weight=14.0).to(device)
    
    # Use distinct learning rates: slower learning rate for variance parameter to prevent explosion
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 1e-3},
        {'params': criterion.parameters(), 'lr': 1e-4}
    ])
    
    epochs = 300
    
    print("Training MLP model with custom right-censored loss...")
    for epoch in range(epochs):
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        total_mse = torch.tensor(0.0, device=device)
        total_mae = torch.tensor(0.0, device=device)
        total_uncensored = torch.tensor(0.0, device=device)
        total_y_sum = torch.tensor(0.0, device=device)
        total_y_sq_sum = torch.tensor(0.0, device=device)
        indices = torch.randperm(len(X_train), device=device)
        num_batches = math.ceil(len(X_train) / batch_size)
        
        train_pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for i in train_pbar:
            batch_idx = indices[i * batch_size : (i + 1) * batch_size]
            batch_x = X_train[batch_idx]
            batch_y = y_train[batch_idx]
            batch_c = c_train[batch_idx]
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y, batch_c)
            loss.backward()
            optimizer.step()
            
            # calculate metrics ONLY for failed (uncensored) data
            preds_flat = preds.view(-1)
            y_flat = batch_y.view(-1)
            c_flat = batch_c.view(-1)
            
            total_loss += loss.detach() * batch_x.size(0)
            
            preds_clipped = torch.clamp(preds_flat, min=-2.0, max=9.0)
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
        
        num_val_batches = math.ceil(len(X_val) / batch_size)
        val_pbar = tqdm(range(num_val_batches), desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for i in val_pbar:
                batch_x = X_val[i * batch_size : (i + 1) * batch_size]
                batch_y = y_val[i * batch_size : (i + 1) * batch_size]
                batch_c = c_val[i * batch_size : (i + 1) * batch_size]
                
                preds = model(batch_x)
                loss = criterion(preds, batch_y, batch_c)
                
                preds_flat = preds.view(-1)
                y_flat = batch_y.view(-1)
                c_flat = batch_c.view(-1)
                
                val_loss += loss.detach() * batch_x.size(0)
                
                preds_clipped = torch.clamp(preds_flat, min=-2.0, max=9.0)
                preds_orig = torch.expm1(preds_clipped)
                y_orig = torch.expm1(y_flat)
                
                uncensored_float = (1.0 - c_flat)
                diff_u = (preds_orig - y_orig) * uncensored_float
                
                val_mse += torch.sum(diff_u ** 2)
                val_mae += torch.sum(torch.abs(diff_u))
                val_y_sum += torch.sum(y_orig * uncensored_float)
                val_y_sq_sum += torch.sum((y_orig ** 2) * uncensored_float)
                val_uncensored += torch.sum(uncensored_float)
                
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
        
    print("Training finished.")

if __name__ == "__main__":
    main()
