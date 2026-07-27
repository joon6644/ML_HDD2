import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Unified Deep Learning Hyperparameters (딥러닝 일관 수치)
HYPERPARAMS = {
    "hidden_dim1": 64,
    "hidden_dim2": 32,
    "dropout": 0.2,
    "epochs": 30,
    "batch_size": 16384,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 5,
    "focal_gamma": 2.0,
    "focal_alpha": 0.25
}

class BinaryFocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) for imbalanced binary classification."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1.0 - p_t) ** self.gamma
        if self.alpha >= 0:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_loss = alpha_t * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss
        return focal_loss.mean()

class MLPClass(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout: float = 0.2):
        super(MLPClass, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def train_mlp_model(X_train, y_train, X_val=None, y_val=None, seed: int = 42, is_cost_sensitive: bool = False, use_focal_loss: bool = False, custom_params: dict = None):
    """
    Trains PyTorch Multi-Layer Perceptron (MLP) with Dropout (0.2) and early stopping on validation set.
    """
    hp = HYPERPARAMS.copy()
    if custom_params:
        hp.update(custom_params)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not isinstance(X_train, torch.Tensor):
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
    else:
        X_train_t, y_train_t = X_train, y_train

    has_val = (X_val is not None and y_val is not None)
    if has_val:
        if not isinstance(X_val, torch.Tensor):
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.float32)
        else:
            X_val_t, y_val_t = X_val, y_val

    if is_cost_sensitive:
        n_pos = torch.sum(y_train_t == 1).item()
        n_neg = len(y_train_t) - n_pos
        scale_pos_weight = math.sqrt(n_neg / n_pos) if n_pos > 0 else 1.0
        pos_weight_tensor = torch.tensor([scale_pos_weight], dtype=torch.float32, device=device)
        print(f"Training MLP (Cost-Sensitive sqrt pos_weight={scale_pos_weight:.2f}, dropout={hp['dropout']}, max_epochs={hp['epochs']}, patience={hp['patience']}) on device: {device}...")
    else:
        pos_weight_tensor = None
        print(f"Training MLP (Unweighted, FocalLoss={use_focal_loss}, dropout={hp['dropout']}, max_epochs={hp['epochs']}, patience={hp['patience']}) on device: {device}...")

    input_dim = X_train_t.shape[1]
    model = MLPClass(input_dim=input_dim, hidden_dim1=hp['hidden_dim1'], hidden_dim2=hp['hidden_dim2'], dropout=hp['dropout']).to(device)
    
    if use_focal_loss:
        criterion = BinaryFocalLoss(alpha=hp['focal_alpha'], gamma=hp['focal_gamma']).to(device)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    
    n_samples = len(X_train_t)
    batch_size = hp['batch_size']
    epochs = hp['epochs']
    num_batches = math.ceil(n_samples / batch_size)
    
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0
    patience = hp['patience']
    
    print(f"Training MLP: {n_samples:,} samples over {num_batches} batches/epoch...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        perm = torch.randperm(n_samples)
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for i in pbar:
            idx = perm[i * batch_size : (i + 1) * batch_size]
            batch_x = X_train_t[idx].to(device)
            batch_y = y_train_t[idx].to(device).view(-1, 1)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
            
        train_loss = total_loss / n_samples
        
        # Validation & Early Stopping Check
        if has_val:
            model.eval()
            with torch.no_grad():
                val_logits = []
                val_batch_size = hp['batch_size']
                for vi in range(0, len(X_val_t), val_batch_size):
                    vx = X_val_t[vi:vi+val_batch_size].to(device)
                    val_logits.append(model(vx))
                val_logits_t = torch.cat(val_logits)
                val_loss = criterion(val_logits_t, y_val_t.to(device).view(-1, 1)).item()
                
            print(f"  Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[Early Stopping] Triggered at epoch {epoch+1}. Restoring best model weights (Val Loss: {best_val_loss:.4f}).")
                    break
        else:
            print(f"  Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}")

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    print("MLP training complete.")
    return model
