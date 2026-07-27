import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Model Hyperparameters
HYPERPARAMS = {
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "epochs": 6,
    "batch_size": 16384,
    "lr": 1e-3,
    "weight_decay": 1e-5,
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

class GRUClass(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(GRUClass, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        return self.fc(last_out)

def train_gru_model(X_train, y_train, X_val=None, y_val=None, seed: int = 42, use_focal_loss: bool = False, custom_params: dict = None):
    """
    Trains PyTorch GRU 3D sequence model for 30-day binary failure classification.
    """
    hp = HYPERPARAMS.copy()
    if custom_params:
        hp.update(custom_params)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training GRU Classifier (FocalLoss={use_focal_loss}) on device: {device}...")
    
    sample_seq = X_train[0]
    input_dim = sample_seq.shape[-1]
    
    model = GRUClass(input_dim=input_dim, hidden_dim=hp['hidden_dim'], num_layers=hp['num_layers'], dropout=hp['dropout']).to(device)
    
    if use_focal_loss:
        criterion = BinaryFocalLoss(alpha=hp['focal_alpha'], gamma=hp['focal_gamma']).to(device)
    else:
        criterion = nn.BCEWithLogitsLoss().to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    
    n_samples = len(X_train)
    batch_size = hp['batch_size']
    epochs = hp['epochs']
    
    pos_idx = torch.where(y_train == 1)[0]
    neg_idx = torch.where(y_train == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    pos_ratio = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0.5
    
    n_pos_per_batch = max(1, round(batch_size * pos_ratio))
    n_neg_per_batch = batch_size - n_pos_per_batch
    num_batches = n_neg // n_neg_per_batch if n_neg_per_batch > 0 else 1
    
    print(f"Training GRU: {n_samples:,} sequences over {num_batches} batches/epoch...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pos_perm = pos_idx[torch.randperm(n_pos)]
        neg_perm = neg_idx[torch.randperm(n_neg)]
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for i in pbar:
            neg_start = i * n_neg_per_batch
            batch_neg_idx = neg_perm[neg_start : neg_start + n_neg_per_batch]
            
            pos_start = (i * n_pos_per_batch) % n_pos
            pos_end = pos_start + n_pos_per_batch
            if pos_end <= n_pos:
                batch_pos_idx = pos_perm[pos_start:pos_end]
            else:
                batch_pos_idx = torch.cat([pos_perm[pos_start:], pos_perm[:pos_end - n_pos]])
                
            batch_idx = torch.cat([batch_pos_idx, batch_neg_idx])
            batch_idx = batch_idx[torch.randperm(len(batch_idx))]
            
            batch_x = X_train[batch_idx].to(device)
            batch_y = y_train[batch_idx].to(device).view(-1, 1)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
            
        print(f"  Epoch [{epoch+1}/{epochs}] Loss: {total_loss / (num_batches * batch_size):.4f}")
        
    print("GRU training complete.")
    return model
