import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .common import BinaryFocalLoss, set_torch_seed, COMMON_HYPERPARAMS, compute_sqrt_scale_pos_weight

# Unified Deep Learning Hyperparameters (shared keys sourced from common.COMMON_HYPERPARAMS)
HYPERPARAMS = {
    **COMMON_HYPERPARAMS,
    "hidden_dim1": 64,
    "hidden_dim2": 32,
}


class MLPClass(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout: float = 0.2):
        super(MLPClass, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout = dropout
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


def _to_float32_tensor(arr):
    """Safely convert any array-like (Tensor / DataFrame / ndarray) to float32 CPU Tensor."""
    if isinstance(arr, torch.Tensor):
        return arr.to(torch.float32)
    elif hasattr(arr, 'values'):
        return torch.from_numpy(np.asarray(arr.values, dtype=np.float32))
    else:
        return torch.from_numpy(np.asarray(arr, dtype=np.float32))


def train_mlp_model(X_train, y_train, X_val=None, y_val=None, seed: int = 42,
                    is_cost_sensitive: bool = False, use_focal_loss: bool = False,
                    custom_params: dict = None):
    """
    Trains PyTorch MLP with Dropout and early stopping on validation set.
    - Full GPU reproducibility via set_torch_seed (CUDA + cudnn deterministic)
    - Train data pre-loaded to GPU for maximum GPU utilization
    """
    hp = HYPERPARAMS.copy()
    if custom_params:
        hp.update(custom_params)

    set_torch_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Safe type-aware conversion for all input types
    X_train_t = _to_float32_tensor(X_train)
    y_train_t = _to_float32_tensor(y_train)

    has_val = (X_val is not None and y_val is not None)
    if has_val:
        X_val_t = _to_float32_tensor(X_val).to(device)   # Val: pre-load to GPU
        y_val_device = _to_float32_tensor(y_val).to(device).view(-1, 1)

    if is_cost_sensitive:
        scale_pos_weight = compute_sqrt_scale_pos_weight(y_train_t)
        pos_weight_tensor = torch.tensor([scale_pos_weight], dtype=torch.float32, device=device)
        print(f"Training MLP (Cost-Sensitive sqrt pos_weight={scale_pos_weight:.2f}, "
              f"dropout={hp['dropout']}, max_epochs={hp['epochs']}, patience={hp['patience']}) on device: {device}...")
    else:
        pos_weight_tensor = None
        print(f"Training MLP (Unweighted, FocalLoss={use_focal_loss}, "
              f"dropout={hp['dropout']}, max_epochs={hp['epochs']}, patience={hp['patience']}) on device: {device}...")

    input_dim = X_train_t.shape[1]
    model = MLPClass(input_dim=input_dim, hidden_dim1=hp['hidden_dim1'],
                     hidden_dim2=hp['hidden_dim2'], dropout=hp['dropout']).to(device)

    criterion = (BinaryFocalLoss(alpha=hp['focal_alpha'], gamma=hp['focal_gamma']).to(device)
                 if use_focal_loss else
                 nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device))
    optimizer = optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])

    n_samples = len(X_train_t)
    batch_size = hp['batch_size']
    epochs = hp['epochs']
    num_batches = math.ceil(n_samples / batch_size)

    # Pre-load train data to GPU only if it fits in VRAM (< 4GB), otherwise stream batch-by-batch
    estimated_size_gb = (X_train_t.element_size() * X_train_t.nelement()) / (1024 ** 3)
    if estimated_size_gb < 4.0:
        X_train_gpu = X_train_t.to(device)
        y_train_gpu = y_train_t.to(device)
        is_preloaded = True
    else:
        is_preloaded = False

    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0
    patience = hp['patience']

    print(f"Training MLP: {n_samples:,} samples over {num_batches} batches/epoch (GPU pre-loaded: {is_preloaded})...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        perm = torch.randperm(n_samples)

        for i in range(num_batches):
            idx = perm[i * batch_size: (i + 1) * batch_size]
            if is_preloaded:
                batch_x = X_train_gpu[idx]
                batch_y = y_train_gpu[idx].view(-1, 1)
            else:
                batch_x = X_train_t[idx].to(device, non_blocking=True)
                batch_y = y_train_t[idx].view(-1, 1).to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        train_loss = total_loss / n_samples

        # Validation & Early Stopping
        if has_val:
            model.eval()
            with torch.no_grad():
                val_logits = torch.cat([
                    model(X_val_t[vi: vi + 65536])
                    for vi in range(0, len(X_val_t), 65536)
                ])
                val_loss = criterion(val_logits, y_val_device).item()

            print(f"  Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            min_delta = 1e-5
            if val_loss < (best_val_loss - min_delta):
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[Early Stopping] Triggered at epoch {epoch+1}. "
                          f"Restoring best model weights (Val Loss: {best_val_loss:.6f}).")
                    break
        else:
            print(f"  Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}")

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    print("MLP training complete.")
    return model
