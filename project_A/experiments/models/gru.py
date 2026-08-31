import copy
import math
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
    class nn:
        Module = object
    optim = None
from .common import BinaryFocalLoss, set_torch_seed, COMMON_HYPERPARAMS, compute_sqrt_scale_pos_weight

# Unified Deep Learning Hyperparameters (shared keys sourced from common.COMMON_HYPERPARAMS)
HYPERPARAMS = {
    **COMMON_HYPERPARAMS,
    "hidden_dim": 64,
    "num_layers": 2,
}


class GRUClass(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(GRUClass, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
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


def _to_float32_tensor(arr):
    """Safely convert any array-like to float32 CPU Tensor.
    Handles: torch.Tensor, LazySequenceTensor, pandas DataFrame/Series, numpy ndarray.
    """
    if isinstance(arr, torch.Tensor):
        return arr.to(torch.float32)
    elif hasattr(arr, 'valid_indices') and hasattr(arr, 'X_raw'):
        # LazySequenceTensor: materialize via __getitem__ over the full valid index range
        idx = torch.arange(len(arr), dtype=torch.long)
        return arr[idx].to(torch.float32)
    elif hasattr(arr, 'values'):
        return torch.from_numpy(np.asarray(arr.values, dtype=np.float32))
    else:
        return torch.from_numpy(np.asarray(arr, dtype=np.float32))


def train_gru_model(X_train, y_train, X_val=None, y_val=None, seed: int = 42,
                    is_cost_sensitive: bool = False, use_focal_loss: bool = False,
                    custom_params: dict = None):
    """
    Trains PyTorch GRU 3D sequence model with early stopping on validation set.
    - Full GPU reproducibility via set_torch_seed (CUDA + cudnn deterministic)
    - Train data pre-loaded to GPU for maximum GPU utilization
    """
    hp = HYPERPARAMS.copy()
    if custom_params:
        hp.update(custom_params)

    set_torch_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Safely handle sequence input: Do NOT materialize 66.8GB LazySequenceTensor to RAM
    is_lazy_train = hasattr(X_train, 'valid_indices') and hasattr(X_train, 'X_raw')
    if is_lazy_train:
        X_train_t = X_train
    else:
        X_train_t = _to_float32_tensor(X_train)

    y_train_t = _to_float32_tensor(y_train)

    has_val = (X_val is not None and y_val is not None)
    if has_val:
        is_lazy_val = hasattr(X_val, 'valid_indices') and hasattr(X_val, 'X_raw')
        if is_lazy_val:
            X_val_t = X_val
        else:
            X_val_t = _to_float32_tensor(X_val)
        y_val_device = _to_float32_tensor(y_val).to(device).view(-1, 1)

    if is_cost_sensitive:
        scale_pos_weight = compute_sqrt_scale_pos_weight(y_train_t)
        pos_weight_tensor = torch.tensor([scale_pos_weight], dtype=torch.float32, device=device)
        print(f"Training GRU (Cost-Sensitive sqrt pos_weight={scale_pos_weight:.2f}, "
              f"max_epochs={hp['epochs']}, patience={hp['patience']}) on device: {device}...")
    else:
        pos_weight_tensor = None
        print(f"Training GRU (Unweighted, FocalLoss={use_focal_loss}, "
              f"max_epochs={hp['epochs']}, patience={hp['patience']}) on device: {device}...")

    input_dim = X_train_t.shape[-1] if hasattr(X_train_t, 'shape') else X_train_t.X_raw.shape[1]
    model = GRUClass(input_dim=input_dim, hidden_dim=hp['hidden_dim'],
                     num_layers=hp['num_layers'], dropout=hp['dropout']).to(device)

    criterion = (BinaryFocalLoss(alpha=hp['focal_alpha'], gamma=hp['focal_gamma']).to(device)
                 if use_focal_loss else
                 nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device))
    optimizer = optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])

    n_samples = len(X_train_t)
    batch_size = hp['batch_size']
    epochs = hp['epochs']
    num_batches = math.ceil(n_samples / batch_size)

    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0
    patience = hp['patience']

    print(f"Training GRU: {n_samples:,} sequences over {num_batches} batches/epoch...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        perm = torch.randperm(n_samples)

        for i in range(num_batches):
            idx = perm[i * batch_size: (i + 1) * batch_size]
            batch_x = X_train_t[idx].to(device)
            batch_y = y_train_t[idx].view(-1, 1).to(device)

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
                val_batches = []
                val_bs = 65536
                for vi in range(0, len(X_val_t), val_bs):
                    vx = X_val_t[vi: vi + val_bs].to(device)
                    val_batches.append(model(vx))
                val_logits = torch.cat(val_batches)
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

    print("GRU training complete.")
    return model
