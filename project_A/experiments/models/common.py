"""
Shared utilities for all model training modules.
- BinaryFocalLoss: single source of truth (avoid duplication across lstm/gru/mlp)
- set_torch_seed: full deterministic seed setup for GPU reproducibility
- COMMON_HYPERPARAMS: training hyperparameters shared by gru/lstm/mlp, kept in one
  place so a fair cross-model comparison can't silently drift out of sync.
- compute_sqrt_scale_pos_weight: single source of truth for the sqrt(N_neg/N_pos)
  cost-sensitive weighting formula used identically by all six models.
"""
import math
import numpy as np
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    class nn:
        Module = object


# Training hyperparameters shared across all three PyTorch models (gru/lstm/mlp).
# Each model file merges its own architecture-specific keys (hidden_dim, num_layers, ...) on top.
COMMON_HYPERPARAMS = {
    "dropout": 0.2,
    "epochs": 30,
    "batch_size": 16384,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 5,
    "focal_gamma": 2.0,
    "focal_alpha": 0.75
}


def compute_sqrt_scale_pos_weight(y) -> float:
    """Cost-sensitive weight sqrt(N_neg / N_pos), shared formula across all six models."""
    if isinstance(y, torch.Tensor):
        n_pos = int((y == 1).sum().item())
        n_total = int(y.shape[0])
    else:
        y_arr = np.asarray(y)
        n_pos = int(np.sum(y_arr == 1))
        n_total = int(len(y_arr))
    n_neg = n_total - n_pos
    return float(math.sqrt(n_neg / n_pos)) if n_pos > 0 else 1.0


def set_torch_seed(seed: int):
    """Set all PyTorch seeds for full GPU reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
