import os
import numpy as np
try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    NearestNeighbors = None

try:
    import torch
except ImportError:
    torch = None

import config

def postprocess_synthetic_samples(X_resampled, X_original):
    """
    Universally valid, domain-agnostic post-processing for synthetic samples.
    Enforces feature values to stay within the empirical feature bounds of the original dataset:
    [min(X_original_j), max(X_original_j)] for each feature j.
    """
    if hasattr(X_resampled, 'values'):
        X_res_np = X_resampled.values.copy()
    else:
        X_res_np = np.array(X_resampled, copy=True)

    if hasattr(X_original, 'values'):
        X_orig_np = X_original.values
    else:
        X_orig_np = np.array(X_original)

    min_bounds = np.nanmin(X_orig_np, axis=0)
    max_bounds = np.nanmax(X_orig_np, axis=0)

    X_res_np = np.clip(X_res_np, min_bounds, max_bounds)
    X_res_np = np.nan_to_num(X_res_np, nan=0.0, posinf=0.0, neginf=0.0)

    return X_res_np


class EasyEnsembleModelWrapper:
    """
    Bagging Ensemble Wrapper for EasyEnsemble (Liu et al., 2009).
    Averages predicted probabilities (Soft Voting) across K independent base estimators.
    """
    def __init__(self, models, is_pytorch=False, device='cpu'):
        self.models = models
        self.is_pytorch = is_pytorch
        self.device = torch.device(device) if (torch is not None and isinstance(device, (str, torch.device))) else None

    def predict_proba(self, X):
        probs_list = []
        for model in self.models:
            if torch is not None and (isinstance(model, torch.nn.Module) or self.is_pytorch):
                model.eval()
                with torch.no_grad():
                    if not isinstance(X, torch.Tensor):
                        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
                    else:
                        X_t = X.to(self.device)
                    p = torch.sigmoid(model(X_t)).cpu().numpy().flatten()
            elif hasattr(model, 'predict_proba'):
                p = model.predict_proba(X)[:, 1]
            elif hasattr(model, 'predict'):
                p = model.predict(X)
                if p.ndim > 1:
                    p = p[:, 1]
            else:
                p = model(X)
            probs_list.append(p)
        avg_prob = np.mean(probs_list, axis=0)
        return np.column_stack([1.0 - avg_prob, avg_prob])

    def predict(self, X):
        return self.predict_proba(X)[:, 1]

    def eval(self):
        for m in self.models:
            if hasattr(m, 'eval'):
                m.eval()

    def to(self, device):
        self.device = torch.device(device)
        for m in self.models:
            if hasattr(m, 'to'):
                m.to(device)
        return self

    def __call__(self, X):
        return self.predict(X)


class BaseImbalanceHandler:
    """Base class interface for imbalance handling strategies."""
    def __init__(self, seed: int = None):
        self.seed = seed if seed is not None else config.SEED

    def process(self, X, y):
        raise NotImplementedError

# ------------------------------------------------------------------------------
# 1. Raw Dataset (None)
# ------------------------------------------------------------------------------
class NoneImbalanceHandler(BaseImbalanceHandler):
    """Strategy 1: Raw data without any resampling or class weight."""
    def process(self, X, y):
        pos_count = int(np.sum(y == 1))
        total_count = len(y)
        print(f"[Imbalance: None] Using raw dataset (Pos: {pos_count:,} / Total: {total_count:,} | Pos ratio: {pos_count/total_count:.4%})")
        return X, y

# ------------------------------------------------------------------------------
# 2. Random Undersampling & Oversampling (Kubat & Matwin, 1997)
# ------------------------------------------------------------------------------
class UndersamplingHandler(BaseImbalanceHandler):
    """Strategy 2a: Random Undersampling of majority class."""
    def __init__(self, ratio: float = 1.0, seed: int = None):
        super().__init__(seed=seed)
        self.ratio = ratio

    def process(self, X, y):
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_pos = len(pos_idx)
        n_neg_sample = int(n_pos * self.ratio)
        
        rng = np.random.default_rng(self.seed)
        sampled_neg_idx = rng.choice(neg_idx, size=n_neg_sample, replace=False)
        
        resampled_idx = np.concatenate([pos_idx, sampled_neg_idx])
        rng.shuffle(resampled_idx)
        
        if hasattr(X, 'iloc'):
            resampled_X = X.iloc[resampled_idx]
        else:
            resampled_X = X[resampled_idx]
        resampled_y = y[resampled_idx]
        
        print(f"[Imbalance: Undersampling] Resampled shape: {resampled_X.shape} (Pos: {n_pos:,}, Neg: {n_neg_sample:,})")
        return resampled_X, resampled_y


class OversamplingHandler(BaseImbalanceHandler):
    """Strategy 2b: Random Oversampling of minority class."""
    def __init__(self, ratio: float = 1.0, seed: int = None):
        super().__init__(seed=seed)
        self.ratio = ratio

    def process(self, X, y):
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_neg = len(neg_idx)
        n_pos_target = int(n_neg * self.ratio)
        
        rng = np.random.default_rng(self.seed)
        oversampled_pos_idx = rng.choice(pos_idx, size=n_pos_target, replace=True)
        
        resampled_idx = np.concatenate([neg_idx, oversampled_pos_idx])
        rng.shuffle(resampled_idx)
        
        if hasattr(X, 'iloc'):
            resampled_X = X.iloc[resampled_idx]
        else:
            resampled_X = X[resampled_idx]
        resampled_y = y[resampled_idx]
        
        print(f"[Imbalance: Oversampling] Resampled shape: {resampled_X.shape} (Pos: {n_pos_target:,}, Neg: {n_neg:,})")
        return resampled_X, resampled_y

# ------------------------------------------------------------------------------
# 3. SMOTE (Chawla et al., 2002) - High Performance Vectorized
# ------------------------------------------------------------------------------
class SMOTEHandler(BaseImbalanceHandler):
    """Strategy 3: Synthetic Minority Over-sampling Technique (SMOTE)."""
    def __init__(self, k_neighbors: int = 5, seed: int = None):
        super().__init__(seed=seed)
        self.k_neighbors = k_neighbors

    def process(self, X, y):
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)
        
        if n_pos < 2:
            print("[Warning] Too few positive samples for SMOTE. Returning raw data.")
            return X, y

        X_np = X.values if hasattr(X, 'values') else np.array(X, dtype=np.float32)
        y_np = y.values if hasattr(y, 'values') else np.array(y)
        
        X_pos = X_np[pos_idx]
        n_synthetic_needed = n_neg - n_pos
        
        k = min(self.k_neighbors, n_pos - 1)
        if NearestNeighbors is None:
            raise ImportError("scikit-learn is required for SMOTE. Please install via pip install scikit-learn.")
            
        print(f"[Imbalance: SMOTE] Fitting k-NN (k={k}) on {n_pos:,} positive minority samples...")
        nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=-1)
        nn.fit(X_pos)
        knn_indices = nn.kneighbors(X_pos, return_distance=False)[:, 1:]
        
        rng = np.random.default_rng(self.seed)
        print(f"[Imbalance: SMOTE] Vectorized generation of {n_synthetic_needed:,} synthetic samples...")
        
        idx_array = rng.integers(0, n_pos, size=n_synthetic_needed)
        neighbor_choices = rng.integers(0, k, size=n_synthetic_needed)
        neighbor_idx_array = knn_indices[idx_array, neighbor_choices]
        
        gaps = rng.uniform(0, 1, size=(n_synthetic_needed, X_pos.shape[1])).astype(np.float32)
        diffs = X_pos[neighbor_idx_array] - X_pos[idx_array]
        synthetic_X = X_pos[idx_array] + gaps * diffs
        synthetic_y = np.ones(len(synthetic_X), dtype=y_np.dtype)
        
        resampled_X = np.vstack([X_np, synthetic_X])
        resampled_y = np.concatenate([y_np, synthetic_y])
        
        resampled_X = postprocess_synthetic_samples(resampled_X, X_np)
        
        perm = rng.permutation(len(resampled_y))
        print(f"[Imbalance: SMOTE Complete] Generated {len(synthetic_X):,} synthetic samples (Total: {len(resampled_y):,}).")
        return resampled_X[perm], resampled_y[perm]

# ------------------------------------------------------------------------------
# 4. ADASYN (He et al., 2008) - High Performance Vectorized
# ------------------------------------------------------------------------------
class ADASYNHandler(BaseImbalanceHandler):
    """Strategy 4: Adaptive Synthetic Sampling Approach (ADASYN)."""
    def __init__(self, k_neighbors: int = 5, seed: int = None):
        super().__init__(seed=seed)
        self.k_neighbors = k_neighbors

    def process(self, X, y):
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)
        
        if n_pos < 2:
            print("[Warning] Too few positive samples for ADASYN. Returning raw data.")
            return X, y
            
        X_np = X.values if hasattr(X, 'values') else np.array(X, dtype=np.float32)
        y_np = y.values if hasattr(y, 'values') else np.array(y)
        
        n_synthetic_total = n_neg - n_pos
        k = min(self.k_neighbors, len(y_np) - 1)
        
        if NearestNeighbors is None:
            raise ImportError("scikit-learn is required for ADASYN. Please install via pip install scikit-learn.")
            
        print(f"[Imbalance: ADASYN] Fitting k-NN (k={k}) on {n_pos:,} positive minority samples...")
        nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=-1)
        nn.fit(X_np)
        knn_indices = nn.kneighbors(X_np[pos_idx], return_distance=False)[:, 1:]
        
        r = np.zeros(n_pos, dtype=np.float32)
        for i in range(n_pos):
            neighbors_y = y_np[knn_indices[i]]
            r[i] = np.sum(neighbors_y == 0) / self.k_neighbors
            
        r_sum = np.sum(r)
        if r_sum == 0:
            r_hat = np.full(n_pos, 1.0 / n_pos)
        else:
            r_hat = r / r_sum
            
        g = np.round(r_hat * n_synthetic_total).astype(int)
        synthetic_list = []
        rng = np.random.default_rng(self.seed)
        
        X_pos = X_np[pos_idx]
        for i in range(n_pos):
            if g[i] == 0:
                continue
            pos_neighbors = [idx for idx in knn_indices[i] if y_np[idx] == 1]
            if len(pos_neighbors) == 0:
                target_neighbor_idx = pos_idx[rng.integers(0, n_pos)]
            else:
                target_neighbor_idx = rng.choice(pos_neighbors)
                
            diff = X_np[target_neighbor_idx] - X_pos[i]
            gaps = rng.uniform(0, 1, size=(g[i], X_np.shape[1])).astype(np.float32)
            synthetic_list.append(X_pos[i] + gaps * diff)
            
        if len(synthetic_list) > 0:
            synthetic_X = np.vstack(synthetic_list)
            synthetic_y = np.ones(len(synthetic_X), dtype=y_np.dtype)
            resampled_X = np.vstack([X_np, synthetic_X])
            resampled_y = np.concatenate([y_np, synthetic_y])
            
            resampled_X = postprocess_synthetic_samples(resampled_X, X_np)
            
            perm = rng.permutation(len(resampled_y))
            print(f"[Imbalance: ADASYN Complete] Generated {len(synthetic_X):,} adaptive samples.")
            return resampled_X[perm], resampled_y[perm]
        return X, y

# ------------------------------------------------------------------------------
# 5. EasyEnsemble (Liu et al., 2009)
# ------------------------------------------------------------------------------
class EasyEnsembleHandler(BaseImbalanceHandler):
    """Strategy 5: EasyEnsemble bagging sampler (Liu et al., 2009)."""
    def __init__(self, n_estimators: int = 10, seed: int = None):
        super().__init__(seed=seed)
        self.n_estimators = n_estimators

    def generate_subsets(self, X, y):
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_pos = len(pos_idx)
        
        X_np = X.values if hasattr(X, 'values') else np.array(X)
        y_np = y.values if hasattr(y, 'values') else np.array(y)

        subsets = []
        for i in range(self.n_estimators):
            sub_rng = np.random.default_rng(self.seed + i * 997)
            sampled_neg_idx = sub_rng.choice(neg_idx, size=n_pos, replace=False)
            sub_idx = np.concatenate([pos_idx, sampled_neg_idx])
            sub_rng.shuffle(sub_idx)
            subsets.append((X_np[sub_idx], y_np[sub_idx]))
            
        print(f"[Imbalance: EasyEnsemble] Generated {self.n_estimators} independent balanced subsets (K={self.n_estimators}, Pos: {n_pos:,}, Neg: {n_pos:,} per subset).")
        return subsets

    def process(self, X, y):
        return self.generate_subsets(X, y)[0]

# ------------------------------------------------------------------------------
# 6. Class Weight (Elkan, 2001) & Focal Loss (Lin et al., 2017)
# ------------------------------------------------------------------------------
class ClassWeightHandler(BaseImbalanceHandler):
    """Strategy 6: Square-Root Dynamic Class Weighting (sqrt(N_neg / N_pos))."""
    def process(self, X, y):
        pos_count = int(np.sum(y == 1))
        neg_count = len(y) - pos_count
        raw_ratio = neg_count / pos_count if pos_count > 0 else 1.0
        sqrt_ratio = float(np.sqrt(raw_ratio))
        print(f"[Imbalance: Class Weight] Square-root class weight applied (Raw: {raw_ratio:.2f} -> Sqrt: {sqrt_ratio:.2f})")
        return X, y

class FocalLossHandler(BaseImbalanceHandler):
    """Strategy 7: Focal Loss."""
    def process(self, X, y):
        print("[Imbalance: Focal Loss] Raw data passed. Focal loss objective will be applied in neural net training.")
        return X, y


# ==============================================================================
# 🚨 CONFIG COMPATIBILITY VALIDATION CHECK & DISK CACHING
# ==============================================================================
SUPPORTED_STRATEGIES = ['none', 'class_weight', 'undersampling', 'oversampling', 'smote', 'adasyn', 'easyensemble', 'focal_loss']
HEAVY_RESAMPLING_STRATEGIES = ['smote', 'adasyn', 'oversampling', 'undersampling']

def validate_config_compatibility(model_name: str, strategy: str):
    """
    Validates model and imbalance strategy compatibility.
    Raises ValueError with explanatory message if combination is incompatible.
    """
    model_name = model_name.lower().strip()
    strategy = strategy.lower().strip()
    
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unsupported imbalance strategy '{strategy}'. "
            f"Available strategies: {SUPPORTED_STRATEGIES}"
        )
        
    if strategy == 'focal_loss' and model_name not in ['mlp', 'lstm', 'gru']:
        raise ValueError(
            f"❌ [Incompatible Configuration] Strategy 'focal_loss' is a custom PyTorch loss function, "
            f"which cannot be applied directly to tree-based model '{model_name.upper()}'.\n"
            f"💡 Recommended options for '{model_name.upper()}': 'class_weight' (scale_pos_weight/class_weight), "
            f"'undersampling', 'easyensemble', 'smote', or 'adasyn'."
        )


def apply_imbalance_treatment(X, y, strategy: str = 'none', seed: int = None, n_estimators: int = 10, dataset_path: str = None):
    """
    Modular factory function to apply imbalance handling with automatic Disk Caching.
    Lookup & Save Rule: data/resampled_cache/{dataset_name}_{strategy}_seed{seed}.npz
    """
    strategy = strategy.lower().strip()
    if seed is None:
        seed = config.SEED

    # 1. Non-resampling strategies return raw inputs directly
    if strategy == 'none':
        return NoneImbalanceHandler(seed=seed).process(X, y)
    elif strategy == 'class_weight':
        return ClassWeightHandler(seed=seed).process(X, y)
    elif strategy == 'focal_loss':
        return FocalLossHandler(seed=seed).process(X, y)
    elif strategy == 'easyensemble':
        return EasyEnsembleHandler(n_estimators=n_estimators, seed=seed).process(X, y)

    # Extract dataset name (e.g., ST12000NM0007_seed1234)
    if dataset_path is None:
        dataset_path = config.DATASET_DIR
    dataset_name = os.path.basename(os.path.normpath(dataset_path))

    # 2. Check Disk Cache for Heavy Resampling Strategies
    # Rule: {dataset_name}_{strategy}_seed{seed}.npz
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(project_root, "data", "resampled_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_filename = f"{dataset_name}_{strategy}_seed{seed}.npz"
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path):
        print(f"\n[Imbalance Cache Hit] Loading pre-computed resampled dataset from cache:")
        print(f"  -> {cache_path}")
        try:
            cached_data = np.load(cache_path)
            X_resampled = cached_data['X']
            y_resampled = cached_data['y']
            print(f"  -> Cache loaded instantly! Shape: {X_resampled.shape} (Pos: {np.sum(y_resampled==1):,}, Neg: {np.sum(y_resampled==0):,})\n")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"⚠️ [Corrupted Cache Warning] Cached file is incomplete or corrupted ({e}). Deleting corrupted cache and recomputing...")
            try:
                os.remove(cache_path)
            except Exception:
                pass

    # 3. Compute Resampling if Cache Miss
    print(f"\n[Imbalance Cache Miss] Computing '{strategy.upper()}' resampling for dataset '{dataset_name}'...")
    if strategy == 'undersampling':
        X_resampled, y_resampled = UndersamplingHandler(seed=seed).process(X, y)
    elif strategy == 'oversampling':
        X_resampled, y_resampled = OversamplingHandler(seed=seed).process(X, y)
    elif strategy == 'smote':
        X_resampled, y_resampled = SMOTEHandler(seed=seed).process(X, y)
    elif strategy == 'adasyn':
        X_resampled, y_resampled = ADASYNHandler(seed=seed).process(X, y)
    else:
        raise ValueError(f"Unknown imbalance strategy '{strategy}'. Supported: {SUPPORTED_STRATEGIES}")

    # Convert to NumPy for saving
    X_res_np = X_resampled.values if hasattr(X_resampled, 'values') else np.array(X_resampled)
    y_res_np = y_resampled.values if hasattr(y_resampled, 'values') else np.array(y_resampled)

    # Save to Cache (Fast Uncompressed savez for 10x write speed)
    print(f"[Imbalance Cache] Saving generated resampled dataset to cache:")
    print(f"  -> {cache_path}")
    np.savez(cache_path, X=X_res_np, y=y_res_np)
    print("  -> Disk cache save complete! Future runs with this strategy will load instantly in <1 second.\n")

    return X_res_np, y_res_np
