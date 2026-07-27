import numpy as np
try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    NearestNeighbors = None

import config

def postprocess_synthetic_samples(X_resampled, X_original):
    """
    Universally valid, domain-agnostic post-processing for synthetic samples.
    Enforces feature values to stay within the empirical feature bounds of the original dataset:
    [min(X_original_j), max(X_original_j)] for each feature j.
    
    💡 Universal Advantages:
    - If raw SMART features (>= 0): min is >= 0, automatically preserving non-negativity.
    - If Z-score standardized features (mean 0, std 1): min can be negative (-3.5 etc.), correctly preserving negative range!
    - 100% generalized for ANY dataset, feature scale, or domain schema without any hardcoded thresholds.
    """
    if hasattr(X_resampled, 'values'):
        X_res_np = X_resampled.values.copy()
    else:
        X_res_np = np.array(X_resampled, copy=True)

    if hasattr(X_original, 'values'):
        X_orig_np = X_original.values
    else:
        X_orig_np = np.array(X_original)

    # Dynamically compute feature-wise min and max from the original training dataset
    min_bounds = np.nanmin(X_orig_np, axis=0)
    max_bounds = np.nanmax(X_orig_np, axis=0)

    # Clip synthetic features dynamically to stay within the empirical bounds of original data
    X_res_np = np.clip(X_res_np, min_bounds, max_bounds)
    X_res_np = np.nan_to_num(X_res_np, nan=0.0, posinf=0.0, neginf=0.0)

    return X_res_np


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
    """Strategy 1: Raw data without any resampling."""
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
        sampled_neg_idx = rng.choice(neg_idx, size=min(len(neg_idx), n_neg_sample), replace=False)
        
        selected_idx = np.concatenate([pos_idx, sampled_neg_idx])
        rng.shuffle(selected_idx)
        
        print(f"[Imbalance: Undersampling] Resampled -> Positives: {len(pos_idx):,}, Negatives: {len(sampled_neg_idx):,}")
        
        if hasattr(X, 'iloc'):
            return X.iloc[selected_idx], y[selected_idx]
        elif hasattr(X, '__getitem__'):
            return X[selected_idx], y[selected_idx]
        else:
            return X[selected_idx], y[selected_idx]

class OversamplingHandler(BaseImbalanceHandler):
    """Strategy 2b: Random Oversampling of minority class."""
    def __init__(self, ratio: float = 1.0, seed: int = None):
        super().__init__(seed=seed)
        self.ratio = ratio

    def process(self, X, y):
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_neg = len(neg_idx)
        n_pos_needed = int(n_neg * self.ratio)
        
        rng = np.random.default_rng(self.seed)
        oversampled_pos_idx = rng.choice(pos_idx, size=n_pos_needed, replace=True)
        
        selected_idx = np.concatenate([oversampled_pos_idx, neg_idx])
        rng.shuffle(selected_idx)
        
        print(f"[Imbalance: Oversampling] Resampled -> Positives: {len(oversampled_pos_idx):,}, Negatives: {n_neg:,}")
        
        if hasattr(X, 'iloc'):
            return X.iloc[selected_idx], y[selected_idx]
        elif hasattr(X, '__getitem__'):
            return X[selected_idx], y[selected_idx]
        else:
            return X[selected_idx], y[selected_idx]

# ------------------------------------------------------------------------------
# 3. SMOTE (Chawla et al., 2002) with Universal Feature Bound Post-Processing
# ------------------------------------------------------------------------------
class SMOTEHandler(BaseImbalanceHandler):
    """Strategy 3: SMOTE with universal empirical feature bound clipping."""
    def __init__(self, k_neighbors: int = 5, max_synthetic_samples: int = 200000, seed: int = None):
        super().__init__(seed=seed)
        self.k_neighbors = k_neighbors
        self.max_synthetic_samples = max_synthetic_samples

    def _smote_numpy(self, X_pos, n_synthetic_needed):
        n_pos, n_features = X_pos.shape
        k = max(1, min(self.k_neighbors, n_pos - 1))
        
        print(f"[SMOTE] Generating {n_synthetic_needed:,} synthetic minority samples using k={k} nearest neighbors...")
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
        nn.fit(X_pos)
        knn_indices = nn.kneighbors(X_pos, return_distance=False)[:, 1:]
        
        rng = np.random.default_rng(self.seed)
        synthetic_samples = np.zeros((n_synthetic_needed, n_features), dtype=np.float32)
        
        for i in range(n_synthetic_needed):
            base_idx = rng.integers(0, n_pos)
            neighbor_col = rng.integers(0, k)
            neighbor_idx = knn_indices[base_idx, neighbor_col]
            
            diff = X_pos[neighbor_idx] - X_pos[base_idx]
            gap = rng.uniform(0, 1)
            synthetic_samples[i] = X_pos[base_idx] + gap * diff
            
        return synthetic_samples

    def process(self, X, y):
        try:
            from imblearn.over_sampling import SMOTE
            print(f"[Imbalance: SMOTE] Applying imbalanced-learn SMOTE...")
            smote = SMOTE(k_neighbors=self.k_neighbors, random_state=self.seed)
            res_X, res_y = smote.fit_resample(X, y)
            res_X = postprocess_synthetic_samples(res_X, X)
            return res_X, res_y
        except Exception:
            if hasattr(X, 'values'):
                X_np = X.values
            else:
                X_np = np.array(X)
            y_np = np.array(y)
            
            pos_idx = np.where(y_np == 1)[0]
            neg_idx = np.where(y_np == 0)[0]
            n_pos = len(pos_idx)
            n_neg = len(neg_idx)
            
            if n_pos == 0 or n_neg == 0:
                return X, y
                
            n_synthetic_needed = min(n_neg - n_pos, self.max_synthetic_samples)
            if n_synthetic_needed <= 0:
                return X, y
                
            if n_neg > (n_pos + n_synthetic_needed):
                rng = np.random.default_rng(self.seed)
                neg_sampled_idx = rng.choice(neg_idx, size=(n_pos + n_synthetic_needed), replace=False)
            else:
                neg_sampled_idx = neg_idx
                
            X_pos = X_np[pos_idx]
            synthetic_X = self._smote_numpy(X_pos, n_synthetic_needed)
            synthetic_y = np.ones(n_synthetic_needed, dtype=y_np.dtype)
            
            resampled_X = np.vstack([X_np[pos_idx], X_np[neg_sampled_idx], synthetic_X])
            resampled_y = np.concatenate([y_np[pos_idx], y_np[neg_sampled_idx], synthetic_y])
            
            # Universal Empirical Feature Bound Post-Processing
            resampled_X = postprocess_synthetic_samples(resampled_X, X_np)
            
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(len(resampled_y))
            
            print(f"[Imbalance: SMOTE Complete] Resampled counts -> Positives: {n_pos + n_synthetic_needed:,}, Negatives: {len(neg_sampled_idx):,}")
            return resampled_X[perm], resampled_y[perm]

# ------------------------------------------------------------------------------
# 4. ADASYN (He et al., 2008) with Universal Feature Bound Post-Processing
# ------------------------------------------------------------------------------
class ADASYNHandler(BaseImbalanceHandler):
    """Strategy 4: ADASYN with universal empirical feature bound clipping."""
    def __init__(self, k_neighbors: int = 5, max_synthetic_samples: int = 200000, seed: int = None):
        super().__init__(seed=seed)
        self.k_neighbors = k_neighbors
        self.max_synthetic_samples = max_synthetic_samples

    def process(self, X, y):
        try:
            from imblearn.over_sampling import ADASYN
            print(f"[Imbalance: ADASYN] Applying imbalanced-learn ADASYN...")
            adasyn = ADASYN(n_neighbors=self.k_neighbors, random_state=self.seed)
            res_X, res_y = adasyn.fit_resample(X, y)
            res_X = postprocess_synthetic_samples(res_X, X)
            return res_X, res_y
        except Exception:
            print(f"[Imbalance: ADASYN] Applying custom ADASYN sampler...")
            if hasattr(X, 'values'):
                X_np = X.values
            else:
                X_np = np.array(X)
            y_np = np.array(y)
            
            pos_idx = np.where(y_np == 1)[0]
            neg_idx = np.where(y_np == 0)[0]
            n_pos, n_neg = len(pos_idx), len(neg_idx)
            
            if n_pos <= self.k_neighbors:
                print("[ADASYN] Too few positives for k-NN, falling back to SMOTE.")
                return SMOTEHandler(seed=self.seed).process(X, y)
                
            n_synthetic_total = min(n_neg - n_pos, self.max_synthetic_samples)
            if n_synthetic_total <= 0:
                return X, y

            nn = NearestNeighbors(n_neighbors=self.k_neighbors + 1, n_jobs=-1)
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
                gaps = rng.uniform(0, 1, size=(g[i], X_np.shape[1]))
                synthetic_list.append(X_pos[i] + gaps * diff)
                
            if len(synthetic_list) > 0:
                synthetic_X = np.vstack(synthetic_list)
                synthetic_y = np.ones(len(synthetic_X), dtype=y_np.dtype)
                resampled_X = np.vstack([X_np, synthetic_X])
                resampled_y = np.concatenate([y_np, synthetic_y])
                
                # Universal Empirical Feature Bound Post-Processing
                resampled_X = postprocess_synthetic_samples(resampled_X, X_np)
                
                perm = rng.permutation(len(resampled_y))
                print(f"[Imbalance: ADASYN Complete] Generated {len(synthetic_X):,} adaptive samples.")
                return resampled_X[perm], resampled_y[perm]
            return X, y

# ------------------------------------------------------------------------------
# 5. EasyEnsemble (Liu et al., 2009)
# ------------------------------------------------------------------------------
class EasyEnsembleHandler(BaseImbalanceHandler):
    """Strategy 5: EasyEnsemble bagging sampler."""
    def __init__(self, n_estimators: int = 10, seed: int = None):
        super().__init__(seed=seed)
        self.n_estimators = n_estimators

    def generate_subsets(self, X, y):
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_pos = len(pos_idx)
        
        subsets = []
        for i in range(self.n_estimators):
            sub_rng = np.random.default_rng(self.seed + i)
            sampled_neg_idx = sub_rng.choice(neg_idx, size=n_pos, replace=False)
            sub_idx = np.concatenate([pos_idx, sampled_neg_idx])
            sub_rng.shuffle(sub_idx)
            
            if hasattr(X, 'iloc'):
                subsets.append((X.iloc[sub_idx], y[sub_idx]))
            else:
                subsets.append((X[sub_idx], y[sub_idx]))
        print(f"[Imbalance: EasyEnsemble] Generated {self.n_estimators} balanced sub-datasets (K={self.n_estimators}).")
        return subsets

    def process(self, X, y):
        return UndersamplingHandler(ratio=1.0, seed=self.seed).process(X, y)

# ------------------------------------------------------------------------------
# 6. Cost-Sensitive Learning (Elkan, 2001) & Focal Loss (Lin et al., 2017)
# ------------------------------------------------------------------------------
class CostSensitiveHandler(BaseImbalanceHandler):
    """Strategy 6: Cost-Sensitive Loss Weighting."""
    def process(self, X, y):
        pos_count = int(np.sum(y == 1))
        neg_count = len(y) - pos_count
        ratio = neg_count / pos_count if pos_count > 0 else 1.0
        print(f"[Imbalance: Cost-Sensitive] Raw data passed with loss scale_pos_weight: {ratio:.2f}")
        return X, y

class FocalLossHandler(BaseImbalanceHandler):
    """Strategy 7: Focal Loss."""
    def process(self, X, y):
        print("[Imbalance: Focal Loss] Raw data passed. Focal loss objective will be applied in neural net training.")
        return X, y


# ==============================================================================
# 🚨 CONFIG COMPATIBILITY VALIDATION CHECK
# ==============================================================================
SUPPORTED_STRATEGIES = ['none', 'undersampling', 'oversampling', 'smote', 'adasyn', 'easyensemble', 'cost_sensitive', 'focal_loss']

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
            f"💡 Recommended options for '{model_name.upper()}': 'cost_sensitive' (scale_pos_weight/class_weight), "
            f"'undersampling', 'easyensemble', 'smote', or 'adasyn'."
        )


def apply_imbalance_treatment(X, y, strategy: str = 'none', seed: int = None, n_estimators: int = 10):
    """
    Modular factory function to apply imbalance handling.
    """
    strategy = strategy.lower().strip()
    if seed is None:
        seed = config.SEED

    if strategy == 'none':
        return NoneImbalanceHandler(seed=seed).process(X, y)
    elif strategy == 'undersampling':
        return UndersamplingHandler(seed=seed).process(X, y)
    elif strategy == 'oversampling':
        return OversamplingHandler(seed=seed).process(X, y)
    elif strategy == 'smote':
        return SMOTEHandler(seed=seed).process(X, y)
    elif strategy == 'adasyn':
        return ADASYNHandler(seed=seed).process(X, y)
    elif strategy == 'easyensemble':
        return EasyEnsembleHandler(n_estimators=n_estimators, seed=seed).process(X, y)
    elif strategy == 'cost_sensitive':
        return CostSensitiveHandler(seed=seed).process(X, y)
    elif strategy == 'focal_loss':
        return FocalLossHandler(seed=seed).process(X, y)
    else:
        raise ValueError(f"Unknown imbalance strategy '{strategy}'. Supported: {SUPPORTED_STRATEGIES}")
