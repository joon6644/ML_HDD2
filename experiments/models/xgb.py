import numpy as np
from xgboost import XGBClassifier
from xgboost.core import XGBoostError
from .common import compute_sqrt_scale_pos_weight

# Standard Benchmark Hyperparameters
HYPERPARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "tree_method": 'hist',
    "eval_metric": 'logloss',
    "early_stopping_rounds": 20
}


def train_xgb_model(X_train, y_train, X_val=None, y_val=None,
                    seed: int = 42, is_cost_sensitive: bool = False,
                    use_gpu: bool = True, custom_params: dict = None):
    """
    Trains XGBoost classifier with early stopping on validation set.
    - eval_set only contains val data (NOT train) to keep early stopping uncontaminated.
    - scale_pos_weight uses sqrt(N_neg / N_pos) consistent with other models.
    - use_gpu mirrors LightGBM's use_gpu flag for symmetric GPU/CPU resource parity;
      falls back to CPU automatically on GPU-related failure.
    """
    hp = HYPERPARAMS.copy()
    if custom_params:
        hp.update(custom_params)

    if is_cost_sensitive:
        scale_pos_weight = compute_sqrt_scale_pos_weight(y_train)
        print(f"Training XGBoost (Cost-Sensitive sqrt scale_pos_weight={scale_pos_weight:.2f}, "
              f"n_estimators={hp['n_estimators']}, early_stopping={hp['early_stopping_rounds']})...")
    else:
        scale_pos_weight = 1.0
        print(f"Training XGBoost (Unweighted, n_estimators={hp['n_estimators']}, "
              f"early_stopping={hp['early_stopping_rounds']})...")

    has_val = (X_val is not None and y_val is not None)

    xgb = XGBClassifier(
        n_estimators=hp['n_estimators'],
        learning_rate=hp['learning_rate'],
        max_depth=hp['max_depth'],
        scale_pos_weight=scale_pos_weight,
        tree_method=hp['tree_method'],
        device='cuda' if use_gpu else 'cpu',
        random_state=seed,
        eval_metric=hp['eval_metric'],
        early_stopping_rounds=hp['early_stopping_rounds'] if has_val else None
    )

    # FIX: eval_set must NOT include train data when val is available.
    # Early stopping in XGBoost uses the LAST entry of eval_set; including train
    # would add noise and inflate memory usage unnecessarily.
    eval_set = [(X_val, y_val)] if has_val else [(X_train, y_train)]

    try:
        xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    except XGBoostError as e:
        print(f"[Notice] XGBoost GPU fallback to CPU due to error: {e}")
        xgb = XGBClassifier(
            n_estimators=hp['n_estimators'],
            learning_rate=hp['learning_rate'],
            max_depth=hp['max_depth'],
            scale_pos_weight=scale_pos_weight,
            tree_method=hp['tree_method'],
            device='cpu',
            random_state=seed,
            eval_metric=hp['eval_metric'],
            early_stopping_rounds=hp['early_stopping_rounds'] if has_val else None
        )
        xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    return xgb
