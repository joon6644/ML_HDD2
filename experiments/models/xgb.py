import numpy as np
from xgboost import XGBClassifier

# Model Hyperparameters
HYPERPARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 6,
    "tree_method": 'hist',
    "eval_metric": 'logloss'
}

def train_xgb_model(X_train, y_train, X_val=None, y_val=None, seed: int = 42, is_cost_sensitive: bool = False, custom_params: dict = None):
    """
    Trains XGBoost classifier for 30-day binary failure classification.
    Uses exact scale_pos_weight (N_neg / N_pos) when cost-sensitive learning is requested.
    """
    hp = HYPERPARAMS.copy()
    if custom_params:
        hp.update(custom_params)

    if is_cost_sensitive:
        n_pos = np.sum(y_train == 1)
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"Training XGBoost (Cost-Sensitive scale_pos_weight={scale_pos_weight:.2f}, n_estimators={hp['n_estimators']})...")
    else:
        scale_pos_weight = 1.0
        print(f"Training XGBoost (Unweighted, n_estimators={hp['n_estimators']})...")

    xgb = XGBClassifier(
        n_estimators=hp['n_estimators'],
        learning_rate=hp['learning_rate'],
        max_depth=hp['max_depth'],
        scale_pos_weight=scale_pos_weight,
        tree_method=hp['tree_method'],
        device='cuda',
        random_state=seed,
        eval_metric=hp['eval_metric']
    )
    
    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))
        
    try:
        xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    except Exception as e:
        print(f"[Notice] XGBoost GPU fallback to CPU due to error: {e}")
        xgb = XGBClassifier(
            n_estimators=hp['n_estimators'],
            learning_rate=hp['learning_rate'],
            max_depth=hp['max_depth'],
            scale_pos_weight=scale_pos_weight,
            tree_method=hp['tree_method'],
            device='cpu',
            random_state=seed,
            eval_metric=hp['eval_metric']
        )
        xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        
    print("XGBoost training complete.")
    return xgb
