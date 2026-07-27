import numpy as np
from xgboost import XGBClassifier

# Standard Benchmark Hyperparameters (국룰값)
HYPERPARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "tree_method": 'hist',
    "eval_metric": 'logloss',
    "early_stopping_rounds": 20
}

def train_xgb_model(X_train, y_train, X_val=None, y_val=None, seed: int = 42, is_cost_sensitive: bool = False, custom_params: dict = None):
    """
    Trains XGBoost classifier with early stopping on validation set.
    """
    hp = HYPERPARAMS.copy()
    if custom_params:
        hp.update(custom_params)

    if is_cost_sensitive:
        n_pos = np.sum(y_train == 1)
        n_neg = len(y_train) - n_pos
        scale_pos_weight = float(np.sqrt(n_neg / n_pos)) if n_pos > 0 else 1.0
        print(f"Training XGBoost (Cost-Sensitive sqrt scale_pos_weight={scale_pos_weight:.2f}, n_estimators={hp['n_estimators']}, early_stopping={hp['early_stopping_rounds']})...")
    else:
        scale_pos_weight = 1.0
        print(f"Training XGBoost (Unweighted, n_estimators={hp['n_estimators']}, early_stopping={hp['early_stopping_rounds']})...")

    has_val = (X_val is not None and y_val is not None)
    
    xgb = XGBClassifier(
        n_estimators=hp['n_estimators'],
        learning_rate=hp['learning_rate'],
        max_depth=hp['max_depth'],
        scale_pos_weight=scale_pos_weight,
        tree_method=hp['tree_method'],
        device='cuda',
        random_state=seed,
        eval_metric=hp['eval_metric'],
        early_stopping_rounds=hp['early_stopping_rounds'] if has_val else None
    )
    
    eval_set = [(X_train, y_train)]
    if has_val:
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
            eval_metric=hp['eval_metric'],
            early_stopping_rounds=hp['early_stopping_rounds'] if has_val else None
        )
        xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    return xgb
