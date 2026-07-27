import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Model Hyperparameters
HYPERPARAMS = {
    "n_estimators": 100,
    "max_depth": 12,
    "n_jobs": -1
}

def train_rf_model(X_train, y_train, seed: int = 42, is_cost_sensitive: bool = False, custom_params: dict = None):
    """
    Trains a Random Forest classifier for 30-day failure binary classification.
    Uses exact scale_pos_weight (N_neg / N_pos) when cost-sensitive learning is requested.
    """
    hp = HYPERPARAMS.copy()
    if custom_params:
        hp.update(custom_params)

    if is_cost_sensitive:
        n_pos = np.sum(y_train == 1)
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        class_weight = {0: 1.0, 1: scale_pos_weight}
        print(f"Training Random Forest (Cost-Sensitive scale_pos_weight={scale_pos_weight:.2f}, n_estimators={hp['n_estimators']})...")
    else:
        class_weight = None
        print(f"Training Random Forest (Unweighted, n_estimators={hp['n_estimators']})...")

    rf = RandomForestClassifier(
        n_estimators=hp['n_estimators'],
        max_depth=hp['max_depth'],
        class_weight=class_weight,
        random_state=seed,
        n_jobs=hp['n_jobs']
    )
    
    rf.fit(X_train, y_train)
    print("Random Forest training complete.")
    return rf
