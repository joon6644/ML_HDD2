import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from .common import compute_sqrt_scale_pos_weight

# Model Hyperparameters
HYPERPARAMS = {
    "n_estimators": 100,
    "max_depth": 12,
    "n_jobs": -1,
    "oob_score": False,
}


def train_rf_model(X_train, y_train, X_val=None, y_val=None,
                   seed: int = 42, is_cost_sensitive: bool = False,
                   custom_params: dict = None):
    """
    Trains a Random Forest classifier for 30-day failure binary classification.
    - Val set is used to report a real Val AUROC after training for comparability with other models.
    - scale_pos_weight uses sqrt(N_neg / N_pos) consistent with LGBM / XGB / DL models.
    """
    hp = HYPERPARAMS.copy()
    if custom_params:
        hp.update(custom_params)

    if is_cost_sensitive:
        scale_pos_weight = compute_sqrt_scale_pos_weight(y_train)
        # class_weight dict mirrors sklearn API: same sqrt formula as other models
        class_weight = {0: 1.0, 1: scale_pos_weight}
        print(f"Training Random Forest (Cost-Sensitive sqrt scale_pos_weight={scale_pos_weight:.2f}, "
              f"n_estimators={hp['n_estimators']})...")
    else:
        class_weight = None
        print(f"Training Random Forest (Unweighted, n_estimators={hp['n_estimators']})...")

    rf = RandomForestClassifier(
        n_estimators=hp['n_estimators'],
        max_depth=hp['max_depth'],
        class_weight=class_weight,
        oob_score=hp['oob_score'],
        random_state=seed,
        n_jobs=hp['n_jobs']
    )

    rf.fit(X_train, y_train)

    val_auc_str = ""
    if X_val is not None and y_val is not None:
        val_probs = rf.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        val_auc_str = f"Val AUROC: {val_auc:.4f}"

    print(f"Random Forest training complete. {val_auc_str}")
    return rf
