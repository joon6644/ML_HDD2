import numpy as np
import lightgbm as lgb
from .common import compute_sqrt_scale_pos_weight

# Standard Benchmark Hyperparameters
HYPERPARAMS = {
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
    'objective': 'binary',
    # Early-stopping monitor only (does NOT change the training loss, which stays
    # binary cross-entropy via 'objective' above). Under severe class imbalance,
    # binary_logloss plateaus almost immediately because predicting near the prior
    # already near-minimizes it, triggering early stopping long before the model
    # has actually learned to separate the minority class. average_precision tracks
    # minority-class ranking directly, so it doesn't false-plateau the same way.
    'metric': 'average_precision',
    'min_data_in_leaf': 100,
    # Stochastic regularization, matching the level applied to the other models
    # (XGBoost: subsample 0.9 / colsample_bytree 0.9, LSTM & GRU: dropout 0.2).
    # Both libraries default these to 1.0, so leaving LightGBM at the default while
    # XGBoost subsamples would regularize only one of the two GBDTs.
    # bagging_freq must be >= 1 or LightGBM ignores bagging_fraction entirely.
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'feature_fraction': 0.9,
    'num_boost_round': 300,
    'early_stopping_rounds': 20
}


def train_lgbm_model(X_train, y_train, X_val=None, y_val=None,
                     seed: int = 42, is_cost_sensitive: bool = False,
                     use_gpu: bool = False, custom_params: dict = None):
    """
    Trains LightGBM classifier with early stopping on validation set.
    - valid_sets only contains val data (NOT train) to keep early stopping uncontaminated.
    - scale_pos_weight uses sqrt(N_neg / N_pos) consistent with other models.
    """
    hp = HYPERPARAMS.copy()
    if custom_params:
        hp.update(custom_params)

    if is_cost_sensitive:
        scale_pos_weight = compute_sqrt_scale_pos_weight(y_train)
        print(f"Training LightGBM (Cost-Sensitive sqrt scale_pos_weight={scale_pos_weight:.2f}, "
              f"max_round={hp['num_boost_round']}, early_stopping={hp['early_stopping_rounds']})...")
    else:
        scale_pos_weight = 1.0
        print(f"Training LightGBM (Unweighted, max_round={hp['num_boost_round']}, "
              f"early_stopping={hp['early_stopping_rounds']})...")

    train_data = lgb.Dataset(X_train, label=y_train)
    has_val = (X_val is not None and y_val is not None)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data) if has_val else None

    lgb_params = {
        'learning_rate': hp['learning_rate'],
        'num_leaves': hp['num_leaves'],
        'verbose': hp['verbose'],
        'objective': hp['objective'],
        'metric': hp['metric'],
        'scale_pos_weight': scale_pos_weight,
        'seed': seed,
        'device': 'gpu' if use_gpu else 'cpu',
        'min_data_in_leaf': hp['min_data_in_leaf'],
        'bagging_fraction': hp['bagging_fraction'],
        'bagging_freq': hp['bagging_freq'],
        'feature_fraction': hp['feature_fraction']
    }

    # FIX: valid_sets must NOT include train_data when val is available.
    # Using [train_data] as the only set when val is absent (no early stopping in that case).
    if has_val:
        valid_sets = [val_data]          # val only → uncontaminated early stopping
        callbacks = [lgb.early_stopping(stopping_rounds=hp['early_stopping_rounds'], verbose=False)]
    else:
        valid_sets = [train_data]        # no val: train loss only, no early stopping
        callbacks = []

    # No GPU->CPU fallback: LightGBM's GPU and CPU learners do not produce bit-identical
    # trees, so silently switching devices mid-batch would make some seeds irreproducible
    # while every result still looks uniform. Fail and let the operator set USE_GPU=False.
    gbm = lgb.train(lgb_params, train_data,
                    num_boost_round=hp['num_boost_round'],
                    valid_sets=valid_sets,
                    callbacks=callbacks)

    return gbm
