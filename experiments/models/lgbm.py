import numpy as np
import lightgbm as lgb

# Model Hyperparameters
HYPERPARAMS = {
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'min_data_in_leaf': 100,
    'num_boost_round': 200,
    'early_stopping_rounds': 20
}

def train_lgbm_model(X_train, y_train, X_val=None, y_val=None, seed: int = 42, is_cost_sensitive: bool = False, use_gpu: bool = False, custom_params: dict = None):
    """
    Trains LightGBM classifier for 30-day binary failure classification.
    Uses exact scale_pos_weight (N_neg / N_pos) when cost-sensitive learning is requested.
    """
    hp = HYPERPARAMS.copy()
    if custom_params:
        hp.update(custom_params)

    if is_cost_sensitive:
        n_pos = np.sum(y_train == 1)
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"Training LightGBM (Cost-Sensitive scale_pos_weight={scale_pos_weight:.2f}, num_boost_round={hp['num_boost_round']})...")
    else:
        scale_pos_weight = 1.0
        print(f"Training LightGBM (Unweighted, num_boost_round={hp['num_boost_round']})...")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data) if (X_val is not None and y_val is not None) else None
    
    lgb_params = {
        'learning_rate': hp['learning_rate'],
        'num_leaves': hp['num_leaves'],
        'verbose': hp['verbose'],
        'objective': hp['objective'],
        'metric': hp['metric'],
        'scale_pos_weight': scale_pos_weight,
        'seed': seed,
        'device': 'gpu' if use_gpu else 'cpu',
        'min_data_in_leaf': hp['min_data_in_leaf']
    }
    
    valid_sets = [train_data]
    if val_data is not None:
        valid_sets.append(val_data)
        
    callbacks = [lgb.early_stopping(stopping_rounds=hp['early_stopping_rounds'], verbose=False)] if val_data is not None else []
    
    try:
        gbm = lgb.train(lgb_params, train_data, num_boost_round=hp['num_boost_round'], valid_sets=valid_sets, callbacks=callbacks)
    except lgb.basic.LightGBMError:
        print("[Notice] LightGBM GPU Learner fallback to CPU...")
        lgb_params['device'] = 'cpu'
        gbm = lgb.train(lgb_params, train_data, num_boost_round=hp['num_boost_round'], valid_sets=valid_sets, callbacks=callbacks)
        
    print("LightGBM training complete.")
    return gbm
