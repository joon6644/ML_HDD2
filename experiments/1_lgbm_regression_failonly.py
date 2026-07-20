import lightgbm as lgb
import numpy as np
from data_loader import get_data

import scipy.special as special

def custom_aft_objective(preds, train_data):
    labels = train_data.get_label()
    censored = train_data.get_weight()
    
    sigma = 0.5
    z = (labels - preds) / sigma
    
    # Uncensored
    grad_uncensored = -z / sigma
    hess_uncensored = np.ones_like(preds) / (sigma ** 2)
    
    # Only calculate censored values if censored data exists
    if np.any(censored == 1):
        # Safe z for large z approximation to avoid division by zero
        safe_z_large = np.where(z > 5, z, 5.0)
        h_z_large = safe_z_large + 1.0 / safe_z_large
        
        # Safe survival for standard calculation using fast math
        pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        surv = np.clip(1.0 - special.ndtr(z), 1e-15, 1.0)
        h_z_small = pdf / surv
        
        h_z = np.where(z > 5, h_z_large, h_z_small)
        
        grad_censored = -h_z / sigma
        # Ensure Hessian is strictly positive to prevent LightGBM leaf explosion
        hess_censored = np.maximum(h_z * (h_z - z) / (sigma ** 2), 1e-4)
        
        grad = np.where(censored == 1, grad_censored, grad_uncensored)
        hess = np.where(censored == 1, hess_censored, hess_uncensored)
    else:
        grad = grad_uncensored
        hess = hess_uncensored
        
    return grad, hess

def aft_eval(preds, train_data):
    labels = train_data.get_label()
    censored = train_data.get_weight()
    
    sigma = 0.5
    z = (labels - preds) / sigma
    
    loss_uncensored = 0.5 * (z ** 2) + np.log(sigma) + 0.5 * np.log(2*np.pi)
    
    if np.any(censored == 1):
        # Safe z for large z approx
        safe_z_large = np.where(z > 5, z, 5.0)
        loss_large = 0.5 * (safe_z_large**2) + 0.5*np.log(2*np.pi) + np.log(safe_z_large)
        
        # Safe survival for small z using fast math
        cdf = special.ndtr(z)
        surv = np.clip(1.0 - cdf, 1e-15, 1.0)
        loss_small = -np.log(surv)
        
        loss_censored = np.where(z > 5, loss_large, loss_small)
        loss = np.where(censored == 1, loss_censored, loss_uncensored)
    else:
        loss = loss_uncensored
    
    # Calculate RUL metrics for uncensored samples
    preds_clipped = np.clip(preds, -20, 15)
    preds_orig = np.expm1(preds_clipped)
    labels_orig = np.expm1(labels)
    
    uncensored_mask = (censored == 0)
    if np.any(uncensored_mask):
        diff_u = preds_orig[uncensored_mask] - labels_orig[uncensored_mask]
        mse = np.mean(diff_u ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(diff_u))
        
        y_u = labels_orig[uncensored_mask]
        ss_res = np.sum(diff_u ** 2)
        ss_tot = np.sum((y_u - np.mean(y_u)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        mse, rmse, mae, r2 = 0.0, 0.0, 0.0, 0.0
        
    return [
        ('aft_loss', np.mean(loss), False),
        ('mse', mse, False),
        ('rmse', rmse, False),
        ('mae', mae, False),
        ('r2', r2, True)
    ]

def custom_log_evaluation(period=10):
    def callback(env):
        if period > 0 and (env.iteration + 1) % period == 0:
            results = {}
            for dataset_name, metric_name, value, is_higher_better in env.evaluation_result_list:
                if dataset_name not in results:
                    results[dataset_name] = []
                results[dataset_name].append((metric_name, value))
            
            print(f"Iteration {env.iteration + 1}")
            for dataset_name, metrics in results.items():
                name_map = {'training': 'Train', 'valid_1': 'Val'}
                disp_name = name_map.get(dataset_name, dataset_name)
                
                metric_strs = []
                for m_name, val in metrics:
                    disp_m_name = 'Loss' if m_name == 'aft_loss' else m_name.upper()
                    metric_strs.append(f"{disp_m_name}: {val:.4f}")
                print(f"  {disp_name:5s} -> " + " | ".join(metric_strs))
    return callback

def main():
    train_df, val_df, features = get_data()
    
    # Only train on failure data
    train_df = train_df[train_df['censored'] == 0].copy()
    
    # Convert RUL to Log scale for AFT model
    train_df['log_RUL'] = np.log1p(train_df['RUL'])
    val_df['log_RUL'] = np.log1p(val_df['RUL'])
    
    print(f"Train shapes: features {train_df[features].shape}, target {train_df['log_RUL'].shape}")
    
    # LightGBM dataset
    # We pass 'censored' as weights so we can access it inside the custom objective
    train_data = lgb.Dataset(
        train_df[features], 
        label=train_df['log_RUL'], 
        weight=train_df['censored']
    )
    
    # Validation set (10%): used for early stopping / monitoring during training
    np.random.seed(42)
    val_sample_idx = np.random.choice(len(val_df), size=min(500000, len(val_df)), replace=False)
    val_sample_df = val_df.iloc[val_sample_idx]

    val_data = lgb.Dataset(
        val_sample_df[features],
        label=val_sample_df['log_RUL'],
        weight=val_sample_df['censored'],
        reference=train_data
    )
    
    params = {
        'learning_rate': 0.02,
        'num_leaves': 31,
        'verbose': 1,
        'objective': custom_aft_objective
    }
    
    # 텐서 변환 완료 후 대용량 pandas DataFrame들을 즉시 지우고 가비지 컬렉터 강제 구동
    import gc
    del train_df, val_df, val_sample_df
    gc.collect()

    params = {
        'learning_rate': 0.02,
        'num_leaves': 31,
        'verbose': 1,
        'objective': custom_aft_objective
    }
    
    from data_loader import get_lgbm_callback
    print("Training LightGBM Log-Normal AFT model...")
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=270,
        valid_sets=[train_data, val_data],
        feval=aft_eval,
        callbacks=[
            custom_log_evaluation(period=10),
            get_lgbm_callback("1_lgbm_regression_failonly")
        ]
    )
    
    print("Training finished.")

if __name__ == "__main__":
    main()
