import os
import sys
import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None
from tqdm import tqdm

class RollingEvaluator:
    def __init__(self, model, features, window_size=28, device='cpu', model_type='pytorch_class', max_log_RUL=None, K=100):
        """
        Rolling Inference Evaluator for disk failure alert policies.
        
        Parameters:
        -----------
        model: Trained model object (PyTorch nn.Module or LightGBM Booster)
        features: List of feature column names.
        window_size: Sequence window size (use 1 for static/LightGBM models).
        device: Torch device ('cpu' or 'cuda').
        model_type: Type of model:
            - 'pytorch_class': Binary classifier outputting logit
            - 'pytorch_survival': DeepHit survival model outputting K-bin PMF
            - 'pytorch_regression': RUL regression model outputting RUL log mean (or multitask outputs)
            - 'lightgbm': LightGBM Booster
        max_log_RUL: Maximum log(RUL + 1) for DeepHit scaling (only needed for 'pytorch_survival')
        K: Number of DeepHit bins (only needed for 'pytorch_survival')
        """
        self.model = model
        self.features = features
        self.window_size = window_size
        self.device = torch.device(device) if torch is not None else None
        self.model_type = model_type
        self.max_log_RUL = max_log_RUL
        self.K = K
        
        # If it's a PyTorch model, move to device and set to evaluation mode
        if torch is not None and self.model_type.startswith('pytorch') and hasattr(self.model, 'eval'):
            self.model.to(self.device)
            self.model.eval()

    def predict_disk(self, disk_df):
        """
        Predict daily failure probability or RUL for a single disk.
        
        Returns:
        --------
        dates: Array of dates where a prediction was made.
        y_true: Daily true failure labels (1 if failed within 30 days, else 0).
        preds: Daily predicted failure probability (or RUL days).
        """
        df_sorted = disk_df.sort_values('date')
        n = len(df_sorted)
        
        if n == 0:
            return np.array([]), np.array([]), np.array([])
            
        # Get true daily labels (failed within 30 days)
        # censored == 0 means failed
        y_true = ((df_sorted['RUL'] <= 30) & (df_sorted['censored'] == 0)).astype(np.float32).values
        valid_dates = df_sorted['date'].values
        
        x_raw = df_sorted[self.features].values
        
        if self.model_type == 'lightgbm':
            # LightGBM takes single-row static features (2D)
            preds = self.model.predict(x_raw)
            return valid_dates, y_true, preds
            
        elif self.model_type == 'sklearn':
            # Scikit-learn models take single-row static features (2D)
            preds = self.model.predict_proba(x_raw)[:, 1]
            return valid_dates, y_true, preds

        # For PyTorch models, convert to tensor on the target device
        x_tensor = torch.tensor(x_raw, dtype=torch.float32, device=self.device)
        
        # Efficiently extract sliding windows using PyTorch unfold, padding the beginning of history if window_size > 1
        if self.window_size > 1:
            first_row = x_tensor[0:1]  # shape (1, feature_dim)
            padding = first_row.repeat(self.window_size - 1, 1)
            x_padded = torch.cat([padding, x_tensor], dim=0)
            x_batch = x_padded.unfold(0, self.window_size, 1).transpose(1, 2)
        else:
            x_batch = x_tensor
            
        with torch.no_grad():
            if self.model_type == 'pytorch_class':
                logits = self.model(x_batch)
                preds = torch.sigmoid(logits).view(-1).cpu().numpy()
                
            elif self.model_type == 'pytorch_survival':
                # DeepHit PMF output of shape (B, K)
                pmf = self.model(x_batch)
                
                # Probability of failure within 30 days: cumulative sum up to bin_30
                if self.max_log_RUL is None:
                    bin_30 = 30
                else:
                    log_30 = np.log1p(30)
                    bin_30 = min(int(log_30 / self.max_log_RUL * (self.K - 1)), self.K - 1)
                
                preds = pmf[:, :bin_30 + 1].sum(dim=1).cpu().numpy()
                
            elif self.model_type == 'pytorch_regression':
                # Can be multitask (mu, log_var, risk_logits) or single output
                outputs = self.model(x_batch)
                if isinstance(outputs, tuple):
                    # Multi-task GRU: check if a risk head is available
                    if len(outputs) >= 3:
                        risk_logits = outputs[2]
                        preds = torch.sigmoid(risk_logits).view(-1).cpu().numpy()
                    else:
                        mu = outputs[0]
                        pred_days = torch.expm1(torch.clamp(mu.view(-1), min=-2.0, max=9.0)).cpu().numpy()
                        preds = (pred_days <= 30).astype(np.float32)
                else:
                    # Single output regression (mu)
                    pred_days = torch.expm1(torch.clamp(outputs.view(-1), min=-2.0, max=9.0)).cpu().numpy()
                    preds = (pred_days <= 30).astype(np.float32)
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")
                
        return valid_dates, y_true, preds

    def get_raw_predictions(self, val_df, sample_size=None):
        """
        Run rolling inference on validation set and collect daily raw predictions.
        """
        serials = val_df['serial_number'].unique()
        if sample_size is not None and len(serials) > sample_size:
            np.random.seed(42)
            serials = np.random.choice(serials, size=sample_size, replace=False)
            
        print(f"Running rolling inference on {len(serials)} serials...")
        
        # Pre-filter and sort validation df
        val_df_filtered = val_df[val_df['serial_number'].isin(serials)].copy()
        val_df_filtered['date'] = pd.to_datetime(val_df_filtered['date'])
        val_df_sorted = val_df_filtered.sort_values(['serial_number', 'date'])
        grouped = val_df_sorted.groupby('serial_number')
        
        raw_preds = []
        
        for serial, group in tqdm(grouped, desc="Predicting disks"):
            has_failed = (group['censored'].iloc[0] == 0)
            failure_date = pd.to_datetime(group['date'].max()) if has_failed else None
            
            dates, y_true, preds = self.predict_disk(group)
            if len(preds) == 0:
                continue
                
            raw_preds.append({
                'serial_number': serial,
                'has_failed': has_failed,
                'failure_date': failure_date,
                'dates': dates,
                'preds': preds
            })
            
        return raw_preds

    def build_report_from_raw_predictions(self, raw_preds, threshold, lead_time=30):
        """
        Build detailed alarm report from raw predictions using a specific threshold.
        """
        records = []
        for disk in raw_preds:
            serial = disk['serial_number']
            has_failed = disk['has_failed']
            failure_date = disk['failure_date']
            dates = disk['dates']
            preds = disk['preds']
            
            max_score = float(preds.max())
            alarm_mask = (preds >= threshold)
            alarm_indices = np.where(alarm_mask)[0]
            
            alarm_triggered = len(alarm_indices) > 0
            first_alarm_date = None
            alarm_score = None
            days_to_failure = None
            
            is_hit = 0
            is_miss = 0
            is_false_alarm = 0
            is_correct_rejection = 0
            
            if alarm_triggered:
                first_alarm_idx = alarm_indices[0]
                first_alarm_date = pd.to_datetime(dates[first_alarm_idx])
                alarm_score = float(preds[first_alarm_idx])
                
                if has_failed:
                    days_to_failure = (failure_date - first_alarm_date).days
                    # Hit: Alarm within the specified lead time before failure
                    if 0 <= days_to_failure <= lead_time:
                        is_hit = 1
                    else:
                        is_miss = 1  # Alarm triggered, but too early or after failure
                else:
                    is_false_alarm = 1  # Healthy disk triggered alarm
            else:
                if has_failed:
                    is_miss = 1  # Failed disk, but no alarm was triggered at all
                else:
                    is_correct_rejection = 1  # Healthy disk, no alarm triggered
                    
            records.append({
                'serial_number': serial,
                'has_failed': 1 if has_failed else 0,
                'alarm_triggered': 1 if alarm_triggered else 0,
                'first_alarm_date': first_alarm_date,
                'actual_failure_date': failure_date,
                'days_to_failure_at_alarm': days_to_failure,
                'is_hit': is_hit,
                'is_false_alarm': is_false_alarm,
                'is_miss': is_miss,
                'is_correct_rejection': is_correct_rejection,
                'max_predicted_score': max_score,
                'alarm_predicted_score': alarm_score
            })
            
        return pd.DataFrame(records)

    def find_best_threshold(self, raw_preds, lead_time=30):
        """
        Find the threshold that maximizes F1-score on the raw predictions.
        """
        # Try thresholds from 0.01 to 0.99
        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1 = -1.0
        best_threshold = 0.5
        
        failed_disks = sum(1 for disk in raw_preds if disk['has_failed'])
        
        for thresh in thresholds:
            hits = 0
            false_alarms = 0
            
            for disk in raw_preds:
                has_failed = disk['has_failed']
                preds = disk['preds']
                
                alarm_mask = (preds >= thresh)
                if not np.any(alarm_mask):
                    continue
                    
                if has_failed:
                    first_alarm_idx = np.where(alarm_mask)[0][0]
                    first_alarm_date = pd.to_datetime(disk['dates'][first_alarm_idx])
                    days_to_failure = (disk['failure_date'] - first_alarm_date).days
                    if 0 <= days_to_failure <= lead_time:
                        hits += 1
                else:
                    false_alarms += 1
                    
            recall = hits / failed_disks if failed_disks > 0 else 0.0
            precision = hits / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
                
        return float(best_threshold), float(best_f1)

    def get_detailed_alarm_report(self, val_df, threshold=0.5, lead_time=30, sample_size=None):
        """
        Run rolling inference on validation set and collect detailed per-disk alarm statistics.
        Keeps backward compatibility with original method signature.
        """
        raw_preds = self.get_raw_predictions(val_df, sample_size=sample_size)
        return self.build_report_from_raw_predictions(raw_preds, threshold, lead_time)

    def evaluate_alarms(self, val_df, threshold='auto', lead_time=30, sample_size=None, output_csv_path=None):
        """
        Evaluate alarm policy metrics on validation set.
        """
        raw_preds = self.get_raw_predictions(val_df, sample_size=sample_size)
        
        if threshold == 'auto':
            print("Searching for the optimal threshold that maximizes F1-score...")
            best_thresh, best_f1 = self.find_best_threshold(raw_preds, lead_time=lead_time)
            print(f"Optimal threshold found: {best_thresh:.4f} (Max F1-Score: {best_f1:.4%})")
            target_threshold = best_thresh
        else:
            target_threshold = float(threshold)
            
        report_df = self.build_report_from_raw_predictions(raw_preds, target_threshold, lead_time)
        metrics = analyze_and_save_report(report_df, output_csv_path=output_csv_path, threshold=target_threshold)
        metrics['threshold'] = target_threshold
        return metrics

def analyze_and_save_report(report_df, output_csv_path=None, threshold=None):
    """
    Generate and print modular statistical analysis from the alarm report DataFrame,
    and save the detailed report to a CSV file if output_csv_path is specified.
    """
    if len(report_df) == 0:
        print("Empty report DataFrame.")
        return {}

    # Save to CSV
    if output_csv_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        report_df.to_csv(output_csv_path, index=False)
        print(f"Detailed alarm report saved to: {output_csv_path}")

    # General counts
    total_disks = len(report_df)
    failed_disks = report_df['has_failed'].sum()
    censored_disks = total_disks - failed_disks
    
    hits = report_df['is_hit'].sum()
    false_alarms = report_df['is_false_alarm'].sum()
    misses = report_df['is_miss'].sum()
    correct_rejections = report_df['is_correct_rejection'].sum()
    
    recall = hits / failed_disks if failed_disks > 0 else 0.0
    far = false_alarms / censored_disks if censored_disks > 0 else 0.0
    precision = hits / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Lead time statistics
    hit_rows = report_df[report_df['is_hit'] == 1]
    lead_times = hit_rows['days_to_failure_at_alarm'].dropna().values
    
    mean_lt = np.mean(lead_times) if len(lead_times) > 0 else 0.0
    median_lt = np.median(lead_times) if len(lead_times) > 0 else 0.0
    std_lt = np.std(lead_times) if len(lead_times) > 0 else 0.0
    min_lt = np.min(lead_times) if len(lead_times) > 0 else 0.0
    max_lt = np.max(lead_times) if len(lead_times) > 0 else 0.0

    # Lead time distribution buckets
    lt_distribution = {}
    if len(lead_times) > 0:
        lt_distribution = {
            '0-5 days': int(np.sum(lead_times <= 5)),
            '6-15 days': int(np.sum((lead_times > 5) & (lead_times <= 15))),
            '16-30 days': int(np.sum((lead_times > 15) & (lead_times <= 30)))
        }

    print("\n" + "="*50)
    print("      ROLLING INFERENCE ALARM DETAILED ANALYSIS      ")
    print("="*50)
    print(f"Total evaluated disks : {total_disks}")
    if threshold is not None:
        print(f"Alarm Threshold       : {threshold:.4f}")
    print(f"  - Failed (Target)   : {failed_disks}")
    print(f"  - Censored (Healthy): {censored_disks}")
    print("-"*50)
    print(f"Hits (True Alerts)    : {hits} (Alarms raised within lead time)")
    print(f"Misses                : {misses} (Failed with no alarm)")
    print(f"False Alarms          : {false_alarms} (Healthy with alarm, or too early)")
    print(f"Correct Rejections    : {correct_rejections} (Healthy with no alarm)")
    print("-"*50)
    print(f"Recall (Hit Rate)     : {recall:.4%}")
    print(f"FAR (False Alarm Rate): {far:.4%}")
    print(f"Precision             : {precision:.4%}")
    print(f"F1-Score              : {f1:.4%}")
    print("-"*50)
    print("Warning Lead Time Statistics (for Hits):")
    if len(lead_times) > 0:
        print(f"  - Mean Lead Time    : {mean_lt:.2f} days")
        print(f"  - Median Lead Time  : {median_lt:.2f} days")
        print(f"  - Std Deviation     : {std_lt:.2f} days")
        print(f"  - Range (Min / Max) : {min_lt} days / {max_lt} days")
        print("  - Distribution Bins :")
        for bucket, count in lt_distribution.items():
            pct = count / len(lead_times)
            print(f"    * {bucket:10s} : {count:3d} ({pct:.1%})")
    else:
        print("  - No hits to analyze lead times.")
    print("="*50 + "\n")

    analysis_metrics = {
        'recall': recall,
        'far': far,
        'precision': precision,
        'f1': f1,
        'mean_lead_time': mean_lt,
        'median_lead_time': median_lt,
        'std_lead_time': std_lt,
        'min_lead_time': min_lt,
        'max_lead_time': max_lt,
        'lead_time_distribution': lt_distribution
    }
    return analysis_metrics
