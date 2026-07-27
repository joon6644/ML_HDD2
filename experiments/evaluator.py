import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, f1_score
try:
    import torch
except ImportError:
    torch = None

from tqdm import tqdm
import config


def calculate_row_level_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes row-level (sample-wise) classification metrics and confusion matrix:
    TP, FP, FN, TN, Precision, Recall, F1, PR-AUC, AUROC, False Alarm Rate (FAR)
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.5
    try:
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(recall_arr, precision_arr))
    except Exception:
        pr_auc = 0.0
        
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    n_neg = np.sum(y_true == 0)
    far = float(fp / n_neg) if n_neg > 0 else 0.0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'pr_auc': pr_auc,
        'auroc': auroc,
        'far': far
    }


class RollingEvaluator:
    """
    Rolling Inference Evaluator for disk failure alert policies over daily time-series sequences.
    """
    def __init__(self, model, features, window_size=None, device='cpu', model_type='sklearn'):
        self.model = model
        self.features = features
        self.window_size = window_size if window_size is not None else config.WINDOW_SIZE
        self.device = torch.device(device) if (torch is not None and isinstance(device, (str, torch.device))) else None
        self.model_type = model_type
        
        if torch is not None and self.model_type.startswith('pytorch') and hasattr(self.model, 'eval'):
            self.model.to(self.device)
            self.model.eval()

    def predict_disk(self, disk_df, lead_time=None):
        if lead_time is None:
            lead_time = config.TARGET_LEAD_TIME

        df_sorted = disk_df.sort_values('date')
        n = len(df_sorted)
        if n == 0:
            return np.array([]), np.array([]), np.array([])
            
        y_true = ((df_sorted['RUL'] <= lead_time) & (df_sorted['censored'] == 0)).astype(np.float32).values
        valid_dates = df_sorted['date'].values
        x_raw = df_sorted[self.features].values
        
        if self.model_type in ['lightgbm', 'xgb', 'rf', 'sklearn']:
            if hasattr(self.model, 'predict_proba'):
                preds = self.model.predict_proba(x_raw)[:, 1]
            elif hasattr(self.model, 'predict'):
                preds = self.model.predict(x_raw)
                if preds.ndim > 1 and preds.shape[1] == 2:
                    preds = preds[:, 1]
            else:
                preds = self.model(x_raw)
            return valid_dates, y_true, preds

        x_tensor = torch.tensor(x_raw, dtype=torch.float32, device=self.device)
        if self.window_size > 1:
            first_row = x_tensor[0:1]
            padding = first_row.repeat(self.window_size - 1, 1)
            x_padded = torch.cat([padding, x_tensor], dim=0)
            x_batch = x_padded.unfold(0, self.window_size, 1).transpose(1, 2)
        else:
            x_batch = x_tensor
            
        with torch.no_grad():
            logits = self.model(x_batch)
            preds = torch.sigmoid(logits).view(-1).cpu().numpy()
            
        return valid_dates, y_true, preds

    def get_raw_predictions(self, val_df, sample_size=None, lead_time=None):
        if lead_time is None:
            lead_time = config.TARGET_LEAD_TIME

        serials = val_df['serial_number'].unique()
        if sample_size is not None and len(serials) > sample_size:
            np.random.seed(config.SEED)
            serials = np.random.choice(serials, size=sample_size, replace=False)
            
        print(f"Running rolling inference on {len(serials)} serials...")
        val_df_filtered = val_df[val_df['serial_number'].isin(serials)].copy()
        val_df_filtered['date'] = pd.to_datetime(val_df_filtered['date'])
        val_df_sorted = val_df_filtered.sort_values(['serial_number', 'date'])
        grouped = val_df_sorted.groupby('serial_number')
        
        raw_preds = []
        for serial, group in tqdm(grouped, desc="Predicting disks"):
            has_failed = (group['censored'].iloc[0] == 0)
            failure_date = pd.to_datetime(group['date'].max()) if has_failed else None
            dates, y_true, preds = self.predict_disk(group, lead_time=lead_time)
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

    def build_report_from_raw_predictions(self, raw_preds, threshold, lead_time=None):
        if lead_time is None:
            lead_time = config.TARGET_LEAD_TIME

        records = []
        for disk in raw_preds:
            serial = disk['serial_number']
            has_failed = disk['has_failed']
            failure_date = disk['failure_date']
            dates = disk['dates']
            preds = disk['preds']
            
            max_score = float(preds.max()) if len(preds) > 0 else 0.0
            alarm_mask = (preds >= threshold)
            alarm_indices = np.where(alarm_mask)[0]
            
            alarm_triggered = len(alarm_indices) > 0
            first_alarm_date = None
            alarm_score = None
            days_to_failure = None
            
            is_hit, is_miss, is_false_alarm, is_correct_rejection = 0, 0, 0, 0
            
            if alarm_triggered:
                first_alarm_idx = alarm_indices[0]
                first_alarm_date = pd.to_datetime(dates[first_alarm_idx])
                alarm_score = float(preds[first_alarm_idx])
                if has_failed:
                    days_to_failure = (failure_date - first_alarm_date).days
                    if 0 <= days_to_failure <= lead_time:
                        is_hit = 1
                    else:
                        is_miss = 1
                else:
                    is_false_alarm = 1
            else:
                if has_failed:
                    is_miss = 1
                else:
                    is_correct_rejection = 1
                    
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

    def find_best_threshold(self, raw_preds, lead_time=None):
        if lead_time is None:
            lead_time = config.TARGET_LEAD_TIME

        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1, best_threshold = -1.0, 0.5
        failed_disks = sum(1 for disk in raw_preds if disk['has_failed'])
        
        for thresh in thresholds:
            hits, false_alarms = 0, 0
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

    def evaluate_alarms(self, val_df, threshold='auto', lead_time=None, sample_size=None):
        if lead_time is None:
            lead_time = config.TARGET_LEAD_TIME

        raw_preds = self.get_raw_predictions(val_df, sample_size=sample_size, lead_time=lead_time)
        if threshold == 'auto':
            target_threshold, best_f1 = self.find_best_threshold(raw_preds, lead_time=lead_time)
            print(f"Optimal threshold found: {target_threshold:.4f} (Max F1: {best_f1:.4%})")
        else:
            target_threshold = float(threshold)
            
        report_df = self.build_report_from_raw_predictions(raw_preds, target_threshold, lead_time=lead_time)
        metrics = analyze_report(report_df, threshold=target_threshold)
        metrics['threshold'] = target_threshold
        return metrics


def analyze_report(report_df, threshold=None):
    if len(report_df) == 0:
        return {}
    total_disks = len(report_df)
    failed_disks = int(report_df['has_failed'].sum())
    censored_disks = int(total_disks - failed_disks)
    
    hits = int(report_df['is_hit'].sum())
    false_alarms = int(report_df['is_false_alarm'].sum())
    misses = int(report_df['is_miss'].sum())
    correct_rejections = int(report_df['is_correct_rejection'].sum())
    
    recall = float(hits / failed_disks) if failed_disks > 0 else 0.0
    far = float(false_alarms / censored_disks) if censored_disks > 0 else 0.0
    precision = float(hits / (hits + false_alarms)) if (hits + false_alarms) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    hit_rows = report_df[report_df['is_hit'] == 1]
    lead_times = hit_rows['days_to_failure_at_alarm'].dropna().values
    
    mean_lt = float(np.mean(lead_times)) if len(lead_times) > 0 else 0.0
    median_lt = float(np.median(lead_times)) if len(lead_times) > 0 else 0.0
    std_lt = float(np.std(lead_times)) if len(lead_times) > 0 else 0.0

    print("\n" + "="*50)
    print("      ROLLING INFERENCE ALARM DETAILED ANALYSIS      ")
    print("="*50)
    print(f"Total Disks           : {total_disks} (Failed: {failed_disks}, Healthy: {censored_disks})")
    if threshold is not None:
        print(f"Alarm Threshold       : {threshold:.4f}")
    print(f"Hits (TP) / Misses(FN): {hits} / {misses}")
    print(f"False Alarms(FP) / CR : {false_alarms} / {correct_rejections}")
    print(f"Precision / Recall / F1: {precision:.4%} / {recall:.4%} / {f1:.4%}")
    print(f"FAR (False Alarm Rate): {far:.4%}")
    print(f"Median Lead Time      : {median_lt:.2f} days (Mean: {mean_lt:.2f}d, Std: {std_lt:.2f}d)")
    print("="*50 + "\n")

    return {
        'tp': hits,
        'fp': false_alarms,
        'fn': misses,
        'tn': correct_rejections,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'far': far,
        'median_lead_time': median_lt,
        'mean_lead_time': mean_lt
    }


def save_concatenated_confusion_matrices(row_metrics: dict, disk_metrics: dict, save_dir: str, filename_prefix: str = "evaluation"):
    """
    Concatenates Row-level (Sample-wise) and Disk-level (Entity-wise) confusion matrices
    into a single structured report and saves it as CSV.
    """
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{filename_prefix}_concatenated_confusion_matrix.csv")

    concatenated_data = [
        {
            'Evaluation_Level': 'Row-Level (Sample-wise)',
            'TN': row_metrics.get('tn', 0),
            'FP': row_metrics.get('fp', 0),
            'FN': row_metrics.get('fn', 0),
            'TP': row_metrics.get('tp', 0),
            'Precision': f"{row_metrics.get('precision', 0.0):.4f}",
            'Recall': f"{row_metrics.get('recall', 0.0):.4f}",
            'F1_Score': f"{row_metrics.get('f1', 0.0):.4f}",
            'FAR': f"{row_metrics.get('far', 0.0):.4f}",
            'Additional_Details': f"PR-AUC: {row_metrics.get('pr_auc', 0.0):.4f} | AUROC: {row_metrics.get('auroc', 0.0):.4f}"
        },
        {
            'Evaluation_Level': 'Disk-Level (Entity-wise)',
            'TN': disk_metrics.get('tn', 0),
            'FP': disk_metrics.get('fp', 0),
            'FN': disk_metrics.get('fn', 0),
            'TP': disk_metrics.get('tp', 0),
            'Precision': f"{disk_metrics.get('precision', 0.0):.4f}",
            'Recall': f"{disk_metrics.get('recall', 0.0):.4f}",
            'F1_Score': f"{disk_metrics.get('f1', 0.0):.4f}",
            'FAR': f"{disk_metrics.get('far', 0.0):.4f}",
            'Additional_Details': f"Threshold: {disk_metrics.get('threshold', 0.5):.4f} | Median LT: {disk_metrics.get('median_lead_time', 0.0):.2f}d | Mean LT: {disk_metrics.get('mean_lead_time', 0.0):.2f}d"
        }
    ]

    df_concat = pd.DataFrame(concatenated_data)
    df_concat.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print("\n" + "="*70)
    print("📊 CONCATENATED CONFUSION MATRIX & EVALUATION SUMMARY REPORT")
    print("="*70)
    print(df_concat.to_string(index=False))
    print("="*70)
    print(f"Saved concatenated confusion matrix to: {csv_path}\n")

    return df_concat
