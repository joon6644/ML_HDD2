import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, f1_score
try:
    import torch
except ImportError:
    torch = None

from tqdm import tqdm
import config


def find_best_threshold_row_level(y_true, y_prob, max_far=0.01):
    """
    Finds the decision threshold that maximizes Row-Level Recall subject to
    Row-Level FAR <= max_far (default 1%) on the given validation (y_true, y_prob) set.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    thresholds = np.linspace(0.01, 0.99, 99)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    best_recall = -1.0
    best_far = 2.0
    best_threshold = 0.5

    fallback_threshold = 0.5
    fallback_min_far = 2.0
    fallback_max_recall = -1.0

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        recall = float(tp / n_pos) if n_pos > 0 else 0.0
        far = float(fp / n_neg) if n_neg > 0 else 0.0

        if (far < fallback_min_far) or (far == fallback_min_far and recall > fallback_max_recall):
            fallback_min_far = far
            fallback_max_recall = recall
            fallback_threshold = thresh

        if far <= max_far:
            if (recall > best_recall) or (recall == best_recall and far < best_far):
                best_recall = recall
                best_far = far
                best_threshold = thresh

    if best_recall < 0:
        best_threshold = fallback_threshold
        best_recall = fallback_max_recall

    return float(best_threshold), float(best_recall)


def calculate_row_level_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes sample-wise (row-level) metrics at specified threshold.
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
    Unified Single Sequential Inference & Proposed Disk-level Rolling Evaluator.
    """
    def __init__(self, model, features, window_size=None, device='cpu', model_type='sklearn', seed=None):
        self.model = model
        self.features = features
        self.window_size = window_size if window_size is not None else config.WINDOW_SIZE
        self.device = torch.device(device) if (torch is not None and isinstance(device, (str, torch.device))) else None
        self.model_type = model_type
        self.seed = seed if seed is not None else config.SEED
        
        if torch is not None and self.model_type.startswith('pytorch') and hasattr(self.model, 'eval'):
            self.model.to(self.device)
            self.model.eval()

    def predict_disk(self, disk_df, lead_time=None):
        if lead_time is None:
            lead_time = config.TARGET_LEAD_TIME

        has_segment = 'segment' in disk_df.columns
        sort_cols = ['segment', 'date'] if has_segment else ['date']
        df_sorted = disk_df.sort_values(sort_cols)

        n = len(df_sorted)
        if n == 0:
            return np.array([]), np.array([]), np.array([])

        y_true = ((df_sorted['RUL'] <= lead_time) & (df_sorted['censored'] == 0)).astype(np.float32).values
        valid_dates = df_sorted['date'].values
        x_raw = df_sorted[self.features].values

        # 1. Tabular Path (window_size == 1)
        if self.window_size == 1 and not (hasattr(self.model, 'is_pytorch') and self.model.is_pytorch):
            if hasattr(self.model, 'predict_proba'):
                preds = self.model.predict_proba(x_raw)[:, 1]
            elif hasattr(self.model, 'predict'):
                preds = self.model.predict(x_raw)
                if preds.ndim > 1 and preds.shape[1] == 2:
                    preds = preds[:, 1]
            else:
                with torch.no_grad():
                    x_t = torch.tensor(x_raw, dtype=torch.float32, device=self.device)
                    preds = torch.sigmoid(self.model(x_t)).view(-1).cpu().numpy()
            return valid_dates, y_true, preds

        # 2. Sequence Path (window_size > 1)
        def _infer_batch(x_batch):
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(x_batch)[:, 1]
            with torch.no_grad():
                return torch.sigmoid(self.model(x_batch)).view(-1).cpu().numpy()

        def _build_windows(x_np):
            x_t = torch.tensor(x_np, dtype=torch.float32, device=self.device)
            if self.window_size > 1:
                padding = x_t[0:1].repeat(self.window_size - 1, 1)
                x_t = torch.cat([padding, x_t], dim=0)
            return x_t.unfold(0, self.window_size, 1).transpose(1, 2)

        if has_segment and self.window_size > 1:
            seg_preds = []
            for seg_id in df_sorted['segment'].unique():
                seg_x = df_sorted.loc[df_sorted['segment'] == seg_id, self.features].values
                seg_preds.append(_infer_batch(_build_windows(seg_x)))
            preds = np.concatenate(seg_preds)
        else:
            preds = _infer_batch(_build_windows(x_raw))

        return valid_dates, y_true, preds

    def get_raw_predictions(self, dataset_df, sample_size=None, lead_time=None):
        """
        Executes single sequential inference over all disks in dataset_df.
        Returns a list of per-disk dictionary predictions.
        """
        if lead_time is None:
            lead_time = config.TARGET_LEAD_TIME

        serials = dataset_df['serial_number'].unique()
        if sample_size is not None and len(serials) > sample_size:
            rng = np.random.default_rng(self.seed)
            serials = rng.choice(serials, size=sample_size, replace=False)
            
        print(f"Running single sequential inference on {len(serials)} serials...")
        df_filtered = dataset_df[dataset_df['serial_number'].isin(serials)].copy()
        df_filtered['date'] = pd.to_datetime(df_filtered['date'])
        df_sorted = df_filtered.sort_values(['serial_number', 'date'])
        grouped = df_sorted.groupby('serial_number')
        
        raw_preds = []
        for serial, group in tqdm(grouped, desc="Sequential Disk Inference"):
            has_failed = (group['censored'].iloc[0] == 0)
            if has_failed:
                rul_zero = group[group['RUL'] == 0]
                failure_date = pd.to_datetime(
                    rul_zero['date'].iloc[0] if len(rul_zero) > 0 else group['date'].max()
                )
            else:
                failure_date = None
            dates, y_true, preds = self.predict_disk(group, lead_time=lead_time)
            if len(preds) == 0:
                continue
            raw_preds.append({
                'serial_number': serial,
                'has_failed': has_failed,
                'failure_date': failure_date,
                'dates': dates,
                'y_true': y_true,
                'preds': preds
            })
        return raw_preds

    def find_best_threshold_proposed_level(self, raw_preds, max_far=None):
        """
        Finds the threshold that maximizes Disk-level Recall on raw predictions
        subject to Disk-level FAR <= max_far (default 1%) on validation predictions.
        """
        if max_far is None:
            max_far = getattr(config, 'MAX_FAR', 0.01)

        thresholds = np.linspace(0.01, 0.99, 99)
        best_recall = -1.0
        best_far = 2.0
        best_threshold = 0.5
        
        failed_disks = sum(1 for disk in raw_preds if disk['has_failed'])
        censored_disks = sum(1 for disk in raw_preds if not disk['has_failed'])
        
        fallback_threshold = 0.5
        fallback_min_far = 2.0
        fallback_max_recall = -1.0

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
                    if days_to_failure >= 0:
                        hits += 1
                else:
                    false_alarms += 1
                    
            recall = hits / failed_disks if failed_disks > 0 else 0.0
            far = false_alarms / censored_disks if censored_disks > 0 else 0.0

            if (far < fallback_min_far) or (far == fallback_min_far and recall > fallback_max_recall):
                fallback_min_far = far
                fallback_max_recall = recall
                fallback_threshold = thresh

            if far <= max_far:
                if (recall > best_recall) or (recall == best_recall and far < best_far):
                    best_recall = recall
                    best_far = far
                    best_threshold = thresh

        if best_recall < 0:
            best_threshold = fallback_threshold
            best_recall = fallback_max_recall

        return float(best_threshold), float(best_recall)

    def evaluate_proposed_level(self, raw_preds, threshold):
        """
        Evaluates Proposed Disk-Level method using raw_preds at a fixed threshold.
        Lead time is sampled from the model's FIRST POSITIVE PREDICTION across ALL failed disks.
        """
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
                    if days_to_failure >= 0:
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
            
        report_df = pd.DataFrame(records)
        
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

        # Lead time calculation sampled over ALL failed disks with positive prediction
        hit_failed_rows = report_df[(report_df['has_failed'] == 1) & (report_df['alarm_triggered'] == 1)]
        lead_times = hit_failed_rows['days_to_failure_at_alarm'].dropna().values
        
        mean_lt = float(np.mean(lead_times)) if len(lead_times) > 0 else 0.0
        median_lt = float(np.median(lead_times)) if len(lead_times) > 0 else 0.0
        std_lt = float(np.std(lead_times)) if len(lead_times) > 0 else 0.0

        edr_15_count = sum(1 for lt in lead_times if lt >= 15)
        edr_15 = float(edr_15_count / failed_disks) if failed_disks > 0 else 0.0

        disk_metrics = {
            'tp': hits,
            'fp': false_alarms,
            'fn': misses,
            'tn': correct_rejections,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'far': far,
            'threshold': float(threshold),
            'median_lead_time': median_lt,
            'mean_lead_time': mean_lt,
            'std_lead_time': std_lt,
            'edr_15': edr_15
        }
        
        return disk_metrics, report_df


# ------------------------------------------------------------------------------
# Master CSV Result Persistence Functions
# ------------------------------------------------------------------------------
def _normalize_key_val(val):
    if pd.isna(val):
        return ""
    try:
        f = float(val)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except (ValueError, TypeError):
        return str(val).strip().lower()


def _clean_and_dedup_rows(df: pd.DataFrame, new_row: pd.DataFrame, key_cols: list) -> pd.DataFrame:
    df = df.dropna(how='all')
    if not df.empty and all(c in df.columns for c in key_cols):
        new_key = new_row.iloc[0]
        mask = pd.Series(True, index=df.index)
        for c in key_cols:
            target_norm = _normalize_key_val(new_key[c])
            col_norm = df[c].apply(_normalize_key_val)
            mask &= (col_norm == target_norm)
        df = df[~mask]
    return pd.concat([df, new_row], ignore_index=True)


def save_experiment_result_to_csv(master_csv_path: str, row_data: dict, key_cols: list):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.dirname(master_csv_path)
    os.makedirs(results_dir, exist_ok=True)

    df_row = pd.DataFrame([row_data])

    try:
        if os.path.exists(master_csv_path):
            df_master = pd.read_csv(master_csv_path, encoding='utf-8-sig')
            df_final = _clean_and_dedup_rows(df_master, df_row, key_cols)
        else:
            df_final = df_row
        df_final.to_csv(master_csv_path, index=False, encoding='utf-8-sig')
        print(f"[RESULT LOG] Updated master CSV: {master_csv_path}")
    except PermissionError:
        backup_path = master_csv_path + ".backup.csv"
        if os.path.exists(backup_path):
            df_backup = pd.read_csv(backup_path, encoding='utf-8-sig')
            df_final = _clean_and_dedup_rows(df_backup, df_row, key_cols)
        else:
            df_final = df_row
        df_final.to_csv(backup_path, index=False, encoding='utf-8-sig')
        print(f"[WARNING] Master CSV locked. Saved to backup: {backup_path}")
