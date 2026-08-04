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
    Find the threshold that maximizes Row-Level (sample-wise) Recall subject to
    Row-Level FAR <= max_far (default 1%) on the given (y_true, y_prob) set.
    If no threshold satisfies FAR <= max_far, returns the threshold that minimizes FAR
    (ties broken by maximizing Recall).
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

        # Sort order: respect segments before date so each segment is contiguous
        sort_cols = ['segment', 'date'] if has_segment else ['date']
        df_sorted = disk_df.sort_values(sort_cols)

        n = len(df_sorted)
        if n == 0:
            return np.array([]), np.array([]), np.array([])

        y_true      = ((df_sorted['RUL'] <= lead_time) & (df_sorted['censored'] == 0)).astype(np.float32).values
        valid_dates = df_sorted['date'].values
        x_raw       = df_sorted[self.features].values

        # ---- Tabular (window_size == 1) path: no segment issue ----
        if self.window_size == 1 and not (hasattr(self.model, 'is_pytorch') and self.model.is_pytorch):
            if hasattr(self.model, 'predict_proba'):
                preds = self.model.predict_proba(x_raw)[:, 1]
            elif hasattr(self.model, 'predict'):
                preds = self.model.predict(x_raw)
                if preds.ndim > 1 and preds.shape[1] == 2:
                    preds = preds[:, 1]
            else:
                with torch.no_grad():
                    x_t   = torch.tensor(x_raw, dtype=torch.float32, device=self.device)
                    preds = torch.sigmoid(self.model(x_t)).view(-1).cpu().numpy()
            return valid_dates, y_true, preds

        # ---- Sequence (window_size > 1) path ----
        def _infer_batch(x_batch):
            """Run model inference on a pre-built (N, W, F) tensor."""
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(x_batch)[:, 1]
            with torch.no_grad():
                return torch.sigmoid(self.model(x_batch)).view(-1).cpu().numpy()

        def _build_windows(x_np):
            """Pad first row (W-1) times and unfold into (N, W, F) tensor."""
            x_t    = torch.tensor(x_np, dtype=torch.float32, device=self.device)
            if self.window_size > 1:
                padding = x_t[0:1].repeat(self.window_size - 1, 1)
                x_t     = torch.cat([padding, x_t], dim=0)
            return x_t.unfold(0, self.window_size, 1).transpose(1, 2)

        if has_segment and self.window_size > 1:
            # Process each segment independently so window history resets at boundaries
            seg_preds = []
            for seg_id in df_sorted['segment'].unique():
                seg_x = df_sorted.loc[df_sorted['segment'] == seg_id, self.features].values
                seg_preds.append(_infer_batch(_build_windows(seg_x)))
            preds = np.concatenate(seg_preds)
        else:
            preds = _infer_batch(_build_windows(x_raw))

        return valid_dates, y_true, preds

    def get_raw_predictions(self, val_df, sample_size=None, lead_time=None):
        if lead_time is None:
            lead_time = config.TARGET_LEAD_TIME

        serials = val_df['serial_number'].unique()
        if sample_size is not None and len(serials) > sample_size:
            rng = np.random.default_rng(self.seed)
            serials = rng.choice(serials, size=sample_size, replace=False)
            
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

    def find_best_threshold(self, raw_preds, max_far=None, lead_time=None):
        """
        Find the threshold that maximizes Recall on raw predictions subject to FAR <= max_far (default 1%).
        If no threshold satisfies FAR <= max_far, returns the threshold that minimizes FAR.
        """
        if max_far is None:
            max_far = getattr(config, 'MAX_FAR', 0.01)
        if lead_time is None:
            lead_time = config.TARGET_LEAD_TIME

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
                    if 0 <= days_to_failure <= lead_time:
                        hits += 1
                else:
                    false_alarms += 1
                    
            recall = hits / failed_disks if failed_disks > 0 else 0.0
            far = false_alarms / censored_disks if censored_disks > 0 else 0.0

            # Track fallback in case no threshold meets max_far (minimize FAR, then maximize Recall)
            if (far < fallback_min_far) or (far == fallback_min_far and recall > fallback_max_recall):
                fallback_min_far = far
                fallback_max_recall = recall
                fallback_threshold = thresh

            # Filter by constraint FAR <= max_far (1%)
            if far <= max_far:
                if (recall > best_recall) or (recall == best_recall and far < best_far):
                    best_recall = recall
                    best_far = far
                    best_threshold = thresh

        if best_recall < 0:
            best_threshold = fallback_threshold
            best_recall = fallback_max_recall

        return float(best_threshold), float(best_recall)

    def evaluate_alarms(self, val_df, threshold='auto', max_far=None, lead_time=None, sample_size=None):
        if max_far is None:
            max_far = getattr(config, 'MAX_FAR', 0.01)
        if lead_time is None:
            lead_time = config.TARGET_LEAD_TIME

        raw_preds = self.get_raw_predictions(val_df, sample_size=sample_size, lead_time=lead_time)
        if threshold == 'auto':
            target_threshold, max_val_recall = self.find_best_threshold(raw_preds, max_far=max_far, lead_time=lead_time)
            print(f"Optimal threshold found: {target_threshold:.4f} (Max Recall @ FAR <= {max_far:.2%}: {max_val_recall:.4%})")
        else:
            target_threshold = float(threshold)
            
        report_df = self.build_report_from_raw_predictions(raw_preds, target_threshold, lead_time=lead_time)
        metrics = analyze_report(report_df, threshold=target_threshold)
        metrics['threshold'] = target_threshold

        return metrics, report_df


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

    # Calculate EDR@15 (Early Detection Ratio within 15 days before failure)
    edr_15_count = sum(1 for lt in lead_times if lt >= 15)
    edr_15 = float(edr_15_count / failed_disks) if failed_disks > 0 else 0.0

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
    print(f"EDR@15 (Early Detect) : {edr_15:.4%}")
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
        'mean_lead_time': mean_lt,
        'edr_15': edr_15
    }


def save_lead_time_distribution(report_df, save_dir: str, filename_prefix: str = "evaluation"):
    """
    Saves lead time distribution per-disk data as CSV and generates a high-quality visualization plot PNG.
    """
    os.makedirs(save_dir, exist_ok=True)
    hit_disks = report_df[report_df['is_hit'] == 1].copy()
    lead_times = hit_disks['days_to_failure_at_alarm'].dropna().values

    # 1. Save Lead Time Distribution Data CSV
    lt_csv_path = os.path.join(save_dir, f"{filename_prefix}_lead_time_distribution.csv")
    cols_to_save = ['serial_number', 'actual_failure_date', 'first_alarm_date', 'days_to_failure_at_alarm', 'alarm_predicted_score']
    existing_cols = [c for c in cols_to_save if c in hit_disks.columns]
    hit_disks[existing_cols].to_csv(lt_csv_path, index=False, encoding='utf-8-sig')

    # 2. Save Visualization Plot PNG
    plot_path = os.path.join(save_dir, f"{filename_prefix}_lead_time_plot.png")
    plt.figure(figsize=(9, 5))
    if len(lead_times) > 0:
        sns.histplot(lead_times, bins=min(30, max(5, len(np.unique(lead_times)))), kde=True, color='#2b5c8f', edgecolor='black')
        median_lt = float(np.median(lead_times))
        mean_lt = float(np.mean(lead_times))
        plt.axvline(median_lt, color='red', linestyle='--', linewidth=2, label=f'Median LT: {median_lt:.1f}d')
        plt.axvline(mean_lt, color='orange', linestyle=':', linewidth=2, label=f'Mean LT: {mean_lt:.1f}d')
        plt.legend(fontsize=11)
    else:
        plt.text(0.5, 0.5, 'No Hit Alarms Triggered', horizontalalignment='center', verticalalignment='center', fontsize=14)

    plt.title(f"Lead Time Distribution ({filename_prefix})", fontsize=13, fontweight='bold')
    plt.xlabel("Lead Time (Days to Failure at Alarm)", fontsize=11)
    plt.ylabel("Disk Count", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved Lead Time CSV : {lt_csv_path}")
    print(f"Saved Lead Time Plot: {plot_path}")
    return lt_csv_path, plot_path


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
            'Additional_Details': f"Threshold: {disk_metrics.get('threshold', 0.5):.4f} | Median LT: {disk_metrics.get('median_lead_time', 0.0):.2f}d | EDR@15: {disk_metrics.get('edr_15', 0.0):.4f}"
        }
    ]

    df_concat = pd.DataFrame(concatenated_data)
    df_concat.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print("\n" + "="*70)
    print("[SUMMARY] CONCATENATED CONFUSION MATRIX & EVALUATION SUMMARY REPORT (2 EVALUATION METHODS)")
    print("="*70)
    print(df_concat.to_string(index=False))
    print("="*70)
    print(f"Saved concatenated confusion matrix to: {csv_path}\n")

    return df_concat


_MASTER_CSV_KEY_COLS = ['Model', '데이터', '불균형 처리', '첫날포함여부', 'Seed', 'Lead Time']


def _normalize_key_val(val):
    """Normalizes key values (float vs int vs string) for exact matching."""
    if pd.isna(val):
        return ""
    try:
        f = float(val)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except (ValueError, TypeError):
        return str(val).strip().lower()


def _clean_and_dedup_rows(df: pd.DataFrame, new_row: pd.DataFrame) -> pd.DataFrame:
    """Drop stray blank rows and replace any existing row matching the exact same experiment key
    (Model, Dataset, Imbalance, FirstDay, Seed, LeadTime). Strictly PRESERVES all runs with different
    seeds or settings so previous experiment results are NEVER lost."""
    df = df.dropna(how='all')
    if not df.empty and all(c in df.columns for c in _MASTER_CSV_KEY_COLS):
        new_key = new_row.iloc[0]
        mask = pd.Series(True, index=df.index)
        for c in _MASTER_CSV_KEY_COLS:
            target_norm = _normalize_key_val(new_key[c])
            col_norm = df[c].apply(_normalize_key_val)
            mask &= (col_norm == target_norm)
        df = df[~mask]
    return pd.concat([df, new_row], ignore_index=True)


def append_master_experiment_result(
    model_name: str,
    dataset_name: str,
    imbalance_strategy: str,
    drop_failure_day: bool,
    row_metrics: dict,
    disk_metrics: dict,
    seed: int = None,
    lead_time: int = None,
    master_csv_path: str = None
):
    """
    Permanently appends experiment run metrics to a master CSV file.
    Logs both evaluation methods (Row-level, Disk-level rolling).
    Re-running the exact same (model, dataset, imbalance, drop_failure_day, seed, lead_time)
    combination replaces the previous row in place instead of appending a duplicate.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    if master_csv_path is None:
        master_csv_path = os.path.join(results_dir, "master_experiment_results.csv")

    backup_csv_path = os.path.join(results_dir, "backup_master_results.csv")

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    first_day_status = "제거(True)" if drop_failure_day else "포함(False)"

    row_data = {
        'Timestamp': timestamp_str,
        'Model': model_name.upper(),
        '데이터': os.path.basename(dataset_name),
        '불균형 처리': imbalance_strategy,
        '첫날포함여부': first_day_status,
        'Seed': seed,
        'Lead Time': lead_time,

        # 1. Row-Level Evaluation (행 단위)
        'Row Precision': round(float(row_metrics.get('precision', 0.0)), 4),
        'Row Recall': round(float(row_metrics.get('recall', 0.0)), 4),
        'Row F1': round(float(row_metrics.get('f1', 0.0)), 4),
        'Row PR-AUC': round(float(row_metrics.get('pr_auc', 0.0)), 4),
        'Row FAR (%)': round(float(row_metrics.get('far', 0.0)) * 100, 2),

        # 2. Disk-Level Rolling Evaluation (개체 롤링 단위)
        'Threshold': round(float(disk_metrics.get('threshold', 0.5)), 4),
        'Disk-level Precision': round(float(disk_metrics.get('precision', 0.0)), 4),
        'Disk-level Recall': round(float(disk_metrics.get('recall', 0.0)), 4),
        'F1-score': round(float(disk_metrics.get('f1', 0.0)), 4),
        'Disk-level FAR (%)': round(float(disk_metrics.get('far', 0.0)) * 100, 2),
        'Median Lead Time': round(float(disk_metrics.get('median_lead_time', 0.0)), 2),
        'EDR@15': round(float(disk_metrics.get('edr_15', 0.0)), 4),
    }

    df_row = pd.DataFrame([row_data])

    # 1. Try to sync any previous pending backup rows first (deduping against master)
    if os.path.exists(backup_csv_path) and os.path.exists(master_csv_path):
        try:
            df_backup = pd.read_csv(backup_csv_path, encoding='utf-8-sig').dropna(how='all')
            df_master = pd.read_csv(master_csv_path, encoding='utf-8-sig')
            for _, backup_row in df_backup.iterrows():
                df_master = _clean_and_dedup_rows(df_master, pd.DataFrame([backup_row]))
            df_master.to_csv(master_csv_path, index=False, encoding='utf-8-sig')
            os.remove(backup_csv_path)
            print(f"[Auto-Sync] Restored and synced pending backup results to master CSV -> {master_csv_path}")
        except Exception:
            pass  # Master CSV might still be locked

    # 2. Try writing current row to master_experiment_results.csv (replacing any prior row with the same key)
    try:
        if os.path.exists(master_csv_path):
            df_master = pd.read_csv(master_csv_path, encoding='utf-8-sig')
            df_final = _clean_and_dedup_rows(df_master, df_row)
        else:
            df_final = df_row
        df_final.to_csv(master_csv_path, index=False, encoding='utf-8-sig')
        print("\n" + "="*80)
        print(f"[MASTER LOG] PERMANENT MASTER EXPERIMENT LOG APPENDED -> {master_csv_path}")
        print("="*80)
        print(df_row.to_string(index=False))
        print("="*80 + "\n")
    except PermissionError:
        # Fallback: Save to backup_master_results.csv with UTF-8-SIG encoding so data/encoding is NEVER corrupted!
        if os.path.exists(backup_csv_path):
            df_backup = pd.read_csv(backup_csv_path, encoding='utf-8-sig')
            df_final = _clean_and_dedup_rows(df_backup, df_row)
        else:
            df_final = df_row
        df_final.to_csv(backup_csv_path, index=False, encoding='utf-8-sig')
        print("\n" + "="*80)
        print(f"[SAFE-GUARD BACKUP] master_experiment_results.csv is currently LOCKED by Excel.")
        print(f"  -> Saved current experiment result with UTF-8 BOM encoding to BACKUP file: {backup_csv_path}")
        print(f"  -> It will automatically merge into master_experiment_results.csv on the next run once Excel is closed.")
        print("="*80)
        print(df_row.to_string(index=False))
        print("="*80 + "\n")

    return df_row
