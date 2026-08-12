import os
import sqlite3
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
try:
    import torch
except ImportError:
    torch = None

from tqdm import tqdm
import config

# Decision-threshold search grid: 0.001 .. 0.999 in steps of 0.001 (paper 4.2).
THRESHOLD_GRID = np.linspace(0.001, 0.999, 999)


def find_best_threshold_row_level(y_true, y_prob, max_far=None):
    """
    Finds the decision threshold that maximizes Row-Level Recall subject to
    Row-Level FAR <= max_far on the given validation (y_true, y_prob) set.

    If NO threshold on the grid satisfies the FAR constraint, the constraint is
    infeasible for this model/dataset. That is a documented outcome, not an
    error: the threshold minimizing FAR (ties broken by higher Recall) is
    returned and a prominent notice is emitted so the run is never silently
    reported as constraint-satisfying.
    """
    if max_far is None:
        max_far = config.MAX_FAR

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"[Threshold Search] Row-level threshold search needs both classes present in the "
            f"validation set, got n_pos={n_pos}, n_neg={n_neg}. Refusing to emit a meaningless "
            f"threshold."
        )

    best_recall = -1.0
    best_far = 2.0
    best_threshold = None

    # Best achievable point if the constraint turns out to be infeasible.
    infeasible_threshold = None
    infeasible_min_far = 2.0
    infeasible_recall = -1.0

    for thresh in THRESHOLD_GRID:
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        recall = float(tp / n_pos)
        far = float(fp / n_neg)

        if (far < infeasible_min_far) or (far == infeasible_min_far and recall > infeasible_recall):
            infeasible_min_far = far
            infeasible_recall = recall
            infeasible_threshold = thresh

        if far <= max_far:
            if (recall > best_recall) or (recall == best_recall and far < best_far):
                best_recall = recall
                best_far = far
                best_threshold = thresh

    if best_threshold is None:
        print("!" * 80)
        print(f"[CONSTRAINT INFEASIBLE] No threshold on the grid satisfies Row-level FAR <= {max_far:.4f}.")
        print(f"  Minimum achievable validation FAR = {infeasible_min_far:.6f} at threshold {infeasible_threshold:.3f}.")
        print(f"  Falling back to that best-achievable threshold. The reported run does NOT satisfy")
        print(f"  the FAR constraint stated in the experiment design -- report it as such.")
        print("!" * 80)
        return float(infeasible_threshold), float(infeasible_recall)

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

    # No try/except: a failing roc_auc_score means the evaluation set is degenerate
    # (single class), which must surface rather than be papered over with 0.5.
    auroc = float(roc_auc_score(y_true, y_prob))
    # NOTE: PR-AUC is deliberately NOT computed (cost); the constant below is a
    # placeholder, not a measurement. Do not report it.
    pr_auc = 0.0

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    n_neg = int(np.sum(y_true == 0))
    if n_neg == 0:
        raise ValueError("[Row Metrics] Evaluation set contains no negative rows; FAR is undefined.")
    far = float(fp / n_neg)

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


def _assert_non_negative_lead_time(days_to_failure, serial_number):
    """A first alarm dated after the failure date breaks the Lead Time definition
    (LT_d = t_failure - t_alarm >= 0) and would be silently miscounted as 'Early'.
    It can only mean upstream corruption (post-failure rows, non-chronological
    segments), so fail loudly instead of absorbing it into a metric."""
    if days_to_failure < 0:
        raise ValueError(
            f"[Operational Evaluation] Disk '{serial_number}' produced a first alarm "
            f"{-days_to_failure} day(s) AFTER its failure date (Lead Time = {days_to_failure}). "
            f"Lead Time must be non-negative. Inspect the preprocessed data for rows observed "
            f"after the failure date or non-chronological segment ordering."
        )


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

    def find_best_threshold_proposed_level(self, raw_preds, max_eap=None, max_far=None, lead_time=None):
        """
        Finds the minimum threshold that satisfies Disk-level EAP (Early Alarm
        Proportion) <= max_eap on validation predictions.

        If NO threshold on the grid satisfies the EAP constraint, the constraint is
        infeasible for this model/dataset (observed for HGST/LGBM, whose skewed
        probability output never drives EAP below 1%). That is a documented outcome,
        not an error: the threshold minimizing EAP (ties broken by higher ODR) is
        returned and a prominent notice is emitted so the run is never silently
        reported as constraint-satisfying.
        """
        if max_eap is None:
            max_eap = config.MAX_EAP
        if lead_time is None:
            lead_time = config.TARGET_LEAD_TIME

        total_disks = len(raw_preds)
        if total_disks == 0:
            raise ValueError("[Threshold Search] Disk-level threshold search received zero disks.")

        infeasible_threshold = None
        infeasible_min_eap = 2.0
        infeasible_odr = -1.0

        for thresh in THRESHOLD_GRID:
            n_ontime, n_early, n_missed, n_cens_early, n_cens_no_alarm = 0, 0, 0, 0, 0
            for disk in raw_preds:
                has_failed = disk['has_failed']
                preds = disk['preds']
                alarm_mask = (preds >= thresh)
                alarm_triggered = np.any(alarm_mask)
                
                if has_failed:
                    if alarm_triggered:
                        first_alarm_idx = np.where(alarm_mask)[0][0]
                        first_alarm_date = pd.to_datetime(disk['dates'][first_alarm_idx])
                        days_to_failure = (disk['failure_date'] - first_alarm_date).days
                        _assert_non_negative_lead_time(days_to_failure, disk['serial_number'])
                        if days_to_failure <= lead_time:
                            n_ontime += 1
                        else:
                            n_early += 1
                    else:
                        n_missed += 1
                else:
                    if alarm_triggered:
                        n_cens_early += 1
                    else:
                        n_cens_no_alarm += 1

            n_failed = n_ontime + n_early + n_missed
            odr = float(n_ontime / n_failed) if n_failed > 0 else 0.0
            eap = float((n_early + n_cens_early) / total_disks)

            if (eap < infeasible_min_eap) or (eap == infeasible_min_eap and odr > infeasible_odr):
                infeasible_min_eap = eap
                infeasible_odr = odr
                infeasible_threshold = thresh

            # Return the MINIMUM threshold satisfying EAP <= max_eap constraint
            if eap <= max_eap:
                return float(thresh), float(odr)

        print("!" * 80)
        print(f"[CONSTRAINT INFEASIBLE] No threshold on the grid satisfies Disk-level EAP <= {max_eap:.4f}.")
        print(f"  Minimum achievable validation EAP = {infeasible_min_eap:.6f} at threshold {infeasible_threshold:.3f}.")
        print(f"  Falling back to that best-achievable threshold. The reported run does NOT satisfy")
        print(f"  the EAP constraint stated in the experiment design -- report it as such.")
        print("!" * 80)
        return float(infeasible_threshold), float(infeasible_odr)

    def evaluate_proposed_level(self, raw_preds, threshold, lead_time=None):
        """
        Evaluates Proposed Disk-Level method using raw_preds at a fixed threshold
        strictly following HDD operational metrics definition:
        - On-time: Failed HDD with first alarm within [0, lead_time] days before failure
        - Early: Failed HDD with first alarm > lead_time days before failure
        - Missed: Failed HDD with no alarm
        - Censored Early: Right-censored HDD with alarm
        - Censored No Alarm: Right-censored HDD with no alarm

        Operational Metrics:
        - OAP = N_ontime / (N_ontime + N_early)
        - ODR = N_ontime / (N_ontime + N_early + N_missed)
        - EAP = (N_early + N_cens_early) / Total_HDDs
        """
        if lead_time is None:
            lead_time = config.TARGET_LEAD_TIME
        if len(raw_preds) == 0:
            raise ValueError("[Operational Evaluation] Received zero disks to evaluate.")

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
            
            is_hit = 0         # N_ontime (TP)
            is_fp_early = 0    # N_early
            is_miss = 0        # N_missed (FN)
            is_fp_cens = 0     # N_cens_early
            is_correct_rejection = 0 # N_cens_no_alarm (TN)
            category = ""
            
            if alarm_triggered:
                first_alarm_idx = alarm_indices[0]
                first_alarm_date = pd.to_datetime(dates[first_alarm_idx])
                alarm_score = float(preds[first_alarm_idx])
                if has_failed:
                    days_to_failure = (failure_date - first_alarm_date).days
                    _assert_non_negative_lead_time(days_to_failure, serial)
                    if days_to_failure <= lead_time:
                        is_hit = 1
                        category = "On time"
                    else:
                        is_fp_early = 1
                        category = "Early"
                else:
                    is_fp_cens = 1
                    category = "Censored Early"
            else:
                if has_failed:
                    is_miss = 1
                    category = "Missed"
                else:
                    is_correct_rejection = 1
                    category = "Censored No Alarm"
                    
            records.append({
                'serial_number': serial,
                'has_failed': 1 if has_failed else 0,
                'alarm_triggered': 1 if alarm_triggered else 0,
                'first_alarm_date': first_alarm_date,
                'actual_failure_date': failure_date,
                'days_to_failure_at_alarm': days_to_failure,
                'category': category,
                'is_hit': is_hit,
                'is_fp_early': is_fp_early,
                'is_fp_cens': is_fp_cens,
                'is_false_alarm': is_fp_early + is_fp_cens,
                'is_miss': is_miss,
                'is_correct_rejection': is_correct_rejection,
                'max_predicted_score': max_score,
                'alarm_predicted_score': alarm_score
            })
            
        report_df = pd.DataFrame(records)
        
        total_disks = len(report_df)
        hits = int(report_df['is_hit'].sum())                  # N_ontime
        fp_early = int(report_df['is_fp_early'].sum())         # N_early
        misses = int(report_df['is_miss'].sum())                # N_missed
        fp_cens = int(report_df['is_fp_cens'].sum())           # N_cens_early
        correct_rejections = int(report_df['is_correct_rejection'].sum()) # N_cens_no_alarm
        
        failed_disks = hits + fp_early + misses
        censored_disks = fp_cens + correct_rejections
        false_alarms = fp_early + fp_cens
        
        oap = float(hits / (hits + fp_early)) if (hits + fp_early) > 0 else 0.0
        odr = float(hits / failed_disks) if failed_disks > 0 else 0.0
        eap = float((fp_early + fp_cens) / total_disks) if total_disks > 0 else 0.0

        precision = float(hits / (hits + false_alarms)) if (hits + false_alarms) > 0 else 0.0
        recall = odr
        far = float(fp_cens / censored_disks) if censored_disks > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Lead time calculation sampled over ALL failed disks where a first alarm exists (On time + Early)
        all_alarm_failed_rows = report_df[(report_df['has_failed'] == 1) & (report_df['alarm_triggered'] == 1)]
        lead_times = all_alarm_failed_rows['days_to_failure_at_alarm'].dropna().values
        
        mean_lt = float(np.mean(lead_times)) if len(lead_times) > 0 else 0.0
        median_lt = float(np.median(lead_times)) if len(lead_times) > 0 else 0.0
        std_lt = float(np.std(lead_times)) if len(lead_times) > 0 else 0.0

        edr_15_count = sum(1 for lt in lead_times if lt >= 15)
        edr_15 = float(edr_15_count / failed_disks) if failed_disks > 0 else 0.0

        disk_metrics = {
            'N_ontime': hits,
            'N_early': fp_early,
            'N_missed': misses,
            'N_cens_early': fp_cens,
            'N_cens_no_alarm': correct_rejections,
            'oap': oap,
            'odr': odr,
            'eap': eap,
            'tp': hits,
            'fp': false_alarms,
            'fp_early': fp_early,
            'fp_cens': fp_cens,
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
# Result Persistence
#
# SINGLE SOURCE OF TRUTH: results/experiments.db (SQLite).
# The master CSV files are a derived export, regenerated from the DB after every
# write. Never read results back from the CSV -- read the DB.
# ------------------------------------------------------------------------------
CSV_ENCODING = 'utf-8-sig'

# master CSV basename -> authoritative SQLite table
_TABLE_BY_CSV_KEYWORD = (
    ('row', 'master_row_threshold_results'),
    ('proposed', 'master_proposed_threshold_results'),
)


def resolve_table_name(master_csv_path: str) -> str:
    """Map a master CSV path to its authoritative SQLite table."""
    filename = os.path.basename(master_csv_path)
    for keyword, table in _TABLE_BY_CSV_KEYWORD:
        if keyword in filename:
            return table
    raise ValueError(
        f"[Result Persistence] Cannot determine the SQLite table for '{filename}'. "
        f"Expected the filename to contain one of: {[k for k, _ in _TABLE_BY_CSV_KEYWORD]}."
    )


def get_db_path(results_dir: str) -> str:
    return os.path.join(results_dir, "experiments.db")


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


def read_results_table(db_path: str, table_name: str) -> pd.DataFrame:
    """Read an experiment result table from the authoritative SQLite DB.
    Returns an empty DataFrame if the DB or table does not exist yet (a genuine
    'no results recorded' state); any other failure propagates."""
    if not os.path.exists(db_path):
        return pd.DataFrame()
    with sqlite3.connect(db_path, timeout=30.0) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if cursor.fetchone() is None:
            return pd.DataFrame()
        return pd.read_sql(f"SELECT * FROM [{table_name}]", conn)


def save_experiment_result_to_sqlite(db_path: str, table_name: str, row_data: dict, key_cols: list) -> pd.DataFrame:
    """
    Saves or updates an experiment result in the authoritative SQLite database,
    replacing any existing row with the same key. Returns the full updated table.

    No try/except: a failed write means results were lost, which must stop the run
    rather than print a warning that scrolls past in a 156-run batch.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    df_row = pd.DataFrame([row_data])

    df_existing = read_results_table(db_path, table_name)
    df_final = df_row if df_existing.empty else _clean_and_dedup_rows(df_existing, df_row, key_cols)

    with sqlite3.connect(db_path, timeout=30.0) as conn:
        df_final.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"[RESULT LOG] Updated SQLite DB table '{table_name}': {db_path}")
    return df_final


def export_results_table_to_csv(df: pd.DataFrame, master_csv_path: str):
    """Export an authoritative results table to its derived master CSV.

    A locked CSV (Excel holding the file open) is a hard error: silently diverting
    to a '.backup.csv' used to leave two files that disagree about the results.
    """
    try:
        df.to_csv(master_csv_path, index=False, encoding=CSV_ENCODING)
    except PermissionError as e:
        raise PermissionError(
            f"[Result Persistence] Cannot write '{master_csv_path}' -- the file is locked "
            f"(most likely open in Excel). The result IS safely stored in the SQLite DB; close the "
            f"file and re-export. Refusing to write a divergent backup copy."
        ) from e
    print(f"[RESULT LOG] Exported master CSV: {master_csv_path}")


def save_experiment_result_to_csv(master_csv_path: str, row_data: dict, key_cols: list):
    """Record one experiment result: write to the SQLite source of truth, then
    regenerate the derived master CSV from it."""
    results_dir = os.path.dirname(master_csv_path)
    os.makedirs(results_dir, exist_ok=True)

    table_name = resolve_table_name(master_csv_path)
    df_final = save_experiment_result_to_sqlite(get_db_path(results_dir), table_name, row_data, key_cols)
    export_results_table_to_csv(df_final, master_csv_path)
