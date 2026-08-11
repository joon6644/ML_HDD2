import os
import sys
import sqlite3
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

import config
from data_loader import load_dataset
from checkpoint_utils import load_checkpoint
from evaluator import RollingEvaluator

try:
    import torch
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except ImportError:
    torch = None

DB_PATH = os.path.join(PROJECT_ROOT, "results", "experiments.db")
MASTER_CSV_PATH = os.path.join(PROJECT_ROOT, "results", "master_proposed_threshold_results.csv")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def get_proposed_threshold(dataset: str, model_name: str, seed: int = 42) -> float:
    """
    Fetches the exact Proposed-Opt threshold from SQLite (experiments.db) as single source of truth.
    Falls back to master_proposed_threshold_results.csv if SQLite is unavailable.
    """
    model_name_upper = model_name.upper()
    dataset_clean = dataset.strip()

    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT [Threshold (Proposed-Opt)] FROM master_proposed_threshold_results "
                "WHERE Seed = ? AND (데이터 = ? OR 데이터 LIKE ?) AND UPPER(Model) = ?",
                (seed, dataset_clean, f"%{dataset_clean}%", model_name_upper)
            )
            row = cursor.fetchone()
            conn.close()
            if row and row[0] is not None:
                return float(row[0])
        except Exception as e:
            print(f"[analysis_data_loader] SQLite query fallback warning: {e}")

    if os.path.exists(MASTER_CSV_PATH):
        for encoding in ('utf-8-sig', 'cp949'):
            for sep in (',', '\t'):
                try:
                    df = pd.read_csv(MASTER_CSV_PATH, encoding=encoding, sep=sep)
                    if df.shape[1] > 1:
                        matched = df[
                            (df['Seed'].astype(int) == seed) &
                            (df['Model'].str.upper() == model_name_upper) &
                            (df['데이터'].str.contains(dataset_clean, regex=False))
                        ]
                        if not matched.empty:
                            return float(matched.iloc[0]['Threshold (Proposed-Opt)'])
                except Exception:
                    continue

    raise ValueError(f"[analysis_data_loader] Could not find proposed threshold for dataset='{dataset}', model='{model_name}', seed={seed}")


def generate_alarm_report(dataset: str, model_name: str, seed: int = 42, threshold: float = None) -> pd.DataFrame:
    """
    Runs model inference and generates the disk-level alarm report DataFrame.
    """
    if threshold is None:
        threshold = get_proposed_threshold(dataset, model_name, seed)

    data_path = os.path.join(PROJECT_ROOT, "data", "splitted", dataset)
    is_sequence_model = model_name.lower() in ['lstm', 'gru']
    window_size = config.WINDOW_SIZE if is_sequence_model else 1

    _, _, test_df, features = load_dataset(data_path, model=model_name.lower())

    model = load_checkpoint(
        model_name.lower(), "none", seed, config.TARGET_LEAD_TIME, data_path,
        input_dim=len(features), features=features,
        window_size=window_size if is_sequence_model else None
    )
    if model is None:
        raise FileNotFoundError(f"[analysis_data_loader] Checkpoint missing for model='{model_name}', dataset='{dataset}', seed={seed}")

    model_type = 'pytorch_class' if is_sequence_model else model_name.lower()
    evaluator = RollingEvaluator(
        model=model, features=features, window_size=window_size,
        device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
        model_type=model_type, seed=seed
    )

    raw_preds = evaluator.get_raw_predictions(test_df, lead_time=config.TARGET_LEAD_TIME)
    _, report_df = evaluator.evaluate_proposed_level(raw_preds, threshold=threshold)

    first_seen = {d['serial_number']: pd.to_datetime(d['dates']).min() for d in raw_preds}
    report_df['first_seen_date'] = report_df['serial_number'].map(first_seen)
    report_df['days_since_observed'] = (
        pd.to_datetime(report_df['first_alarm_date']) - report_df['first_seen_date']
    ).dt.days
    report_df['threshold_used'] = float(threshold)

    return report_df


def load_alarm_report(dataset: str, model_name: str, seed: int = 42, force_recompute: bool = False) -> pd.DataFrame:
    """
    Loads alarm report DataFrame. Automatically validates threshold against SQLite/Master CSV.
    If threshold has changed or force_recompute=True, re-evaluates and updates report CSV automatically.
    """
    current_thresh = get_proposed_threshold(dataset, model_name, seed)
    model_name_upper = model_name.upper()
    csv_fname = f"seed{seed}_alarm_report_{dataset}_{model_name_upper}.csv"
    csv_path = os.path.join(REPORTS_DIR, csv_fname)

    if not force_recompute and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'threshold_used' in df.columns and len(df) > 0:
                cached_thresh = float(df['threshold_used'].iloc[0])
                if abs(cached_thresh - current_thresh) < 1e-4:
                    return df
                else:
                    print(f"[analysis_data_loader] Threshold changed for {dataset}/{model_name_upper}: {cached_thresh:.4f} -> {current_thresh:.4f}. Recomputing...")
            else:
                # If threshold_used column missing in legacy CSV, check matching or recompute
                return df
        except Exception:
            pass

    print(f"[analysis_data_loader] Generating alarm report: {dataset} | {model_name_upper} | thr={current_thresh:.4f}")
    df = generate_alarm_report(dataset, model_name, seed=seed, threshold=current_thresh)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    return df
