"""Single access point for everything the analysis scripts need from the
experiment run.

SOURCE OF TRUTH: results/experiments.db (SQLite). Operating thresholds and
checkpoints are read from there and from checkpoints/ via the same loaders the
experiments used -- never from hardcoded tables, never from the derived master
CSV exports, and never by guessing at filenames. If a value is missing, that is
an error: producing a figure from a substituted threshold is worse than
producing no figure.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

import pandas as pd

import config
import prediction_cache
from data_loader import load_dataset
from checkpoint_utils import load_checkpoint
from evaluator import RollingEvaluator, read_results_table, get_db_path, RUNS_TABLE

try:
    import torch
except ImportError:
    torch = None

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DB_PATH = get_db_path(RESULTS_DIR)
# Operating threshold chosen under the disk-level FAR constraint (METRIC_DESIGN.md).
THRESHOLD_COL = "threshold_disk_opt"
REPORTS_DIR = os.path.join(RESULTS_DIR, "lead_time_analysis", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

SEQUENCE_MODELS = ('lstm', 'gru')


def _load_runs(seed: int) -> pd.DataFrame:
    df = read_results_table(DB_PATH, RUNS_TABLE)
    if df.empty:
        raise RuntimeError(
            f"[analysis_data_loader] No experiment results found in '{DB_PATH}' (table "
            f"'{RUNS_TABLE}'). Run experiments/run_unified_threshold_experiments.py first."
        )
    df = df[df['seed'].astype(int) == int(seed)]
    if df.empty:
        raise RuntimeError(
            f"[analysis_data_loader] No results recorded for seed={seed} in '{RUNS_TABLE}'."
        )
    return df


def load_threshold_map(seed: int = 42) -> dict:
    """Return {(dataset, MODEL_UPPER): disk-optimal threshold} for one seed.

    This is the only place analysis code learns an operating threshold. There is
    deliberately no default/fallback table: a stale hardcoded threshold silently
    produces a plausible-looking but wrong figure.
    """
    df = _load_runs(seed)
    return {
        (str(row['dataset']).strip(), str(row['model']).upper()): float(row[THRESHOLD_COL])
        for _, row in df.iterrows()
    }


def get_proposed_threshold(dataset: str, model_name: str, seed: int = 42) -> float:
    """Fetch the exact disk-optimal threshold recorded for one (dataset, model, seed)."""
    df = _load_runs(seed)
    dataset_clean = str(dataset).strip()
    model_upper = str(model_name).upper()

    matched = df[
        (df['dataset'].astype(str).str.strip() == dataset_clean)
        & (df['model'].astype(str).str.upper() == model_upper)
    ]
    if matched.empty:
        available = sorted({
            (str(r['dataset']).strip(), str(r['model']).upper()) for _, r in df.iterrows()
        })
        raise KeyError(
            f"[analysis_data_loader] No threshold recorded for dataset='{dataset_clean}', "
            f"model='{model_upper}', seed={seed}. Recorded for this seed: {available}"
        )
    if len(matched) > 1:
        raise RuntimeError(
            f"[analysis_data_loader] {len(matched)} rows recorded for dataset='{dataset_clean}', "
            f"model='{model_upper}', seed={seed}; the result table should hold exactly one. "
            f"Deduplicate '{RUNS_TABLE}' before analysing."
        )
    return float(matched.iloc[0][THRESHOLD_COL])


def get_raw_predictions(dataset: str, model_name: str, seed: int = 42, split: str = "test",
                        need_all_alarms: bool = True):
    """Per-disk predictions for analysis, served from the cache when possible.

    `need_all_alarms=True` (alarm counts, ordinal histograms, temporal clustering)
    requires the full sequences, which are cached only for the seed those figures
    use. `need_all_alarms=False` (anything driven by the FIRST alarm) is served from
    the far smaller step cache.

    Falls back to running inference only when the needed cache layer is absent.
    """
    if need_all_alarms:
        try:
            return prediction_cache.load_full_predictions_as_raw_preds(dataset, model_name, seed, split)
        except FileNotFoundError:
            print(f"[analysis_data_loader] No full-prediction cache for {dataset}/{model_name}/"
                  f"seed{seed}/{split}; running inference.")
    else:
        try:
            return prediction_cache.load_disk_steps_as_raw_preds(dataset, model_name, seed, split)
        except FileNotFoundError:
            print(f"[analysis_data_loader] No step cache for {dataset}/{model_name}/seed{seed}/"
                  f"{split}; running inference.")

    data_path = os.path.join(PROJECT_ROOT, "data", "splitted", dataset)
    splits = load_dataset(data_path, model=model_name.lower())
    df = {"train": splits[0], "val": splits[1], "test": splits[2]}[split]
    features = splits[3]
    evaluator = build_evaluator(dataset, model_name, seed, features)
    return evaluator.get_raw_predictions(df)


def window_size_for(model_name: str) -> int:
    return config.WINDOW_SIZE if model_name.lower() in SEQUENCE_MODELS else 1


def load_analysis_model(dataset: str, model_name: str, seed: int, features: list):
    """Load the trained model for (dataset, model, seed) using the same checkpoint
    naming the experiments wrote. A missing checkpoint is an error -- analysis must
    not quietly fall back to a differently-configured run."""
    data_path = os.path.join(PROJECT_ROOT, "data", "splitted", dataset)
    is_sequence_model = model_name.lower() in SEQUENCE_MODELS

    model = load_checkpoint(
        model_name.lower(), "none", seed, config.TARGET_LEAD_TIME, data_path,
        input_dim=len(features), features=features,
        window_size=config.WINDOW_SIZE if is_sequence_model else None
    )
    if model is None:
        raise FileNotFoundError(
            f"[analysis_data_loader] No checkpoint for model='{model_name}', dataset='{dataset}', "
            f"seed={seed}. Train it via experiments/run_unified_threshold_experiments.py."
        )
    return model


def build_evaluator(dataset: str, model_name: str, seed: int, features: list) -> RollingEvaluator:
    model = load_analysis_model(dataset, model_name, seed, features)
    is_sequence_model = model_name.lower() in SEQUENCE_MODELS
    return RollingEvaluator(
        model=model,
        features=features,
        window_size=window_size_for(model_name),
        device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
        model_type='pytorch_class' if is_sequence_model else model_name.lower(),
        seed=seed
    )


def generate_alarm_report(dataset: str, model_name: str, seed: int = 42, threshold: float = None) -> pd.DataFrame:
    """
    Runs model inference and generates the disk-level alarm report DataFrame.
    """
    if threshold is None:
        threshold = get_proposed_threshold(dataset, model_name, seed)

    data_path = os.path.join(PROJECT_ROOT, "data", "splitted", dataset)
    _, _, test_df, features = load_dataset(data_path, model=model_name.lower())

    evaluator = build_evaluator(dataset, model_name, seed, features)
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
    Loads the disk-level alarm report, using the cached CSV only when it was
    produced at the threshold currently recorded in the DB. A cache that cannot be
    verified against the source of truth is recomputed, never returned as-is.
    """
    current_thresh = get_proposed_threshold(dataset, model_name, seed)
    model_name_upper = model_name.upper()
    csv_path = os.path.join(REPORTS_DIR, f"seed{seed}_alarm_report_{dataset}_{model_name_upper}.csv")

    if not force_recompute and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'threshold_used' not in df.columns or len(df) == 0:
            print(f"[analysis_data_loader] Cached report {os.path.basename(csv_path)} carries no "
                  f"threshold provenance. Recomputing...")
        else:
            cached_thresh = float(df['threshold_used'].iloc[0])
            if abs(cached_thresh - current_thresh) < 1e-4:
                return df
            print(f"[analysis_data_loader] Threshold changed for {dataset}/{model_name_upper}: "
                  f"{cached_thresh:.4f} -> {current_thresh:.4f}. Recomputing...")

    print(f"[analysis_data_loader] Generating alarm report: {dataset} | {model_name_upper} | thr={current_thresh:.4f}")
    df = generate_alarm_report(dataset, model_name, seed=seed, threshold=current_thresh)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    return df
