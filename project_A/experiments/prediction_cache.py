"""Reusable caches of a run's predictions, so that changing a decision threshold,
a constraint, or the prediction horizon H never requires re-running inference.

Three layers, each the smallest sufficient statistic for what it serves:

1. ROW CURVE  -- (threshold, tp, fp, fn, tn) at every grid threshold, per split.
   Every row-level metric (precision / recall / FAR / F1) at any grid threshold is
   a function of these four counts, so the 27M individual row predictions do not
   need to be kept. A few KB per run.

2. DISK STEPS -- per HDD, the points where the running maximum of its prediction
   sequence increases, with the date of each increase.
   The first alarm at threshold t is the first step whose value is >= t, so this
   reconstructs first-alarm behaviour exactly, for ANY threshold and ANY horizon H.
   Roughly 27:1 to 364:1 smaller than the raw sequences.

   A disk rebuilt from its steps is behaviourally identical to the original for
   first-alarm queries, so `RollingEvaluator.evaluate_proposed_level` and
   `find_best_threshold_disk_level` run on it unchanged.

3. FULL PREDS -- the complete per-row prediction sequence.
   Only the first alarm is needed for the reported metrics, but several analysis
   figures aggregate over EVERY alarm (alarm counts per HDD, alarm ordinal
   histograms, temporal clustering), which the step curve cannot reconstruct.
   Written only for the seed those figures use, to keep the cache small.

Predictions are stored in whatever dtype the model produced; no quantization, so
`preds >= threshold` reproduces the original decision exactly.
"""
import os

import numpy as np
import pandas as pd

CACHE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "prediction_cache"
)

# Analysis figures that aggregate over all alarms are written for this seed only.
FULL_PREDICTION_SEED = 42
FULL_PREDICTION_SPLITS = ("test",)


def _run_dir(dataset: str, model_name: str, seed: int) -> str:
    return os.path.join(CACHE_ROOT, os.path.basename(dataset), f"{model_name.lower()}_seed{seed}")


def row_curve_path(dataset, model_name, seed, split):
    return os.path.join(_run_dir(dataset, model_name, seed), f"row_curve_{split}.parquet")


def disk_steps_path(dataset, model_name, seed, split):
    return os.path.join(_run_dir(dataset, model_name, seed), f"disk_steps_{split}.parquet")


def full_preds_path(dataset, model_name, seed, split):
    return os.path.join(_run_dir(dataset, model_name, seed), f"full_preds_{split}.parquet")


def wants_full_predictions(seed: int, split: str) -> bool:
    return int(seed) == FULL_PREDICTION_SEED and split in FULL_PREDICTION_SPLITS


# ------------------------------------------------------------------------------
# 1. Row-level confusion curve
# ------------------------------------------------------------------------------
def build_row_curve(y_true, y_prob, thresholds) -> pd.DataFrame:
    """Confusion counts at every threshold, computed in one sorted pass."""
    y_true = np.asarray(y_true).astype(bool)
    y_prob = np.asarray(y_prob)
    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)

    order = np.argsort(y_prob, kind="mergesort")
    sorted_prob = y_prob[order]
    # cumulative positives/negatives among predictions strictly below each threshold
    pos_below = np.cumsum(y_true[order])
    neg_below = np.cumsum(~y_true[order])

    # index of the first prediction >= threshold
    cut = np.searchsorted(sorted_prob, np.asarray(thresholds), side="left")
    fn = np.where(cut > 0, pos_below[np.clip(cut - 1, 0, None)], 0)
    tn = np.where(cut > 0, neg_below[np.clip(cut - 1, 0, None)], 0)

    return pd.DataFrame({
        "threshold": np.asarray(thresholds, dtype=np.float64),
        "tp": (n_pos - fn).astype(np.int64),
        "fp": (n_neg - tn).astype(np.int64),
        "fn": fn.astype(np.int64),
        "tn": tn.astype(np.int64),
    })


def save_row_curve(df: pd.DataFrame, dataset, model_name, seed, split):
    path = row_curve_path(dataset, model_name, seed, split)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_row_curve(dataset, model_name, seed, split) -> pd.DataFrame:
    path = row_curve_path(dataset, model_name, seed, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[Prediction Cache] No row curve at '{path}'. Run the experiment for "
            f"{dataset}/{model_name}/seed{seed} to populate the cache."
        )
    return pd.read_parquet(path)


# ------------------------------------------------------------------------------
# 2. Disk-level step curves
# ------------------------------------------------------------------------------
def build_disk_steps(raw_preds) -> pd.DataFrame:
    """Running-maximum increase points per disk: the sufficient statistic for
    first-alarm behaviour at any threshold."""
    frames = []
    for disk in raw_preds:
        preds = np.asarray(disk["preds"])
        dates = pd.to_datetime(disk["dates"])
        running_max = np.maximum.accumulate(preds)
        keep = np.flatnonzero(np.diff(running_max, prepend=-np.inf) > 0)
        frames.append(pd.DataFrame({
            "serial_number": disk["serial_number"],
            "has_failed": bool(disk["has_failed"]),
            "failure_date": pd.to_datetime(disk["failure_date"]) if disk["failure_date"] is not None else pd.NaT,
            "step_value": preds[keep],
            "step_date": dates[keep],
        }))
    if not frames:
        raise ValueError("[Prediction Cache] build_disk_steps received zero disks.")
    return pd.concat(frames, ignore_index=True)


def save_disk_steps(df: pd.DataFrame, dataset, model_name, seed, split):
    path = disk_steps_path(dataset, model_name, seed, split)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_disk_steps_as_raw_preds(dataset, model_name, seed, split):
    """Rebuild a raw_preds list from cached steps.

    Each rebuilt disk keeps only its step points, which yields the same first alarm
    as the original sequence at every threshold, so the evaluator runs on it
    unchanged. It does NOT reproduce every alarm -- use the full-prediction cache
    for analyses that count all alarms.
    """
    path = disk_steps_path(dataset, model_name, seed, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[Prediction Cache] No disk steps at '{path}'. Run the experiment for "
            f"{dataset}/{model_name}/seed{seed} to populate the cache."
        )
    df = pd.read_parquet(path)
    raw_preds = []
    for serial, g in df.groupby("serial_number", sort=False):
        g = g.sort_values("step_date")
        failure_date = g["failure_date"].iloc[0]
        raw_preds.append({
            "serial_number": serial,
            "has_failed": bool(g["has_failed"].iloc[0]),
            "failure_date": None if pd.isna(failure_date) else pd.Timestamp(failure_date),
            "dates": g["step_date"].values,
            "preds": g["step_value"].values,
            "y_true": np.zeros(len(g), dtype=np.float32),  # not meaningful on steps
        })
    return raw_preds


# ------------------------------------------------------------------------------
# 3. Full prediction sequences
# ------------------------------------------------------------------------------
def build_full_predictions(raw_preds) -> pd.DataFrame:
    frames = []
    for disk in raw_preds:
        frames.append(pd.DataFrame({
            "serial_number": disk["serial_number"],
            "has_failed": bool(disk["has_failed"]),
            "failure_date": pd.to_datetime(disk["failure_date"]) if disk["failure_date"] is not None else pd.NaT,
            "date": pd.to_datetime(disk["dates"]),
            "pred": np.asarray(disk["preds"]),
            "y_true": np.asarray(disk["y_true"]),
        }))
    return pd.concat(frames, ignore_index=True)


def save_full_predictions(df: pd.DataFrame, dataset, model_name, seed, split):
    path = full_preds_path(dataset, model_name, seed, split)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_full_predictions_as_raw_preds(dataset, model_name, seed, split):
    """Rebuild the exact raw_preds list, including every alarm."""
    path = full_preds_path(dataset, model_name, seed, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[Prediction Cache] No full predictions at '{path}'. They are stored only for "
            f"seed {FULL_PREDICTION_SEED} / splits {FULL_PREDICTION_SPLITS}; for other runs use "
            f"the disk-step cache (first-alarm analyses only) or re-run inference."
        )
    df = pd.read_parquet(path)
    raw_preds = []
    for serial, g in df.groupby("serial_number", sort=False):
        g = g.sort_values("date")
        failure_date = g["failure_date"].iloc[0]
        raw_preds.append({
            "serial_number": serial,
            "has_failed": bool(g["has_failed"].iloc[0]),
            "failure_date": None if pd.isna(failure_date) else pd.Timestamp(failure_date),
            "dates": g["date"].values,
            "preds": g["pred"].values,
            "y_true": g["y_true"].values,
        })
    return raw_preds


def save_all_layers(raw_preds, y_true, y_prob, thresholds, dataset, model_name, seed, split):
    """Write every cache layer that applies to this run/split. Returns written paths."""
    written = {
        "row_curve": save_row_curve(
            build_row_curve(y_true, y_prob, thresholds), dataset, model_name, seed, split),
        "disk_steps": save_disk_steps(
            build_disk_steps(raw_preds), dataset, model_name, seed, split),
    }
    if wants_full_predictions(seed, split):
        written["full_preds"] = save_full_predictions(
            build_full_predictions(raw_preds), dataset, model_name, seed, split)
    return written
