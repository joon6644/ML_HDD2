"""
Plots the Proposed Disk-Level Recall vs False Alarm Rate (FAR) trade-off curve for
each (dataset, model) combination, i.e. how recall degrades as the FAR budget is
tightened by raising the threshold.

Reuses the per-threshold sweep CSVs produced by run_threshold_leadtime_sweep.py
(threshold, recall, far, hits, ...) when available; otherwise runs the sweep itself
by importing the shared inference/sweep functions from that module.

The operating point actually used in master_proposed_threshold_results.csv (the
threshold chosen under the FAR <= config.MAX_FAR constraint) is marked on each curve
for reference.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, ANALYSIS_DIR)

import config
from run_threshold_leadtime_sweep import (
    DATASETS, MODELS, MANUFACTURER_MAP, get_raw_preds, sweep_thresholds
)


def _read_master_csv(path: str) -> pd.DataFrame:
    """Robustly reads the master results CSV: normally utf-8-sig/comma, but can end
    up as cp949/tab-delimited after being opened and re-saved in Excel."""
    for encoding in ('utf-8-sig', 'cp949'):
        for sep in (',', '\t'):
            try:
                df = pd.read_csv(path, encoding=encoding, sep=sep)
                if df.shape[1] > 1:
                    return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
    raise ValueError(f"Could not parse master CSV with known encodings/separators: {path}")


def load_operating_points(seed: int) -> dict:
    """Reads (dataset, model) -> chosen threshold from master_proposed_threshold_results.csv."""
    master_csv = os.path.join(PROJECT_ROOT, "results", "master_proposed_threshold_results.csv")
    op_map = {}
    if not os.path.exists(master_csv):
        return op_map
    df = _read_master_csv(master_csv)
    df = df[df['Seed'].astype(int) == seed]
    for _, row in df.iterrows():
        op_map[(str(row['데이터']).strip(), str(row['Model']).upper())] = float(row['Threshold (Proposed-Opt)'])
    return op_map


def get_sweep_df(dataset: str, model_name: str, seed: int, step: float, results_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(results_dir, f"seed{seed}_threshold_sweep_{dataset}_{model_name.upper()}.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, encoding='utf-8-sig')

    print(f"[Cache Miss] No existing sweep CSV for {dataset}/{model_name.upper()} -> running inference...")
    thresholds = np.round(np.arange(0.01, 1.00, step), 4)
    evaluator, raw_preds = get_raw_preds(dataset, model_name, seed)
    sweep_df = sweep_thresholds(evaluator, raw_preds, thresholds)
    sweep_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    return sweep_df


def plot_facet_grid(all_sweeps: dict, op_points: dict, results_dir: str, seed: int, max_far: float):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(len(DATASETS), len(MODELS), figsize=(20, 12), sharex=False, sharey=True)

    # Shared x-axis limit across every panel so the FAR<=constraint line lands at the
    # same horizontal position everywhere (panels are visually comparable at a glance).
    # Fixed at 5%; curves simply run off the right edge of panels whose FAR range
    # exceeds it (e.g. HGST reaches ~15% for some models).
    shared_x_max = 5.0

    for i, dataset in enumerate(DATASETS):
        for j, model_name in enumerate(MODELS):
            key = (dataset, model_name.upper())
            df = all_sweeps[key].sort_values('far')
            mfr = MANUFACTURER_MAP.get(dataset, dataset)
            ax = axes[i, j]

            far_pct = df['far'].values * 100
            recall_pct = df['recall'].values * 100

            ax.plot(far_pct, recall_pct, color="#0a1f38", linewidth=2.2, alpha=0.9, label="Recall vs FAR")

            ax.axvline(max_far * 100, color="#c0392b", linestyle="--", linewidth=1.3,
                        label=f"FAR <= {max_far*100:.0f}% constraint")

            op_thresh = op_points.get(key)
            if op_thresh is not None:
                row = df.iloc[(df['threshold'] - op_thresh).abs().argsort()[:1]]
                if len(row) > 0:
                    ax.scatter(row['far'].values * 100, row['recall'].values * 100,
                               color="#c0392b", s=55, zorder=5, edgecolor='black', linewidth=0.6,
                               label=f"Selected thr={op_thresh:.2f}")

            # FAR is a ratio and can never be negative; anchor the axis at 0 instead of
            # letting matplotlib's default margin push it into negative territory.
            ax.set_xlim(0, shared_x_max)
            ax.set_ylim(0, 100)
            ax.margins(x=0)

            ax.set_title(f"{mfr}\n{model_name.upper()}", fontsize=9)
            if i == len(DATASETS) - 1:
                ax.set_xlabel("FAR (%)", fontsize=8)
            if j == 0:
                ax.set_ylabel("Recall (%)", fontsize=8)
            ax.legend(fontsize=6, loc="lower right")

    fig.suptitle(f"Proposed Disk-Level Recall vs False Alarm Rate (Seed={seed})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(results_dir, f"seed{seed}_recall_vs_far_grid.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Plot Saved] -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Recall vs FAR trade-off curve per (dataset, model)")
    parser.add_argument('--seed', type=int, default=config.SEED, help='Seed to evaluate (must have a saved checkpoint)')
    parser.add_argument('--datasets', type=str, nargs='+', default=DATASETS)
    parser.add_argument('--models', type=str, nargs='+', default=MODELS)
    parser.add_argument('--step', type=float, default=0.01, help='Threshold step size (only used if no cached sweep CSV exists)')
    args = parser.parse_args()

    results_dir = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
    os.makedirs(results_dir, exist_ok=True)

    max_far = getattr(config, 'MAX_FAR', 0.01)

    print("=" * 80)
    print(f" RECALL vs FAR SWEEP (SEED={args.seed})")
    print(f" Datasets : {args.datasets}")
    print(f" Models   : {args.models}")
    print(f" MAX_FAR constraint (from config): {max_far*100:.1f}%")
    print("=" * 80)

    op_points = load_operating_points(args.seed)

    all_sweeps = {}
    for dataset in args.datasets:
        for model_name in args.models:
            print(f"\n[Processing] {dataset} | {model_name.upper()}")
            sweep_df = get_sweep_df(dataset, model_name, args.seed, args.step, results_dir)
            all_sweeps[(dataset, model_name.upper())] = sweep_df

    plot_facet_grid(all_sweeps, op_points, results_dir, args.seed, max_far)
    print("\nDone.")


if __name__ == "__main__":
    main()
