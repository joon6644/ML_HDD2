import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

import config
from data_loader import load_dataset
from checkpoint_utils import CHECKPOINT_DIR
from evaluator import RollingEvaluator
from analysis.lead_time_analysis.run_operational_timeline_analysis import load_checkpoint_flexible

config.PIPELINE_VERSION = "v2"

MODEL_TITLES = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "lstm": "LSTM",
    "gru": "GRU"
}

# Exact Seed 42 Proposed Thresholds matching the summary benchmark table
SEED42_THRESHOLDS = {
    "lgbm": 0.99,
    "xgb": 0.47,
    "lstm": 0.09,
    "gru": 0.16
}


def collect_ordinal_alarm_data(hdd_name: str, model_name: str, threshold: float):
    hdd_path = os.path.join(PROJECT_ROOT, "data", "splitted", hdd_name)
    train_df, val_df, test_df, features = load_dataset(hdd_path, model=model_name)
    is_seq = model_name in ['lstm', 'gru']

    model = load_checkpoint_flexible(
        model_name=model_name,
        seed=config.SEED,
        lead_time=config.TARGET_LEAD_TIME,
        dataset_name=hdd_name,
        input_dim=len(features)
    )

    if model is None:
        raise FileNotFoundError(f"Checkpoint missing for model '{model_name}' on HDD '{hdd_name}'")

    evaluator = RollingEvaluator(
        model=model,
        features=features,
        window_size=config.WINDOW_SIZE if is_seq else 1,
        device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
        model_type='pytorch_class' if is_seq else model_name,
        seed=config.SEED
    )

    raw_preds = evaluator.get_raw_predictions(test_df, lead_time=config.TARGET_LEAD_TIME)

    alarm_records = []
    for disk in raw_preds:
        preds = disk['preds']
        dates = pd.to_datetime(disk['dates'])
        has_failed = disk['has_failed']
        f_date = pd.to_datetime(disk['failure_date']) if (has_failed and disk['failure_date'] is not None) else None

        alarm_mask = (preds >= threshold)
        if not np.any(alarm_mask):
            continue

        alarm_dates = dates[alarm_mask]
        for idx_k, ad in enumerate(alarm_dates, start=1):
            # Benchmark Disk-level precision logic: an alarm on a failed HDD before failure is a Valid Alarm (TP Hit).
            # An alarm on a non-failing HDD (or after failure) is a False Alarm (FP).
            if has_failed and f_date is not None:
                days_to_fail = (f_date - ad).days
                is_valid = (days_to_fail >= 0)
            else:
                is_valid = False

            alarm_records.append({
                'serial_number': disk['serial_number'],
                'alarm_index': idx_k,
                'is_valid': int(is_valid),
                'is_false': int(not is_valid)
            })

    return pd.DataFrame(alarm_records)


def plot_ordinal_alarm_histogram_2x2(model_data_map: dict, hdd_name: str, output_paths: list, max_n: int = 50):
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Calibri', 'sans-serif']
    plt.rcParams['axes.edgecolor'] = '#2D3748'
    plt.rcParams['axes.linewidth'] = 1.2

    sns.set_theme(style="ticks", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2), dpi=300, sharey=True)

    fig.suptitle(
        f"Distribution of Valid vs. False Alarms by Alarm Sequence Order — {hdd_name}",
        fontsize=15, fontweight="bold", y=0.98, color="#1A202C"
    )

    models_order = ["lgbm", "xgb", "lstm", "gru"]
    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    subplot_labels = [
        "(a) LightGBM",
        "(b) XGBoost",
        "(c) LSTM",
        "(d) GRU"
    ]

    color_valid = "#1E40AF"  # Valid Alarm (Hit)
    color_false = "#94A3B8"  # False Alarm (FP)

    # Calculate global max Y across subplots up to max_n
    max_y_global = 0
    for idx, m_key in enumerate(models_order):
        df_rec = model_data_map[m_key]
        if df_rec.empty:
            continue
        df_sub = df_rec[df_rec['alarm_index'] <= max_n]
        if df_sub.empty:
            continue
        counts = df_sub.groupby('alarm_index').size()
        if len(counts) > 0:
            max_y_global = max(max_y_global, counts.max())

    max_y_lim = max_y_global * 1.15

    for idx, m_key in enumerate(models_order):
        r, c = subplot_positions[idx]
        ax = axes[r, c]
        df_rec = model_data_map[m_key]

        if df_rec.empty:
            ax.text(0.5, 0.5, "No alarm data", ha='center', va='center')
            continue

        k_indices = np.arange(1, max_n + 1)
        valid_counts = np.zeros(max_n)
        false_counts = np.zeros(max_n)

        g = df_rec[df_rec['alarm_index'] <= max_n].groupby('alarm_index')[['is_valid', 'is_false']].sum()

        for k in k_indices:
            if k in g.index:
                valid_counts[k - 1] = g.loc[k, 'is_valid']
                false_counts[k - 1] = g.loc[k, 'is_false']

        bar_width = 0.85

        # Plot stacked histogram bars for 1st to N-th alarms
        ax.bar(
            k_indices, valid_counts, width=bar_width,
            color=color_valid, edgecolor='#FFFFFF', linewidth=0.5,
            align='center', label='Valid Alarm (TP)'
        )

        ax.bar(
            k_indices, false_counts, width=bar_width, bottom=valid_counts,
            color=color_false, edgecolor='#FFFFFF', linewidth=0.5,
            align='center', label='False Alarm (FP)'
        )

        # Print 1st alarm precision annotation on the subplot
        first_total = valid_counts[0] + false_counts[0]
        first_prec = (valid_counts[0] / first_total * 100) if first_total > 0 else 0
        ax.text(
            0.03, 0.84, f"1st Alarm Precision: {first_prec:.2f}% ({int(valid_counts[0])}/{int(first_total)})",
            transform=ax.transAxes, fontsize=10, fontweight='bold', color='#1E40AF',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#F1F5F9', edgecolor='#CBD5E1', alpha=0.95)
        )

        ax.set_title(
            subplot_labels[idx],
            fontsize=12.5, fontweight='bold', pad=9, loc='left', color="#1A202C"
        )

        ax.set_ylim(0, max_y_lim)
        ax.set_xlim(0, max_n + 1)
        ax.set_xticks(np.arange(0, max_n + 1, 5 if max_n <= 50 else 10))

        if c == 0:
            ax.set_ylabel("Alarm Occurrence Count", fontsize=11, fontweight="bold", labelpad=6)
        else:
            ax.set_ylabel("")

        if r == 1:
            ax.set_xlabel("Alarm Sequence Index (1st to n-th Alarm)", fontsize=11, fontweight="bold", labelpad=6)
        else:
            ax.set_xlabel("")

        ax.grid(True, axis="y", linestyle="--", alpha=0.3, color="#CBD5E1")
        ax.grid(False, axis="x")
        sns.despine(ax=ax, top=True, right=True)

        if r == 0 and c == 0:
            ax.legend(
                fontsize=9.5, loc='upper right', frameon=True,
                facecolor='#FFFFFF', edgecolor='#CBD5E1', framealpha=0.95
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.28, wspace=0.15)

    for path in output_paths:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Saved 2x2 Ordinal Alarm Histogram plot -> {output_paths[0]}")


def main():
    hdd_name = "HGST_20HUH721212ALN604"
    models = ["lgbm", "xgb", "lstm", "gru"]

    results_dir = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(" GENERATING 2X2 ORDINAL ALARM HISTOGRAM MATCHING BENCHMARK PRECISION ")
    print("=" * 80)

    model_data_map = {}
    for m in models:
        thresh = SEED42_THRESHOLDS[m]
        print(f"\n[Processing] Model: {MODEL_TITLES[m]} | Threshold: {thresh:.4f}")

        df_rec = collect_ordinal_alarm_data(hdd_name, m, thresh)
        model_data_map[m] = df_rec

    out1 = os.path.join(results_dir, "HGST_20HUH721212ALN604_ordinal_alarm_histogram_2x2.png")

    plot_ordinal_alarm_histogram_2x2(model_data_map, hdd_name, [out1], max_n=50)


if __name__ == "__main__":
    main()
