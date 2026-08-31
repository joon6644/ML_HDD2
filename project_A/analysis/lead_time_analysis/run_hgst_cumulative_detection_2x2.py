import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import torch
except ImportError:
    torch = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, ANALYSIS_DIR)

import config
from data_loader import load_dataset
from evaluator import RollingEvaluator
from analysis_data_loader import load_threshold_map, load_analysis_model

MODEL_TITLES = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "lstm": "LSTM",
    "gru": "GRU"
}


def collect_cumulative_detection_data(hdd_name: str, model_name: str, threshold: float, max_n: int = 50):
    hdd_path = os.path.join(PROJECT_ROOT, "data", "splitted", hdd_name)
    train_df, val_df, test_df, features = load_dataset(hdd_path, model=model_name)
    is_seq = model_name in ['lstm', 'gru']

    model = load_analysis_model(
        dataset=hdd_name,
        model_name=model_name,
        seed=config.SEED,
        features=features
    )

    evaluator = RollingEvaluator(
        model=model,
        features=features,
        window_size=config.WINDOW_SIZE if is_seq else 1,
        device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
        model_type='pytorch_class' if is_seq else model_name,
        seed=config.SEED
    )

    raw_preds = evaluator.get_raw_predictions(test_df, lead_time=config.TARGET_LEAD_TIME)

    failed_disks = [d for d in raw_preds if d['has_failed']]
    total_failed = len(failed_disks)

    first_valid_ordinals = []
    for disk in failed_disks:
        preds = disk['preds']
        dates = pd.to_datetime(disk['dates'])
        f_date = pd.to_datetime(disk['failure_date']) if disk['failure_date'] is not None else None

        alarm_mask = (preds >= threshold)
        if not np.any(alarm_mask):
            continue

        alarm_dates = dates[alarm_mask]
        for idx_k, ad in enumerate(alarm_dates, start=1):
            if f_date is not None and 0 <= (f_date - ad).days <= 30:
                first_valid_ordinals.append(idx_k)
                break

    k_indices = np.arange(1, max_n + 1)
    newly_detected_counts = np.zeros(max_n, dtype=int)

    if first_valid_ordinals:
        counts_series = pd.Series(first_valid_ordinals).value_counts()
        for k in k_indices:
            if k in counts_series.index:
                newly_detected_counts[k - 1] = counts_series[k]

    cum_detected_counts = np.cumsum(newly_detected_counts)
    cum_recall_pct = (cum_detected_counts / total_failed * 100) if total_failed > 0 else np.zeros(max_n)

    return {
        'total_failed': total_failed,
        'k_indices': k_indices,
        'newly_detected': newly_detected_counts,
        'cum_detected': cum_detected_counts,
        'cum_recall_pct': cum_recall_pct
    }


def plot_cumulative_detection_2x2(model_data_map: dict, hdd_name: str, output_paths: list, max_n: int = 50):
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Calibri', 'sans-serif']
    plt.rcParams['axes.edgecolor'] = '#2D3748'
    plt.rcParams['axes.linewidth'] = 1.2

    sns.set_theme(style="ticks", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2), dpi=300, sharey=True)

    fig.suptitle(
        f"Cumulative Failure Detection Rate by Sequential Alarm Order (Target 30-Day Window) — {hdd_name}",
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

    color_bar = "#64748B"    # Slate Gray for newly detected count per ordinal
    color_line = "#1D4ED8"   # Vibrant Royal Blue for cumulative detection curve

    # Max Y for Recall percentage
    max_y_lim = 45.0  # Max recall around 40%

    for idx, m_key in enumerate(models_order):
        r, c = subplot_positions[idx]
        ax = axes[r, c]
        data = model_data_map[m_key]

        k_indices = data['k_indices']
        new_cnt = data['newly_detected']
        cum_rec = data['cum_recall_pct']
        tot_failed = data['total_failed']

        bar_width = 0.85

        # Bar plot showing newly detected failed HDDs at each k-th alarm
        bars = ax.bar(
            k_indices, cum_rec, width=bar_width,
            color='#DBEAFE', edgecolor='#93C5FD', linewidth=0.6,
            align='center', label='Cumulative Recall (%)'
        )

        # Line plot for cumulative detection curve
        ax.plot(
            k_indices, cum_rec, color=color_line, linewidth=2.2,
            marker='o', markersize=3.5, label='Cumulative Curve'
        )

        # Annotate 1st alarm recall vs max cumulative recall
        rec_1st = cum_rec[0]
        rec_final = cum_rec[-1]
        cnt_1st = data['cum_detected'][0]
        cnt_final = data['cum_detected'][-1]

        ax.text(
            0.04, 0.81,
            f"1st Alarm Recall: {rec_1st:.1f}% ({cnt_1st}/{tot_failed})\n"
            f"Max Cum. Recall (≤50th): {rec_final:.1f}% ({cnt_final}/{tot_failed})",
            transform=ax.transAxes, fontsize=9.5, fontweight='bold', color='#1E3A8A',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8FAFC', edgecolor='#CBD5E1', alpha=0.95)
        )

        ax.set_title(
            subplot_labels[idx],
            fontsize=12.5, fontweight='bold', pad=9, loc='left', color="#1A202C"
        )

        ax.set_ylim(0, max_y_lim)
        ax.set_xlim(0, max_n + 1)
        ax.set_xticks(np.arange(0, max_n + 1, 5 if max_n <= 50 else 10))

        if c == 0:
            ax.set_ylabel("Cumulative Disk Recall (%)", fontsize=11, fontweight="bold", labelpad=6)
        else:
            ax.set_ylabel("")

        if r == 1:
            ax.set_xlabel("Alarm Sequence Threshold k (1st to k-th Alarm Allowed)", fontsize=11, fontweight="bold", labelpad=6)
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

    print(f"[SUCCESS] Saved 2x2 Cumulative Detection plot -> {output_paths[0]}")


def main():
    hdd_name = "HGST_20HUH721212ALN604"
    models = ["lgbm", "xgb", "lstm", "gru"]
    threshold_map = load_threshold_map(seed=config.SEED)

    results_dir = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(" GENERATING 2X2 CUMULATIVE DETECTION RATE PLOT (HGST) ")
    print("=" * 80)

    model_data_map = {}
    for m in models:
        lookup_key = "LGBM" if m.lower() == "lgbm" else ("XGB" if m.lower() == "xgb" else m.upper())
        thresh = threshold_map[(hdd_name, lookup_key)]
        print(f"\n[Processing] Model: {MODEL_TITLES[m]} | Threshold: {thresh:.4f}")

        data = collect_cumulative_detection_data(hdd_name, m, thresh, max_n=50)
        model_data_map[m] = data

    out1 = os.path.join(results_dir, "HGST_20HUH721212ALN604_cumulative_detection_2x2.png")

    plot_cumulative_detection_2x2(model_data_map, hdd_name, [out1], max_n=50)


if __name__ == "__main__":
    main()
