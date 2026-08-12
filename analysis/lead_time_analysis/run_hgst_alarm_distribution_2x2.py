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

def collect_alarm_data(hdd_name: str, model_name: str, threshold: float):
    model_upper = model_name.upper()
    lookup_key = "LGBM" if model_name.lower() == "lgbm" else ("XGB" if model_name.lower() == "xgb" else model_name.upper())

    # Fast Path: Check if cached report CSV exists
    reports_dir = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis", "reports")
    report_csv = os.path.join(reports_dir, f"seed42_alarm_report_{hdd_name}_{lookup_key}.csv")

    if os.path.exists(report_csv):
        print(f"[CACHE HIT] Instant load from report CSV -> {report_csv}")
        df = pd.read_csv(report_csv)
        df_rec = pd.DataFrame({
            'serial_number': df['serial_number'],
            'has_failed': df['has_failed'],
            'total_alarm_count': df['alarm_triggered'].fillna(0).astype(int),
            'valid_alarm_count': df['is_hit'].fillna(0).astype(int),
            'false_alarm_count': df['is_false_alarm'].fillna(0).astype(int)
        })
        return df_rec[df_rec['total_alarm_count'] > 0]

    print(f"\n[Processing] Running inference for Model: {model_upper} | Threshold: {threshold:.4f}")
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

    hdd_records = []
    for disk in raw_preds:
        preds = disk['preds']
        dates = pd.to_datetime(disk['dates'])
        has_failed = disk['has_failed']
        f_date = pd.to_datetime(disk['failure_date']) if (has_failed and disk['failure_date'] is not None) else None

        alarm_mask = (preds >= threshold)
        total_alarm_count = int(np.sum(alarm_mask))

        if total_alarm_count == 0:
            continue

        alarm_dates = dates[alarm_mask]
        valid_count = 0
        false_count = 0

        for ad in alarm_dates:
            if has_failed and f_date is not None and 0 <= (f_date - ad).days <= 30:
                valid_count += 1
            else:
                false_count += 1

        hdd_records.append({
            'serial_number': disk['serial_number'],
            'has_failed': has_failed,
            'total_alarm_count': total_alarm_count,
            'valid_alarm_count': valid_count,
            'false_alarm_count': false_count
        })

    return pd.DataFrame(hdd_records)


def plot_stacked_alarm_histogram_2x2(model_data_map: dict, hdd_name: str, output_paths: list):
    # Set sophisticated academic font & style
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Calibri', 'sans-serif']
    plt.rcParams['axes.edgecolor'] = '#2D3748'
    plt.rcParams['axes.linewidth'] = 1.2

    sns.set_theme(style="ticks", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2), dpi=300, sharey=True)

    fig.suptitle(
        f"Histogram of Alarm Triggers per HDD Unit — {hdd_name}",
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

    # Sophisticated Elegant Academic Palette (No Pepsi Cola!)
    # Valid Alarm: Deep Elegant Indigo / Slate Blue (#1E40AF)
    # False Alarm: Muted Slate Gray (#94A3B8)
    color_valid = "#1E40AF"  # Deep Indigo Blue
    color_false = "#94A3B8"  # Elegant Slate Gray

    # Fine-grained histogram binning: bin width = 2 across range 0 to 100 (50 bins)
    max_x = 100
    bin_width = 2
    bin_edges = np.arange(0, max_x + bin_width, bin_width)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Shared max Y calculation across subplots
    max_y_global = 0

    for idx, m_key in enumerate(models_order):
        df_rec = model_data_map[m_key]
        if df_rec.empty:
            continue

        counts_clipped = np.clip(df_rec['total_alarm_count'].values, 0, max_x - 0.1)
        valid_weighted = [r['valid_alarm_count'] / r['total_alarm_count'] for _, r in df_rec.iterrows()]
        false_weighted = [r['false_alarm_count'] / r['total_alarm_count'] for _, r in df_rec.iterrows()]

        hist_valid, _ = np.histogram(counts_clipped, bins=bin_edges, weights=valid_weighted)
        hist_false, _ = np.histogram(counts_clipped, bins=bin_edges, weights=false_weighted)

        total_bin_y = hist_valid + hist_false
        if len(total_bin_y) > 0:
            max_y_global = max(max_y_global, np.max(total_bin_y))

    max_y_lim = max_y_global * 1.15

    for idx, m_key in enumerate(models_order):
        r, c = subplot_positions[idx]
        ax = axes[r, c]
        df_rec = model_data_map[m_key]

        if df_rec.empty:
            ax.text(0.5, 0.5, "No alarm data", ha='center', va='center')
            continue

        counts_clipped = np.clip(df_rec['total_alarm_count'].values, 0, max_x - 0.1)
        valid_weighted = [r['valid_alarm_count'] / r['total_alarm_count'] for _, r in df_rec.iterrows()]
        false_weighted = [r['false_alarm_count'] / r['total_alarm_count'] for _, r in df_rec.iterrows()]

        hist_valid, _ = np.histogram(counts_clipped, bins=bin_edges, weights=valid_weighted)
        hist_false, _ = np.histogram(counts_clipped, bins=bin_edges, weights=false_weighted)

        # Plot dense histogram bars with crisp white edges
        ax.bar(
            bin_centers, hist_valid, width=bin_width,
            color=color_valid, edgecolor='#FFFFFF', linewidth=0.5,
            align='center', label='Valid Alarm (Within 30 Days)'
        )

        ax.bar(
            bin_centers, hist_false, width=bin_width, bottom=hist_valid,
            color=color_false, edgecolor='#FFFFFF', linewidth=0.5,
            align='center', label='False Alarm'
        )

        ax.set_title(
            subplot_labels[idx],
            fontsize=12.5, fontweight='bold', pad=9, loc='left', color="#1A202C"
        )

        ax.set_ylim(0, max_y_lim)
        ax.set_xlim(-1, max_x + 1)
        ax.set_xticks(np.arange(0, max_x + 1, 10))

        if c == 0:
            ax.set_ylabel("HDD Unit Count", fontsize=11, fontweight="bold", labelpad=6)
        else:
            ax.set_ylabel("")

        if r == 1:
            ax.set_xlabel("Alarm Triggers per HDD Unit", fontsize=11, fontweight="bold", labelpad=6)
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

    print(f"[SUCCESS] Saved 2x2 Sophisticated Histogram plot -> {output_paths[0]}")


def main():
    hdd_name = "HGST_20HUH721212ALN604"
    threshold_map = load_threshold_map(seed=config.SEED)
    models = ["lgbm", "xgb", "lstm", "gru"]

    results_dir = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(" GENERATING 2X2 SOPHISTICATED ALARM HISTOGRAM PER HDD (HGST) ")
    print("=" * 80)

    model_data_map = {}
    for m in models:
        lookup_key = "LGBM" if m == "lgbm" else m.upper()
        thresh = threshold_map[(hdd_name, lookup_key)]
        print(f"\n[Processing] Model: {MODEL_TITLES[m]} | Threshold: {thresh:.4f}")

        df_rec = collect_alarm_data(hdd_name, m, thresh)
        model_data_map[m] = df_rec

    out1 = os.path.join(results_dir, "HGST_20HUH721212ALN604_alarm_count_stacked_2x2.png")

    plot_stacked_alarm_histogram_2x2(model_data_map, hdd_name, [out1])


if __name__ == "__main__":
    main()
