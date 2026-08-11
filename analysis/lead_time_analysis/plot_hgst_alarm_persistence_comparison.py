import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis", "lead_time_analysis")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

if ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, ANALYSIS_DIR)
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

from analysis_data_loader import get_proposed_threshold
from data_loader import load_dataset
from checkpoint_utils import load_checkpoint
from evaluator import RollingEvaluator
import config

HDD_NAME = "HGST_20HUH721212ALN604"
SEED = 42

MODELS = [
    ("LGBM", "LightGBM"),
    ("XGB", "XGBoost"),
    ("LSTM", "LSTM"),
    ("GRU", "GRU")
]

MODEL_COLORS = {
    "LightGBM": "#2b5c8f",
    "XGBoost":  "#d95f02",
    "LSTM":     "#7570b3",
    "GRU":      "#1b9e77"
}


def load_alarm_burden_metrics(model_code: str):
    m_lower = model_code.lower()
    thr = get_proposed_threshold(HDD_NAME, m_lower, seed=SEED)
    data_path = os.path.join(PROJECT_ROOT, "data", "splitted", HDD_NAME)
    is_seq = m_lower in ['lstm', 'gru']
    w_size = config.WINDOW_SIZE if is_seq else 1

    _, _, test_df, features = load_dataset(data_path, model=m_lower)
    model = load_checkpoint(
        m_lower, "none", SEED, config.TARGET_LEAD_TIME, data_path,
        input_dim=len(features), features=features,
        window_size=w_size if is_seq else None
    )
    m_type = 'pytorch_class' if is_seq else m_lower
    evaluator = RollingEvaluator(
        model, features, window_size=w_size,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model_type=m_type, seed=SEED
    )
    raw_preds = evaluator.get_raw_predictions(test_df, lead_time=config.TARGET_LEAD_TIME)

    alarm_durations = []
    for d in raw_preds:
        if d['has_failed']:
            probs = np.array(d['preds'])
            alarm_mask = probs >= thr
            c = alarm_mask.sum()
            if c > 0:
                alarm_durations.append(c)

    n_disks = len(alarm_durations)
    tot_days = sum(alarm_durations)
    mean_dur = float(np.mean(alarm_durations)) if n_disks > 0 else 0.0
    med_dur = float(np.median(alarm_durations)) if n_disks > 0 else 0.0

    return {
        "n_disks": n_disks,
        "total_alarm_days": tot_days,
        "mean_dur": mean_dur,
        "med_dur": med_dur
    }


def main():
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Calibri', 'sans-serif']
    plt.rcParams['axes.edgecolor'] = '#111111'
    plt.rcParams['axes.linewidth'] = 1.1

    print("=" * 80)
    print(" [PROCESSING] Extracting Operational Alarm Burden metrics for Dumbbell Plot...")
    print("=" * 80)

    metrics_map = {}
    for m_code, _ in MODELS:
        metrics_map[m_code] = load_alarm_burden_metrics(m_code)

    models_list = [m_title for _, m_title in MODELS]
    m_codes = [m_code for m_code, _ in MODELS]

    alarmed_disks = [metrics_map[m]["n_disks"] for m in m_codes]
    total_alarm_days = [metrics_map[m]["total_alarm_days"] for m in m_codes]
    mean_durations = [metrics_map[m]["mean_dur"] for m in m_codes]
    med_durations = [metrics_map[m]["med_dur"] for m in m_codes]

    sns.set_theme(style="ticks")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0), dpi=300)

    fig.suptitle(
        "Operational Alarm Burden and Persistence on Failed HDDs — HGST",
        fontsize=15, fontweight="bold", y=1.02, color="#111111"
    )

    # -------------------------------------------------------------------------
    # Panel (a): Number of Failed HDDs Receiving Alarms
    # -------------------------------------------------------------------------
    colors_bar = [MODEL_COLORS[m] for m in models_list]
    rects1 = ax1.bar(models_list, alarmed_disks, color=colors_bar, edgecolor="#000000", alpha=0.85, width=0.48)
    ax1.set_ylabel("Failed HDDs Receiving ≥1 Alarm (n)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Number of Failed HDDs Receiving Alarms", fontsize=12, fontweight="bold", loc="left", pad=10)
    ax1.set_ylim(0, max(alarmed_disks) * 1.25)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.30, color="#888888")
    sns.despine(ax=ax1, top=True, right=True)

    for rect, tot_d in zip(rects1, total_alarm_days):
        h = rect.get_height()
        ax1.annotate(
            f"{int(h)}\n{tot_d:,} alarm-days",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points",
            ha='center', va='bottom', fontsize=9.2, fontweight="bold", color="#111111"
        )

    # -------------------------------------------------------------------------
    # Panel (b): Alarm Persistence per Failed HDD (Dumbbell Plot: Mean vs Median)
    # -------------------------------------------------------------------------
    y_positions = np.arange(len(models_list))[::-1] # LGBM, XGB, LSTM, GRU from top to bottom

    for idx, m_title in enumerate(models_list):
        y_pos = y_positions[idx]
        med_val = med_durations[idx]
        mean_val = mean_durations[idx]
        m_color = MODEL_COLORS[m_title]

        # Horizontal connecting line (Dumbbell bar)
        ax2.plot([med_val, mean_val], [y_pos, y_pos], color="#555555", linewidth=2.2, zorder=2)

        # Median marker (Circle ●)
        ax2.scatter(
            med_val, y_pos, color=m_color, marker="o", s=110,
            edgecolor="#000000", linewidth=1.0, zorder=4,
            label="Median Persistence" if idx == 0 else ""
        )
        # Mean marker (Diamond ◆)
        ax2.scatter(
            mean_val, y_pos, color=m_color, marker="D", s=110,
            edgecolor="#000000", linewidth=1.0, zorder=4,
            label="Mean Persistence" if idx == 0 else ""
        )

        # Numerical text annotations
        ax2.annotate(
            f"{med_val:.1f}d",
            xy=(med_val, y_pos), xytext=(0, -14), textcoords="offset points",
            ha='center', va='top', fontsize=9.0, fontweight="bold", color="#111111"
        )
        ax2.annotate(
            f"{mean_val:.1f}d",
            xy=(mean_val, y_pos), xytext=(0, 11), textcoords="offset points",
            ha='center', va='bottom', fontsize=9.0, fontweight="bold", color="#111111"
        )

    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(models_list, fontsize=11, fontweight="bold", color="#111111")
    ax2.set_ylim(-0.6, len(models_list) - 0.4)
    ax2.set_xlim(0, max(mean_durations) * 1.15)
    ax2.set_xlabel("Alarm Persistence (Days / HDD)", fontsize=11, fontweight="bold", labelpad=6)
    ax2.set_title("(b) Alarm Persistence per Failed HDD", fontsize=12, fontweight="bold", loc="left", pad=10)
    ax2.grid(True, axis="x", linestyle=":", alpha=0.35, color="#888888")
    ax2.grid(False, axis="y")
    sns.despine(ax=ax2, top=True, right=True, left=False, bottom=False)

    # Custom legend for Dumbbell plot markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Median', markerfacecolor='#444444', markeredgecolor='#000000', markersize=9),
        Line2D([0], [0], marker='D', color='w', label='Mean', markerfacecolor='#444444', markeredgecolor='#000000', markersize=9)
    ]
    ax2.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=9.5)

    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "seed42_alarm_persistence_bar_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("=" * 80)
    print(" [SUCCESS] Academic Dumbbell Plot (Operational Alarm Burden & Persistence) Generated!")
    print(f" Saved to:\n  -> {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
