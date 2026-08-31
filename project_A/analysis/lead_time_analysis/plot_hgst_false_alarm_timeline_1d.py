import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis", "lead_time_analysis")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
if ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, ANALYSIS_DIR)

from analysis_data_loader import load_alarm_report

HDD_NAME = "HGST_20HUH721212ALN604"
SEED = 42
MODELS = [
    ("LGBM", "LightGBM"),
    ("XGB", "XGBoost"),
    ("LSTM", "LSTM"),
    ("GRU", "GRU")
]

MODEL_COLORS = {
    "LGBM": "#2b5c8f",
    "XGB":  "#d95f02",
    "LSTM": "#7570b3",
    "GRU":  "#1b9e77"
}


def load_operational_false_alarm_data(model_code: str) -> np.ndarray:
    df = load_alarm_report(HDD_NAME, model_code, seed=SEED)
    
    # HDD-level FAR population: Censored Early only. Early alarms on failed
    # HDDs are not false alarms under the operational FAR definition; they
    # are penalized through Recall/Precision instead (METRIC_DESIGN.md).
    censored_fa = df[(df['has_failed'] == 0) & (df['alarm_triggered'] == 1)]
    days_since = censored_fa['days_since_observed'].dropna().values
    return days_since[days_since >= 0]


def main():
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Calibri', 'sans-serif']
    plt.rcParams['axes.edgecolor'] = '#111111'
    plt.rcParams['axes.linewidth'] = 1.1

    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(14, 5.0), dpi=300)
    
    fig.suptitle(
        "Temporal Distribution of Operational False Alarms — HGST (20HUH721212ALN604)",
        fontsize=15, fontweight="bold", y=0.98, color="#111111"
    )

    y_positions = [3, 2, 1, 0] # LGBM, XGB, LSTM, GRU from top to bottom
    y_labels = []

    model_data = {}
    for idx, (m_code, m_title) in enumerate(MODELS):
        days = load_operational_false_alarm_data(m_code)
        n_total = len(days)
        model_data[m_code] = days
        y_labels.append(f"{m_title} (n = {n_total})")

    max_days = 2600

    # Draw horizontal timeline baseline for each model
    for y_pos in y_positions:
        ax.axhline(y_pos, color="#e0e0e0", linestyle="-", linewidth=1.5, zorder=1)

    # Plot unified false alarm dots on model timeline rows
    for idx, (m_code, m_title) in enumerate(MODELS):
        y_pos = y_positions[idx]
        days = model_data[m_code]
        color = MODEL_COLORS[m_code]

        if len(days) > 0:
            ax.scatter(
                days, np.full_like(days, y_pos),
                color=color,
                marker="o",
                s=85,
                alpha=0.85,
                edgecolor="#000000",
                linewidth=0.8,
                zorder=4
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=12, fontweight="bold", color="#111111")
    ax.set_ylim(-0.6, 3.6)

    ax.set_xlabel("Days Since Observation Start (Days)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_xlim(0, max_days)
    ax.set_xticks(np.arange(0, max_days + 1, 500))

    ax.grid(True, axis="x", linestyle=":", alpha=0.35, color="#888888")
    ax.grid(False, axis="y")
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(RESULTS_DIR, "seed42_false_alarm_timeline_1d.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("=" * 80)
    print(" [SUCCESS] Clean Unified 1D False Alarm Timeline Generated!")
    print(f" Saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
