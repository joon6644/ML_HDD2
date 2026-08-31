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
    "LGBM": {"fill": "#2b5c8f", "edge": "#000000"},
    "XGB":  {"fill": "#d95f02", "edge": "#000000"},
    "LSTM": {"fill": "#7570b3", "edge": "#000000"},
    "GRU":  {"fill": "#1b9e77", "edge": "#000000"}
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

    sns.set_theme(style="ticks", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300, sharey=True)
    
    fig.suptitle(
        "Operational False Alarm Timing Distribution — HGST (20HUH721212ALN604)",
        fontsize=15, fontweight="bold", y=0.98, color="#111111"
    )

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    labels = ["(a)", "(b)", "(c)", "(d)"]

    model_data = {}
    for m_code, _ in MODELS:
        model_data[m_code] = load_operational_false_alarm_data(m_code)

    max_days = 2600
    bins = np.linspace(0, max_days, 36)

    for idx, (m_code, m_title) in enumerate(MODELS):
        r, c = positions[idx]
        ax = axes[r, c]
        days = model_data[m_code]
        n_total = len(days)
        style = MODEL_COLORS[m_code]

        if n_total > 0:
            sns.histplot(
                x=days,
                bins=bins,
                color=style["fill"],
                edgecolor=style["edge"],
                alpha=0.75,
                linewidth=0.8,
                ax=ax
            )

        ax.set_title(
            f"{labels[idx]} {m_title}  (n = {n_total})",
            fontsize=13, fontweight="bold", pad=10, loc="left", color="#111111"
        )

        ax.set_xlabel("Days Since Observation Start (Days)", fontsize=11, fontweight="bold", labelpad=6)
        if c == 0:
            ax.set_ylabel("False Alarm Count", fontsize=11, fontweight="bold", labelpad=6)
        else:
            ax.set_ylabel("")

        ax.set_xlim(0, max_days)
        ax.set_xticks(np.arange(0, max_days + 1, 500))

        ax.label_outer()
        ax.set_xlabel("Days Since Observation Start (Days)", fontsize=11, fontweight="bold", labelpad=6)

        ax.grid(True, axis="y", linestyle=":", alpha=0.20, color="#666666")
        ax.grid(False, axis="x")
        sns.despine(ax=ax, top=True, right=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(RESULTS_DIR, "seed42_operational_false_alarm_timing_grid.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("=" * 80)
    print(" [SUCCESS] Operational False Alarm Timing Grid (Unified Color) Generated!")
    print(f"  - LightGBM: Total FA = {len(model_data['LGBM'])}")
    print(f"  - XGBoost : Total FA = {len(model_data['XGB'])}")
    print(f"  - LSTM    : Total FA = {len(model_data['LSTM'])}")
    print(f"  - GRU     : Total FA = {len(model_data['GRU'])}")
    print(f" Saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
