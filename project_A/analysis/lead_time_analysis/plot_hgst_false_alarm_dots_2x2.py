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
    "LGBM": {"dot": "#2b5c8f", "edge": "#000000"},
    "XGB":  {"dot": "#d95f02", "edge": "#000000"},
    "LSTM": {"dot": "#7570b3", "edge": "#000000"},
    "GRU":  {"dot": "#1b9e77", "edge": "#000000"}
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=300, sharey=True)
    
    fig.suptitle(
        "Operational False Alarm Event Timeline Scatter Plot — HGST (20HUH721212ALN604)",
        fontsize=15, fontweight="bold", y=0.98, color="#111111"
    )

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    labels = ["(a)", "(b)", "(c)", "(d)"]

    model_data = {}
    for m_code, _ in MODELS:
        model_data[m_code] = load_operational_false_alarm_data(m_code)

    max_days = 2600

    np.random.seed(42) # Reproducible vertical jitter

    for idx, (m_code, m_title) in enumerate(MODELS):
        r, c = positions[idx]
        ax = axes[r, c]
        days = model_data[m_code]
        n_total = len(days)
        style = MODEL_COLORS[m_code]

        if n_total > 0:
            # Deterministic/clean vertical jitter to prevent point overlap
            y_jitter = np.linspace(0.2, 0.8, n_total) if n_total > 1 else np.array([0.5])
            
            # Scatter points
            ax.scatter(
                days, y_jitter,
                color=style["dot"],
                edgecolor=style["edge"],
                s=80,
                alpha=0.85,
                linewidth=1.0,
                zorder=4,
                label=f"False Alarm Event (n = {n_total})"
            )

            # Rug plot tick lines along bottom
            ax.vlines(
                days, ymin=0.0, ymax=0.15,
                colors=style["dot"], linewidth=1.2, alpha=0.6, zorder=3
            )

        ax.set_title(
            f"{labels[idx]} {m_title}  (n = {n_total})",
            fontsize=13, fontweight="bold", pad=10, loc="left", color="#111111"
        )

        ax.set_xlabel("Days Since Observation Start (Days)", fontsize=11, fontweight="bold", labelpad=6)
        if c == 0:
            ax.set_ylabel("Event Distribution", fontsize=11, fontweight="bold", labelpad=6)
        else:
            ax.set_ylabel("")

        ax.set_xlim(0, max_days)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([]) # Hide arbitrary Y ticks

        ax.set_xticks(np.arange(0, max_days + 1, 500))

        ax.label_outer()
        ax.set_xlabel("Days Since Observation Start (Days)", fontsize=11, fontweight="bold", labelpad=6)

        ax.grid(True, axis="x", linestyle=":", alpha=0.30, color="#888888")
        ax.axhline(0.15, color="#cccccc", linestyle="-", linewidth=0.8, zorder=1)

        sns.despine(ax=ax, top=True, right=True, left=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(RESULTS_DIR, "seed42_false_alarm_dot_plot_grid.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("=" * 80)
    print(" [SUCCESS] Operational False Alarm Dot Plot (Scatter Timeline) Generated!")
    print(f"  - LightGBM: n = {len(model_data['LGBM'])}")
    print(f"  - XGBoost : n = {len(model_data['XGB'])}")
    print(f"  - LSTM    : n = {len(model_data['LSTM'])}")
    print(f"  - GRU     : n = {len(model_data['GRU'])}")
    print(f" Saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
