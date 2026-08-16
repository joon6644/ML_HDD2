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

STYLE_CONFIG = {
    "LGBM": {"fill": "#2b5c8f", "edge": "#000000"},
    "XGB":  {"fill": "#d95f02", "edge": "#000000"},
    "LSTM": {"fill": "#7570b3", "edge": "#000000"},
    "GRU":  {"fill": "#1b9e77", "edge": "#000000"}
}


def load_lead_time_data(model_code: str):
    df = load_alarm_report(HDD_NAME, model_code, seed=SEED)
    failed_df = df[df['has_failed'] == 1]
    total_failed = len(failed_df)
    failed_alarms = failed_df[failed_df['alarm_triggered'] == 1]
    days = failed_alarms['days_to_failure_at_alarm'].dropna().values
    return days, total_failed


def main():
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Calibri', 'sans-serif']
    plt.rcParams['axes.edgecolor'] = '#111111'
    plt.rcParams['axes.linewidth'] = 1.1

    sns.set_theme(style="ticks", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300, sharey=True)
    
    fig.suptitle(
        "Lead Time Distribution at First Operational Alarm — HGST (20HUH721212ALN604)",
        fontsize=16, fontweight="bold", y=0.98, color="#111111"
    )

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    labels = ["(a)", "(b)", "(c)", "(d)"]

    model_data = {}
    model_total_failed = {}
    for m_code, _ in MODELS:
        days, total_failed = load_lead_time_data(m_code)
        model_data[m_code] = days
        model_total_failed[m_code] = total_failed

    max_days = 360
    bins = np.linspace(0, max_days, 37)

    for idx, (m_code, m_title) in enumerate(MODELS):
        r, c = positions[idx]
        ax = axes[r, c]
        style = STYLE_CONFIG[m_code]
        days = model_data[m_code]
        n_disks = len(days)
        total_failed = model_total_failed[m_code]

        if n_disks > 0:
            counts, edges = np.histogram(days, bins=bins)
            pct = counts / len(days) * 100.0
            over_pct = (days > 360).sum() / len(days) * 100.0
            ax.bar(
                edges[:-1], pct, width=np.diff(edges), align="edge",
                facecolor=style["fill"], edgecolor=style["edge"],
                alpha=0.72, linewidth=0.9
            )
            # Lead times beyond the axis are pooled into one hatched overflow
            # bar, so displayed bars always sum to 100% of the samples that the
            # median line is computed on.
            ax.bar(
                366, over_pct, width=18, align="edge",
                facecolor=style["fill"], edgecolor=style["edge"],
                alpha=0.45, linewidth=0.9, hatch="//"
            )
            median_val = np.median(days)
            ax.axvline(
                median_val,
                color="#d9534f",
                linestyle="--",
                linewidth=2.0,
                zorder=5,
                label=f"Median (per-disk): {median_val:.1f} days"
            )
            ax.axvline(30, color="#444444", linestyle=":", linewidth=1.6, zorder=4, label="H = 30 days")
            ax.legend(
                loc="upper right",
                frameon=True,
                facecolor="#ffffff",
                edgecolor="#cccccc",
                framealpha=0.95,
                fontsize=10.5
            )

        ax.set_title(
            f"{labels[idx]} {m_title}  (n = {n_disks})",
            fontsize=13, fontweight="bold", pad=10, loc="left", color="#111111"
        )

        ax.set_xlabel("Lead Time (Days)", fontsize=11, fontweight="bold", labelpad=6)
        if c == 0:
            ax.set_ylabel("Frequency (%)", fontsize=11, fontweight="bold", labelpad=6)
        else:
            ax.set_ylabel("")

        ax.set_xlim(0, 390)
        ax.set_xticks(list(np.arange(0, 301, 60)) + [375])
        ax.set_xticklabels([str(v) for v in np.arange(0, 301, 60)] + [">360"])
        ax.label_outer()
        ax.set_xlabel("Lead Time (Days)", fontsize=11, fontweight="bold", labelpad=6)

        ax.grid(True, axis="y", linestyle=":", alpha=0.20, color="#666666")
        ax.grid(False, axis="x")
        sns.despine(ax=ax, top=True, right=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(RESULTS_DIR, "HGST_20HUH721212ALN604_4models_2x2_lead_time_density.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("=" * 80)
    print(" [SUCCESS] Clean Y-axis Label Outer 2x2 Density Plot Generated!")
    print(f" Saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
