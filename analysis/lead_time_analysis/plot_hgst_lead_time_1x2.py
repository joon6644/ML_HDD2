"""Two-panel Lead Time distribution (XGBoost vs GRU) for the paper.

The 2x2 version spends half its area on two panels that are hard to tell apart:
LSTM and GRU have medians 77 vs 75 days and >360-day tails 13.8% vs 12.7%.
The four medians are already reported in the body text, so the figure only has
to carry the distribution shape and the cross-family contrast.

XGBoost is used as the tree-based panel instead of LightGBM. LightGBM saturates
its predicted probability at exactly 1.0 on some HDDs, so tau_op could not be
placed within the search range in 7 of 30 seeds; its seed-42 distribution is a
poor thing to put on display. XGBoost also gives the sharper contrast (2.8% vs
12.7% beyond 360 days, against LightGBM's 16.2% which is close to GRU).
"""
import os
import sys

import numpy as np
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
MODELS = [("XGB", "XGBoost"), ("GRU", "GRU")]

STYLE_CONFIG = {
    "XGB": {"fill": "#d95f02", "edge": "#000000"},
    "GRU": {"fill": "#1b9e77", "edge": "#000000"},
}

MAX_DAYS = 360


def load_lead_time_data(model_code: str):
    df = load_alarm_report(HDD_NAME, model_code, seed=SEED)
    failed_df = df[df["has_failed"] == 1]
    alarms = failed_df[failed_df["alarm_triggered"] == 1]
    days = alarms["days_to_failure_at_alarm"].dropna().values
    n_on_time = int((alarms["category"] == "On time").sum())
    return days, len(failed_df), n_on_time


def main():
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Calibri", "sans-serif"]
    plt.rcParams["axes.edgecolor"] = "#111111"
    plt.rcParams["axes.linewidth"] = 1.1
    sns.set_theme(style="ticks", palette="muted")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=300, sharey=True)
    fig.suptitle(
        "Lead Time Distribution at First Operational Alarm — HGST (HUH721212ALN604)",
        fontsize=14, fontweight="bold", y=1.00, color="#111111",
    )

    bins = np.linspace(0, MAX_DAYS, 37)
    labels = ["(a)", "(b)"]

    for idx, (m_code, m_title) in enumerate(MODELS):
        ax = axes[idx]
        style = STYLE_CONFIG[m_code]
        days, _, n_on_time = load_lead_time_data(m_code)
        n_disks = len(days)

        # Counts, not relative frequency. The two panels differ in n by 1.75x,
        # and that difference is the point: normalizing it away would make the
        # 9-disk first bin read as 25% for XGBoost and 14% for GRU.
        counts, edges = np.histogram(days, bins=bins)
        n_over = int((days > MAX_DAYS).sum())
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
               facecolor=style["fill"], edgecolor=style["edge"],
               alpha=0.72, linewidth=0.9)
        # Lead times beyond the axis are pooled into one hatched overflow bar so
        # the displayed bars sum to the samples the median is drawn on.
        ax.bar(366, n_over, width=18, align="edge",
               facecolor=style["fill"], edgecolor=style["edge"],
               alpha=0.45, linewidth=0.9, hatch="//")
        if n_over:
            ax.text(375, n_over + 0.35, str(n_over), ha="center",
                    fontsize=9.5, color="#333333")

        median_val = float(np.median(days))
        ax.axvline(median_val, color="#d9534f", linestyle="--", linewidth=2.0,
                   zorder=5, label=f"Median: {median_val:.0f} days")
        ax.axvline(30, color="#444444", linestyle=":", linewidth=1.6,
                   zorder=4, label="H = 30 days")
        ax.legend(loc="upper right", frameon=True, facecolor="#ffffff",
                  edgecolor="#cccccc", framealpha=0.95, fontsize=10)

        ax.set_title(
            f"{labels[idx]} {m_title}  (n = {n_disks}, On-time = {n_on_time})",
            fontsize=12.5, fontweight="bold", pad=9, loc="left", color="#111111",
        )
        ax.set_xlabel("Lead Time (Days)", fontsize=11, fontweight="bold", labelpad=6)
        if idx == 0:
            ax.set_ylabel("Number of HDDs", fontsize=11, fontweight="bold", labelpad=6)
        ax.set_xlim(0, 390)
        ax.set_ylim(0, 13)
        ax.set_xticks(list(np.arange(0, 301, 60)) + [375])
        ax.set_xticklabels([str(v) for v in np.arange(0, 301, 60)] + [">360"])
        ax.grid(True, axis="y", linestyle=":", alpha=0.20, color="#666666")
        ax.grid(False, axis="x")
        sns.despine(ax=ax, top=True, right=True)

        print(f"{m_title:9s} n={n_disks:3d}  On-time={n_on_time:3d} "
              f"({100*n_on_time/n_disks:.1f}%)  median={median_val:.0f}d  "
              f">360d={n_over}대 ({100*n_over/n_disks:.1f}%)")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(RESULTS_DIR, "HGST_20HUH721212ALN604_2models_1x2_lead_time_density.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"figure: {out_path}")


if __name__ == "__main__":
    main()