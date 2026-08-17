"""Lead Time distribution as a 2x1 stack (XGBoost over GRU), one column wide.

Both panels share the same x-axis (Lead Time in days), so stacking them lets
the axis label, tick labels and the H = 30 marker be drawn once instead of
twice. The result is narrow enough to sit inside a single column of a
2-column layout, which halves the page cost of the side-by-side version and
lets the neighbouring column keep flowing.

Counts, not relative frequency: the two panels differ in n by 1.75x and that
difference is the point (see plot_hgst_lead_time_1x2.py).
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
STYLE = {"XGB": "#d95f02", "GRU": "#1b9e77"}
BAR_ALPHA, BAR_EDGE = 0.72, "#000000"
MAX_DAYS = 360


def load(model_code):
    df = load_alarm_report(HDD_NAME, model_code, seed=SEED)
    alarms = df[(df["has_failed"] == 1) & (df["alarm_triggered"] == 1)]
    days = alarms["days_to_failure_at_alarm"].dropna().values
    return days, int((alarms["category"] == "On time").sum())


def main():
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.edgecolor"] = "#111111"
    plt.rcParams["axes.linewidth"] = 0.8
    sns.set_theme(style="ticks", palette="muted")

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.7), dpi=400,
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.18})
    fig.suptitle("Lead time at first operational alarm\n"
                 "HGST (HUH721212ALN604), seed 42",
                 fontsize=8, fontweight="bold", y=1.005, linespacing=1.35)

    bins = np.linspace(0, MAX_DAYS, 37)
    labels = ["(a)", "(b)"]

    for idx, (code, title) in enumerate(MODELS):
        ax = axes[idx]
        days, n_on = load(code)
        n = len(days)
        counts, edges = np.histogram(days, bins=bins)
        n_over = int((days > MAX_DAYS).sum())

        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
               facecolor=STYLE[code], alpha=BAR_ALPHA, edgecolor=BAR_EDGE,
               linewidth=0.35, zorder=3)
        ax.bar(366, n_over, width=18, align="edge", facecolor=STYLE[code],
               alpha=0.45, edgecolor=BAR_EDGE, linewidth=0.35, hatch="//", zorder=3)
        if n_over:
            ax.text(375, n_over + 0.4, str(n_over), ha="center",
                    fontsize=5.8, color="#333333")

        med = float(np.median(days))
        ax.axvline(med, color="#d9534f", linestyle="--", linewidth=1.1, zorder=5)
        ax.text(med + 8, 11.4, f"median {med:.0f} d", fontsize=5.8,
                color="#c0392b", fontweight="bold", va="top")

        ax.set_title(f"{labels[idx]} {title}  (n = {n}, On-time = {n_on})",
                     fontsize=7, fontweight="bold", loc="left", pad=3)
        ax.set_ylim(0, 12.2)
        ax.set_yticks([0, 5, 10])
        ax.tick_params(labelsize=6.2)
        ax.grid(True, axis="y", linestyle=":", alpha=0.25, linewidth=0.5)
        sns.despine(ax=ax, top=True, right=True)
        print(f"{title:9s} n={n:3d} On-time={n_on:3d} median={med:.0f}d >360d={n_over}")

    # x-axis drawn once, at the bottom
    ax = axes[-1]
    ax.set_xlim(0, 390)
    ax.set_xticks(list(np.arange(0, 301, 60)) + [375])
    ax.set_xticklabels([str(v) for v in np.arange(0, 301, 60)] + [">360"], fontsize=6.2)
    ax.set_xlabel("Lead time (days)", fontsize=7, fontweight="bold", labelpad=3)
    fig.supylabel("Number of HDDs", fontsize=7, fontweight="bold", x=0.015)

    plt.tight_layout(rect=[0.015, 0, 1, 0.93])
    out = os.path.join(RESULTS_DIR, f"{HDD_NAME}_lead_time_2x1.png")
    plt.savefig(out, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"figure: {out}")


if __name__ == "__main__":
    main()