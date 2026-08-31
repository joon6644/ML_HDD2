"""Lead Time of the first alarm, XGBoost and GRU in one panel (HGST, seed 42).

Supersedes plot_hgst_lead_time_2x1.py. The journal now wants a caption under
every plot, so a stacked pair costs two figure numbers and two bilingual
captions to show one comparison. Overlaying is not just cheaper here, it is
better: the claim in 5.4 is that the RNN raises a first alarm on ~1.8x as many
disks over a wider spread, and on a shared axis that is one glance instead of
two. The two panels were previously 8.8cm tall at column width; this is ~4.5cm.

Counts, not density: the series differ in n by 1.75x and that difference is the
finding, so normalising them away would delete the point.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis", "lead_time_analysis")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
if ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, ANALYSIS_DIR)

import figure_style as fs
from analysis_data_loader import load_alarm_report

HDD_NAME = "HGST_20HUH721212ALN604"
SEED = 42
MODELS = [("XGB", "XGBoost", "#d95f02"), ("GRU", "GRU", "#1b9e77")]
MAX_DAYS = 360
N_BINS = 36

# Overflow bars for the >360 d tail, dodged so the two series stay separable.
OVF_L, OVF_W = 366.0, 9.0


def load(model_code):
    df = load_alarm_report(HDD_NAME, model_code, seed=SEED)
    alarms = df[(df["has_failed"] == 1) & (df["alarm_triggered"] == 1)]
    days = alarms["days_to_failure_at_alarm"].dropna().values
    return days, int((alarms["category"] == "On time").sum())


def main():
    fs.apply()

    fig, ax = plt.subplots(figsize=(fs.COL_W, 1.95), dpi=fs.DPI)
    bins = np.linspace(0, MAX_DAYS, N_BINS + 1)

    for i, (code, title, color) in enumerate(MODELS):
        days, n_on = load(code)
        n = len(days)
        med = float(np.median(days))
        n_over = int((days > MAX_DAYS).sum())

        # Fill and outline drawn separately: a stepfilled patch applies its alpha
        # to the edge too, which leaves the two outlines too faint to trace where
        # they overlap.
        ax.hist(days, bins=bins, histtype="stepfilled", color=color,
                alpha=fs.FILL_ALPHA + 0.15, edgecolor="none", zorder=2 + i)
        ax.hist(days, bins=bins, histtype="step", color=color,
                linewidth=fs.LW_DATA, zorder=4 + i,
                label=f"{title} (n = {n}, On-time = {n_on})")

        ax.bar(OVF_L + i * OVF_W, n_over, width=OVF_W, align="edge",
               facecolor=color, alpha=0.40, edgecolor=color,
               linewidth=fs.LW_EDGE, hatch="///", zorder=3)
        if n_over:
            ax.text(OVF_L + (i + 0.5) * OVF_W, n_over + 0.35, str(n_over),
                    ha="center", fontsize=fs.FS_NOTE, color=color, fontweight="bold")

        # Median in the series colour, so no legend lookup is needed to tell the
        # two rules apart.
        ax.axvline(med, color=color, linestyle="--", linewidth=fs.LW_RULE, zorder=6)
        ax.text(med + 5, 12.4 - i * 1.9, f"{med:.0f} d", fontsize=fs.FS_NOTE,
                color=color, fontweight="bold", va="top")
        print(f"{title:9s} n={n:3d} On-time={n_on:3d} median={med:.0f}d >360d={n_over}")

    ax.set_xlim(0, 393)
    ax.set_ylim(0, 13.2)
    ax.set_yticks([0, 5, 10])
    ax.set_xticks(list(np.arange(0, 301, 60)) + [OVF_L + OVF_W])
    ax.set_xticklabels([str(v) for v in np.arange(0, 301, 60)] + [">360"],
                       fontsize=fs.FS_TICK)
    ax.set_xlabel("Lead time (days)", fontsize=fs.FS_LABEL,
                  fontweight="bold", labelpad=fs.LABELPAD)
    ax.set_ylabel("Number of HDDs", fontsize=fs.FS_LABEL,
                  fontweight="bold", labelpad=fs.LABELPAD)
    ax.legend(loc="upper right", handlelength=1.4, labelspacing=0.25,
              borderpad=0.2, borderaxespad=0.3)
    fs.style_axes(ax)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{HDD_NAME}_lead_time_single.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
