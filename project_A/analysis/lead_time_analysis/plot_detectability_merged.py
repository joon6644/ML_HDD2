"""Single-panel version of the per-HDD detectability figure.

The 1x2 layout spends a whole second axis on what is really one distribution
with a dominant first bin: 109-119 HDDs are never detected On-time while no
other bin exceeds 27. A broken y-axis carries both on one x-axis, which fits a
single column in a 2-column layout instead of forcing a full-width break.

The gray/colour split of the 0 bin (no alarm at all vs alarmed but always Early)
moves to the body text -- it is four pairs of numbers, not a shape.
"""
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
SEEDS = list(range(42, 72))
MODELS = [("LGBM", "LightGBM"), ("XGB", "XGBoost"), ("LSTM", "LSTM"), ("GRU", "GRU")]
MODEL_COLORS = {"LGBM": "#2b5c8f", "XGB": "#d95f02", "LSTM": "#7570b3", "GRU": "#1b9e77"}
BAR_ALPHA, BAR_EDGE = 0.72, "#000000"

BINS = [(0, 0, "0"), (1, 5, "1–5"), (6, 10, "6–10"), (11, 15, "11–15"),
        (16, 20, "16–20"), (21, 25, "21–25"), (26, 30, "26–30")]


def counts(model_code):
    frames = []
    for seed in SEEDS:
        rep = load_alarm_report(HDD_NAME, model_code, seed=seed)
        frames.append(rep[rep["has_failed"] == 1][["serial_number", "category"]])
    d = pd.concat(frames)
    piv = d.pivot_table(index="serial_number", columns="category",
                        aggfunc="size", fill_value=0)
    for c in ("On time", "Missed"):
        if c not in piv:
            piv[c] = 0
    return piv


def main():
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
    sns.set_theme(style="ticks", palette="muted")

    per = {code: counts(code) for code, _ in MODELS}
    n_failed = len(per["LGBM"])

    # Sized for one column of a 2-column layout (~8.6 cm), so the figure sits
    # inside a single column and the other column keeps flowing. Rendering at
    # the final physical size keeps the type at its intended point size.
    fig, (hi, lo) = plt.subplots(
        2, 1, figsize=(3.4, 3.9), dpi=400, sharex=True,
        gridspec_kw={"height_ratios": [0.55, 2.0], "hspace": 0.09},
    )
    fig.suptitle("Per-HDD On-time detection, 30 seeds\n"
                 f"HGST (HUH721212ALN604), {n_failed} failed HDDs",
                 fontsize=8, fontweight="bold", y=1.005, linespacing=1.35)

    xs = np.arange(len(BINS))
    width = 0.2
    for i, (code, title) in enumerate(MODELS):
        oc = per[code]["On time"]
        h = [int(((oc >= lo_) & (oc <= hi_)).sum()) for lo_, hi_, _ in BINS]
        off = (i - 1.5) * width
        for ax in (hi, lo):
            ax.bar(xs + off, h, width=width, color=MODEL_COLORS[code],
                   alpha=BAR_ALPHA, edgecolor=BAR_EDGE, linewidth=0.35,
                   zorder=3, label=title if ax is lo else None)
        # Only the 0 bin and GRU's 26-30 spike are annotated; at one-column
        # width a label on all 28 bars collides.
        hi.text(xs[0] + off, h[0] + 1.5, str(h[0]), ha="center",
                fontsize=5.8, fontweight="bold", color="#333333")
    gru = per["GRU"]["On time"]
    lo.text(xs[-1] + 1.5 * width, int((gru >= 26).sum()) + 0.6,
            str(int((gru >= 26).sum())), ha="center", fontsize=6,
            fontweight="bold", color="#333333")

    hi.set_ylim(100, 126)
    lo.set_ylim(0, 30)
    hi.set_yticks([110, 120])
    hi.spines["bottom"].set_visible(False)
    lo.spines["top"].set_visible(False)
    hi.tick_params(bottom=False)

    # Diagonal break marks on the shared edge.
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=5, linestyle="none",
              color="#111111", mec="#111111", mew=0.9, clip_on=False)
    hi.plot([0, 1], [0, 0], transform=hi.transAxes, **kw)
    lo.plot([0, 1], [1, 1], transform=lo.transAxes, **kw)

    lo.set_xticks(xs)
    lo.set_xticklabels([b[2] for b in BINS], fontsize=6.2)
    lo.set_xlabel("Seeds detected On-time (out of 30)",
                  fontsize=7, fontweight="bold", labelpad=3)
    hi.tick_params(labelsize=6.2)
    lo.tick_params(labelsize=6.2)
    fig.supylabel("Number of failed HDDs", fontsize=7, fontweight="bold", x=0.015)
    lo.legend(loc="upper right", fontsize=6, frameon=True, ncol=2,
              handlelength=1.1, columnspacing=0.8, handletextpad=0.4,
              borderpad=0.4, labelspacing=0.3)
    for ax in (hi, lo):
        ax.grid(True, axis="y", linestyle=":", alpha=0.3, linewidth=0.5)
        sns.despine(ax=ax, top=True, right=True)

    plt.tight_layout(rect=[0.015, 0, 1, 0.93])
    out = os.path.join(RESULTS_DIR, f"hdd_detectability_merged_{HDD_NAME}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    for code, title in MODELS:
        oc, ms = per[code]["On time"], per[code]["Missed"]
        print(f"{title:9s} never On-time {int((oc==0).sum()):3d}  "
              f"(no alarm {int((ms==len(SEEDS)).sum()):3d})")
    print(f"figure: {out}")


if __name__ == "__main__":
    main()