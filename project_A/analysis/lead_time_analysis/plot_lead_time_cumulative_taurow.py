"""Cumulative share of first alarms against lead time, at tau_row (seed 42, LightGBM).

Backs the widest claim in the paper: under a row-level FAR 1% threshold the
median first-alarm Lead Time falls outside the Prediction Horizon in all 12
combinations (32-229 d, Table 3). The figure draws that claim as a reading at a
single point: the cumulative share at H is under 0.5 for every dataset, which is
the same statement -- more than half of the first alarms arrived too early to
act on.

Three datasets, one model, because the spread that matters is between datasets:
at H the cumulative share runs 9-16% on HGST, 42-47% on Seagate and 26-34% on
Toshiba across the four models. The models are close on HGST and Seagate but
split on Toshiba (trees 33-34%, RNNs 26-27%), so the caption reports the
per-dataset range rather than claiming the four curves coincide.

Share, not count, on y: the reading at H is then directly the share of first
alarms that landed inside the horizon, and the three datasets stay comparable.
Log x because the datasets differ by an order of magnitude in scale (Seagate's
median is 39 d, HGST's 163 d) and a linear axis long enough for one squashes
the other.
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
import analysis_data_loader as adl

SEED = 42
# XGBoost, not LightGBM. LightGBM is the model the paper flags twice as an edge
# case -- probability saturation on HGST (5.2) and a row-level FAR of 0.36% on
# Toshiba -- so drawing the evidence from it invites the obvious question. And
# XGBoost is the conservative pick: at H its cumulative share is the highest of
# the four models on Seagate (47%) and Toshiba (34%), so the claim survives on
# the model least favourable to it.
MODEL = "XGB"
H = 30
DATASETS = [
    ("HGST_20HUH721212ALN604", "HGST", "#2b5c8f"),
    ("ST12000NM0007", "Seagate", "#d95f02"),
    ("TOSHIBA_20MG07ACA14TA", "Toshiba", "#1b9e77"),
]
X_LO, X_HI = 1.0, 2000.0
CACHE = os.path.join(RESULTS_DIR, "reports", f"lead_time_taurow_{MODEL}.npz")

# Labels sit above every curve, and carry a soft white backing: at H the three
# curves are climbing steeply, so a number placed there is crossed by a line no
# matter which side of the rule it goes on.
Z_TEXT = 9
HALO = dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75)
# Where the share labels start, in points right of the rule. Left-aligned from
# there, so the first digit of each label sits at the same x.
LABEL_DX = 4
# Below the shared FS_LABEL: this panel's axis labels are far longer than the
# other figures' and at 7pt they outweighed the three thin curves. Nothing here
# is set bold either -- the data is three hairlines, so bold type anywhere on
# the panel becomes the darkest thing on it.
FS_AXIS = 5.4
# Under FS_AXIS, not the shared FS_TICK. Once the axis labels came down to 5.4
# the 6.2pt ticks were the largest type on the panel, which reads as a figure
# scaled down rather than a figure drawn small: the numbers held their size
# while everything around them shrank.
FS_TICKS = 5.0


def _compute(dataset: str):
    """First-alarm lead times of failure-observed HDDs, judged at tau_row."""
    runs = adl._load_runs(SEED)
    m = runs[(runs["dataset"].astype(str).str.strip() == dataset)
             & (runs["model"].astype(str).str.upper() == MODEL)]
    if len(m) != 1:
        raise RuntimeError(f"expected 1 run row for {dataset}/{MODEL}, got {len(m)}")
    tau = float(m.iloc[0]["threshold_row_opt"])
    rep = adl.generate_alarm_report(dataset, MODEL, seed=SEED, threshold=tau)
    alarms = rep[(rep["has_failed"] == 1) & (rep["alarm_triggered"] == 1)]
    return np.sort(alarms["days_to_failure_at_alarm"].dropna().values), tau


def lead_times(refresh: bool = False) -> dict:
    """{dataset: (lead times, tau_row)}, cached to disk.

    Computing this runs inference on three datasets, which is a minute or two --
    far too slow to sit behind a one-point nudge of a label. The cache holds the
    only two things the figure needs, so tweaking layout is a one-second redraw.
    Pass --refresh (or delete the file) after retraining or after the recorded
    thresholds change; nothing here would notice on its own.
    """
    names = [ds for ds, _, _ in DATASETS]
    if not refresh and os.path.exists(CACHE):
        z = np.load(CACHE)
        if all(n in z and f"tau__{n}" in z for n in names):
            print(f"[cache] {CACHE}")
            return {n: (z[n], float(z[f"tau__{n}"])) for n in names}

    out = {n: _compute(n) for n in names}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE,
             **{n: v[0] for n, v in out.items()},
             **{f"tau__{n}": np.array(v[1]) for n, v in out.items()})
    print(f"[cache] wrote {CACHE}")
    return out


def main():
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.COL_W, 2.05), dpi=fs.DPI)
    data = lead_times(refresh="--refresh" in sys.argv)

    # Alarms left of H are On-time. The median reading is left to the 0.5 tick
    # on the axis rather than a drawn rule.
    ax.axvline(H, color=fs.ACCENT, linestyle="--", linewidth=fs.LW_RULE, zorder=5)
    # No unit on the number: the x label already declares days, and Fig. 1 sets
    # the same constant as "H = 30" for the same reason.
    ax.annotate(f"$H$ = {H}", xy=(H, 1.0), xycoords=ax.get_xaxis_transform(),
                xytext=(4, -4), textcoords="offset points", ha="left", va="top",
                fontsize=fs.FS_NOTE, color=fs.ACCENT,
                zorder=Z_TEXT, bbox=HALO)

    for dataset, label, color in DATASETS:
        d, tau = data[dataset]
        n = len(d)
        med = float(np.median(d))
        share_h = float((d <= H).mean())

        # Same-day alarms (LT = 0) are folded onto the first plotted day; the
        # axis is logarithmic and the observations are daily.
        x = np.clip(d, X_LO, None)
        xs = np.concatenate(([X_LO], x, [X_HI]))
        ys = np.concatenate(([0.0], np.arange(1, n + 1) / n, [1.0]))
        # No median in the legend. The share at H is below 0.5 for all three,
        # which *is* the statement "the median lies outside H" -- printing the
        # value as well would only invite comparison with Table 3, whose medians
        # are over 30 seeds and so differ from this single-seed example.
        ax.step(xs, ys, where="post", color=color, linewidth=fs.LW_DATA, zorder=4,
                label=f"{label} (n = {n})")
        ax.plot([H], [share_h], marker="o", markersize=3.0, color=color,
                markeredgecolor="white", markeredgewidth=fs.LW_EDGE, zorder=6)
        # Left-aligned, not right-aligned: the digits are proportional, so
        # anchoring the "%" would line up the ends of the labels and leave their
        # first digits at three different x.
        # The only bold on the panel: these three numbers are what the figure is
        # read for, and they sit on top of the curves rather than beside them.
        ax.annotate(f"{share_h*100:.0f}%", xy=(H, share_h), xytext=(LABEL_DX, -2),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=fs.FS_NOTE, fontweight="bold", color=color,
                    zorder=Z_TEXT, bbox=HALO)
        print(f"{label:8s} tau_row={tau:.4f} n={n:4d} median={med:6.0f} d  "
              f"<=H: {int((d <= H).sum()):4d} ({share_h*100:.0f}%)")

    ax.set_xscale("log")
    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(0, 1.04)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"], fontsize=FS_TICKS)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1.0"], fontsize=FS_TICKS)
    # Tighter than matplotlib's 3.5pt default. These labels are one to three
    # characters, so the default gap opens a visible empty column between them
    # and the spine and pushes the y label further out than this width affords.
    ax.tick_params(axis="y", pad=1.2)
    ax.set_xlabel("Lead time of the first alarm (days, log scale)",
                  fontsize=FS_AXIS, labelpad=fs.LABELPAD)
    ax.set_ylabel("Cumulative share", fontsize=FS_AXIS,
                  labelpad=fs.LABELPAD)
    # Smaller than FS_NOTE: three long entries would otherwise be the heaviest
    # block on a panel whose subject is three thin curves.
    ax.legend(loc="upper left", fontsize=5.0, handlelength=1.2,
              labelspacing=0.2, borderpad=0.15, borderaxespad=0.35,
              handletextpad=0.5)
    fs.style_axes(ax)
    ax.grid(False, axis="y")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "lead_time_cumulative_taurow.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
