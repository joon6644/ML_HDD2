"""Cumulative first-alarm count against lead time, at tau_row (HGST, seed 42).

This figure backs the widest claim the paper makes: under a row-level FAR 1%
threshold, the median first-alarm Lead Time sits outside the Prediction Horizon
in all 12 combinations (32-229 d, Table 3). So it has to be drawn at tau_row.
An earlier version used tau_op, which put the paper's only data figure under its
narrowest claim while the widest one had none -- and mixed the two threshold
criteria on facing pages.

A cumulative count answers by inspection the question a histogram makes the
reader integrate: how many alarms had arrived by the time the horizon closed.
The height at H is the On-time count, the right end is the number of disks that
got a first alarm at all inside the plotted range, and the gap between them is
the Early group. Monotone curves also cannot obscure each other.

tau_row is read from the same runs table the experiments wrote; the alarm report
is then judged at that threshold rather than at the disk-optimal one the cached
reports use.
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

HDD_NAME = "HGST_20HUH721212ALN604"
SEED = 42
H = 30
MODELS = [("XGB", "XGBoost", "#d95f02"), ("GRU", "GRU", "#1b9e77")]
X_MAX = 720          # two years; at tau_row the alarms run much further out
X_TICK = 120


def tau_row(model_code: str) -> float:
    """Row-level FAR 1% threshold recorded for this (dataset, model, seed)."""
    df = adl._load_runs(SEED)
    m = df[(df["dataset"].astype(str).str.strip() == HDD_NAME)
           & (df["model"].astype(str).str.upper() == model_code.upper())]
    if len(m) != 1:
        raise RuntimeError(f"expected 1 run row for {model_code}, got {len(m)}")
    return float(m.iloc[0]["threshold_row_opt"])


def lead_times(model_code: str):
    t = tau_row(model_code)
    rep = adl.generate_alarm_report(HDD_NAME, model_code, seed=SEED, threshold=t)
    alarms = rep[(rep["has_failed"] == 1) & (rep["alarm_triggered"] == 1)]
    return np.sort(alarms["days_to_failure_at_alarm"].dropna().values), t


def main():
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.COL_W, 2.0), dpi=fs.DPI)

    ax.axvline(H, color=fs.ACCENT, linestyle="--", linewidth=fs.LW_RULE, zorder=2)
    ax.annotate(f"H = {H} d", xy=(H, 1.0), xycoords=ax.get_xaxis_transform(),
                xytext=(3, -1), textcoords="offset points", ha="left", va="top",
                fontsize=fs.FS_NOTE, fontweight="bold", color=fs.ACCENT)

    y_top = 0
    for code, title, color in MODELS:
        d, t = lead_times(code)
        n, med = len(d), float(np.median(d))
        n_h = int((d <= H).sum())
        n_in = int((d <= X_MAX).sum())
        y_top = max(y_top, n_in)

        xs = np.concatenate(([0.0], d[d <= X_MAX], [X_MAX]))
        ys = np.concatenate(([0.0], np.arange(1, n_in + 1), [n_in]))
        ax.step(xs, ys, where="post", color=color, linewidth=fs.LW_DATA,
                zorder=4, label=f"{title} (n = {n}, median {med:.0f} d)")

        # The number the panel exists to show: how far the curve has risen by the
        # time the horizon closes.
        ax.plot([H], [n_h], marker="o", markersize=3.0, color=color,
                markeredgecolor="white", markeredgewidth=fs.LW_EDGE, zorder=6)
        ax.annotate(str(n_h), xy=(H, n_h), xytext=(5, -1),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=fs.FS_NOTE, fontweight="bold", color=color)
        print(f"{title:8s} tau_row={t:.4f} n={n:3d} median={med:.0f}d  "
              f"<=H:{n_h:3d} ({100*n_h/n:.0f}%)  <={X_MAX}:{n_in:3d}  beyond:{n - n_in}")

    ax.set_xlim(0, X_MAX + 20)
    ax.set_ylim(0, y_top * 1.12)
    ax.set_xticks(np.arange(0, X_MAX + 1, X_TICK))
    ax.set_xlabel("Lead time of the first alarm (days)", fontsize=fs.FS_LABEL,
                  fontweight="bold", labelpad=fs.LABELPAD)
    ax.set_ylabel("Cumulative HDDs", fontsize=fs.FS_LABEL,
                  fontweight="bold", labelpad=fs.LABELPAD)
    ax.legend(loc="lower right", handlelength=1.4, labelspacing=0.25,
              borderpad=0.2, borderaxespad=0.4)
    fs.style_axes(ax)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{HDD_NAME}_lead_time_cumulative_taurow.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
