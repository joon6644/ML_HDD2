"""Per-HDD detectability across 30 training seeds (HGST, tau_op).

Row-level evaluation cannot ask this question at all: it loses disk identity, so
it cannot separate "the same drives are missed every time" from "a different
drive is missed each run". Here every failed test HDD is followed across all
30 seeds and 4 models, and classified by how often it was detected On-time.

Figure layout: the two findings live on incompatible scales -- the never-On-time
group is 109-119 disks while no other bin exceeds 27 -- so a shared axis buries
the second one. They are split into two panels instead of four per-model panels,
which also keeps all four models in view (the redundant pair here is the trees,
not the RNNs, so dropping models would not work cleanly).

Outputs
  - per-HDD detection counts (csv)
  - per-model summary: all-Missed / never-On-time / any-On-time / all-On-time
  - cross-model summary over all 120 runs
  - 1x2 figure: (a) never-On-time breakdown, (b) detection-count distribution
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
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
if ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, ANALYSIS_DIR)

import figure_style as fs
from analysis_data_loader import load_alarm_report

HDD_NAME = "HGST_20HUH721212ALN604"
SEEDS = list(range(42, 72))
MODELS = [("LGBM", "LightGBM"), ("XGB", "XGBoost"), ("LSTM", "LSTM"), ("GRU", "GRU")]
MODEL_COLORS = {"LGBM": "#2b5c8f", "XGB": "#d95f02", "LSTM": "#7570b3", "GRU": "#1b9e77"}
GRAY = "#b0b0b0"
# Matches plot_hgst_lead_time_1x2.py so the two figures read as one palette.
# At this alpha the fills are too light for white in-bar labels, hence LABEL_INK.
BAR_ALPHA = 0.72
BAR_EDGE = fs.BAR_EDGE
LABEL_INK = "#1a1a1a"

BINS = [(1, 5, "1–5"), (6, 10, "6–10"), (11, 15, "11–15"),
        (16, 20, "16–20"), (21, 25, "21–25"), (26, 30, "26–30")]


def collect(model_code: str) -> pd.DataFrame:
    """One row per (seed, failed HDD) with its judgement category."""
    frames = []
    for seed in SEEDS:
        rep = load_alarm_report(HDD_NAME, model_code, seed=seed)
        f = rep[rep["has_failed"] == 1][["serial_number", "category"]].copy()
        f["seed"] = seed
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def saturation(runs: np.ndarray, reps: int = 400, seed: int = 0) -> np.ndarray:
    """Disks covered after k runs, averaged over random orderings of the runs.

    The three-group split in 5.3 is defined over a finite number of training
    runs, so "never alarmed in any run" is a statement about this experiment and
    not about the population -- run it longer and some of those disks would
    eventually alarm. What can be checked is whether the count has stopped
    moving, which is what this measures: `runs` is a (run x disk) boolean of
    "covered in this run", and the return is the expected size of the union
    after k runs for k = 1..len(runs).
    """
    rng = np.random.default_rng(seed)
    n_runs = runs.shape[0]
    acc = np.zeros(n_runs)
    for _ in range(reps):
        order = rng.permutation(n_runs)
        acc += np.logical_or.accumulate(runs[order], axis=0).sum(axis=1)
    return acc / reps


def counts_per_hdd(long_df: pd.DataFrame) -> pd.DataFrame:
    piv = long_df.pivot_table(index="serial_number", columns="category",
                              aggfunc="size", fill_value=0)
    for c in ("On time", "Early", "Missed"):
        if c not in piv:
            piv[c] = 0
    return piv[["On time", "Early", "Missed"]]


def main():
    fs.apply()

    per_model, summary, long_frames = {}, [], []
    for code, title in MODELS:
        print(f"[collect] {code}", flush=True)
        long_df = collect(code)
        long_frames.append(long_df)
        piv = counts_per_hdd(long_df)
        per_model[code] = piv

        all_missed = piv["Missed"] == len(SEEDS)
        any_on = piv["On time"] > 0
        detected = piv.loc[any_on, "On time"]
        summary.append({
            "model": title, "n_failed": len(piv),
            "all_Missed": int(all_missed.sum()),
            "alarmed_but_never_On_time": int(((~any_on) & ~all_missed).sum()),
            "never_On_time": int((~any_on).sum()),
            "any_On_time": int(any_on.sum()),
            "all_On_time": int((piv["On time"] == len(SEEDS)).sum()),
            "median_On_time_count_among_detected":
                float(detected.median()) if len(detected) else np.nan,
        })

    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(REPORTS_DIR, f"detectability_per_model_{HDD_NAME}.csv"),
              index=False, encoding="utf-8-sig")

    # ---- cross-model: 120 runs -------------------------------------------
    n_runs = len(SEEDS) * len(MODELS)
    cross = pd.DataFrame({
        "on_time_runs": sum(per_model[c]["On time"] for c, _ in MODELS),
        "missed_runs": sum(per_model[c]["Missed"] for c, _ in MODELS),
    })
    cross["never_on_time_any_run"] = cross["on_time_runs"] == 0
    cross["never_alarmed_any_run"] = cross["missed_runs"] == n_runs
    cross.to_csv(os.path.join(REPORTS_DIR, f"detectability_per_hdd_{HDD_NAME}.csv"),
                 encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print(sm.to_string(index=False))
    print("-" * 78)
    print(f"[전체 {n_runs}회 실행 기준]  고장 관측 HDD {len(cross)}대")
    print(f"  한 번도 On-time 아님 : {int(cross['never_on_time_any_run'].sum())}대 "
          f"({100*cross['never_on_time_any_run'].mean():.0f}%)")
    print(f"  한 번도 Alarm 없음   : {int(cross['never_alarmed_any_run'].sum())}대 "
          f"({100*cross['never_alarmed_any_run'].mean():.0f}%)")

    # ---- is the "never alarmed" group an artefact of running only 120 times? --
    n_hdd = len(cross)
    alarmed_runs = []
    for df in long_frames:
        for _, g in df.groupby("seed", sort=True):
            g = g.set_index("serial_number")["category"].reindex(cross.index)
            alarmed_runs.append((g != "Missed").to_numpy())
    A = np.array(alarmed_runs)
    cov = saturation(A)
    n_runs = len(cov)
    alarm_count = A.sum(axis=0)

    print("-" * 78)
    print("[포화] 무작위 순서로 실행을 누적했을 때 한 번도 Alarm이 없는 HDD 수")
    for k in (1, 30, 60, 90, n_runs):
        print(f"  {k:3d}회 : {n_hdd - cov[k-1]:5.1f}대")
    print(f"  마지막 30회로 새로 Alarm이 발생한 HDD : {cov[-1] - cov[-31]:.2f}대 "
          f"(1회당 {cov[-1] - cov[-2]:.3f}대)")
    print(f"  120회 중 1회만 Alarm : {int((alarm_count == 1).sum())}대   "
          f"전 실행에서 Alarm : {int((alarm_count == n_runs).sum())}대")
    print("=" * 78)

    # ---- figure -----------------------------------------------------------
    # One panel only. The never-On-time breakdown that used to sit above this
    # one is gone: every number it carried (109 / 87 / 22 / 57 disks) is stated
    # in the text, the text argues over the four-model union rather than the
    # per-model bars the panel drew, and the journal now charges a full figure
    # number and caption per plot. What cannot be written out is this panel --
    # the shape of the tail, i.e. that only the GRU has mass at the right edge.
    # A median of 5 vs 12 does not show that.
    fig, ax = plt.subplots(figsize=(fs.COL_W, 1.95), dpi=fs.DPI)

    width = 0.2
    bxs = np.arange(len(BINS))
    n_any = [int((per_model[c]["On time"] > 0).sum()) for c, _ in MODELS]
    for i, ((code, title), n_a) in enumerate(zip(MODELS, n_any)):
        oc = per_model[code]["On time"]
        h = [int(((oc >= lo) & (oc <= hi)).sum()) for lo, hi, _ in BINS]
        off = (i - 1.5) * width
        ax.bar(bxs + off, h, width=width, color=MODEL_COLORS[code],
               alpha=BAR_ALPHA, edgecolor=BAR_EDGE, linewidth=fs.LW_EDGE,
               zorder=3, label=f"{title} (n = {n_a})")
    # Per-bar value labels are dropped at this width: 24 bars across 3.4in
    # leaves ~3mm each, and the numbers would overlap their neighbours.
    ax.set_xticks(bxs)
    ax.set_xticklabels([b[2] for b in BINS], fontsize=fs.FS_TICK)
    ax.set_xlabel("Seeds detected On-time (out of 30)",
                  fontsize=fs.FS_LABEL, fontweight="bold", labelpad=fs.LABELPAD)
    ax.set_ylabel("Failed HDDs", fontsize=fs.FS_LABEL,
                  fontweight="bold", labelpad=fs.LABELPAD)
    ax.set_ylim(0, 31)
    # n in the legend, not in a panel title: with one panel the caption below
    # names the figure, so a title would just say it twice.
    ax.legend(loc="upper right", ncol=2, handlelength=1.2, columnspacing=1.0,
              borderpad=0.3, labelspacing=0.25, borderaxespad=0.3)
    fs.style_axes(ax)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"hdd_detectability_single_{HDD_NAME}.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"figure: {out}")


if __name__ == "__main__":
    main()