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

from analysis_data_loader import load_alarm_report

HDD_NAME = "HGST_20HUH721212ALN604"
SEEDS = list(range(42, 72))
MODELS = [("LGBM", "LightGBM"), ("XGB", "XGBoost"), ("LSTM", "LSTM"), ("GRU", "GRU")]
MODEL_COLORS = {"LGBM": "#2b5c8f", "XGB": "#d95f02", "LSTM": "#7570b3", "GRU": "#1b9e77"}
GRAY = "#b0b0b0"
# Matches plot_hgst_lead_time_1x2.py so the two figures read as one palette.
# At this alpha the fills are too light for white in-bar labels, hence LABEL_INK.
BAR_ALPHA = 0.72
BAR_EDGE = "#000000"
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


def counts_per_hdd(long_df: pd.DataFrame) -> pd.DataFrame:
    piv = long_df.pivot_table(index="serial_number", columns="category",
                              aggfunc="size", fill_value=0)
    for c in ("On time", "Early", "Missed"):
        if c not in piv:
            piv[c] = 0
    return piv[["On time", "Early", "Missed"]]


def main():
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
    sns.set_theme(style="ticks", palette="muted")

    per_model, summary = {}, []
    for code, title in MODELS:
        print(f"[collect] {code}", flush=True)
        piv = counts_per_hdd(collect(code))
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
    print("=" * 78)

    # ---- figure -----------------------------------------------------------
    # Single-column (3.4in) stacked layout, matching plot_hgst_lead_time_2x1.py:
    # in a two-column journal a single-column figure lets the facing column keep
    # flowing, so it costs about half the page of the same figure run full width.
    # No suptitle -- the Korean caption below the figure already names it.
    fig, (axL, axR) = plt.subplots(
        2, 1, figsize=(3.4, 4.4), dpi=400,
        gridspec_kw={"height_ratios": [0.85, 1.0], "hspace": 0.46},
    )

    n_failed = len(per_model["LGBM"])
    titles = [t for _, t in MODELS]
    xs = np.arange(len(MODELS))

    # (a) never On-time, split by whether the disk ever raised an alarm
    base = [int((per_model[c]["Missed"] == len(SEEDS)).sum()) for c, _ in MODELS]
    never = [int((per_model[c]["On time"] == 0).sum()) for c, _ in MODELS]
    top = [n - b for n, b in zip(never, base)]
    # Thinner than the lead-time histogram's 0.9: these four bars are ~10x wider,
    # so the same stroke reads much heavier around them.
    axL.bar(xs, base, color=GRAY, alpha=BAR_ALPHA, edgecolor=BAR_EDGE,
            linewidth=0.5, zorder=3, label="No alarm in any seed")
    axL.bar(xs, top, bottom=base, alpha=BAR_ALPHA, edgecolor=BAR_EDGE,
            linewidth=0.5, zorder=3,
            color=[MODEL_COLORS[c] for c, _ in MODELS],
            label="Alarmed, but always Early")
    for x, b, t, n in zip(xs, base, top, never):
        axL.text(x, b / 2, str(b), ha="center", va="center", fontsize=6,
                 color=LABEL_INK, fontweight="bold", zorder=4)
        axL.text(x, b + t / 2, str(t), ha="center", va="center", fontsize=6,
                 color=LABEL_INK, fontweight="bold", zorder=4)
        axL.text(x, n + 3, f"{n} ({100*n/n_failed:.0f}%)", ha="center",
                 fontsize=5.8, color="#333333", fontweight="bold")
    axL.set_title(f"(a) Never detected On-time (of {n_failed} failed HDDs)",
                  fontsize=7, fontweight="bold", loc="left", pad=5)
    axL.set_xticks(xs)
    axL.set_xticklabels(titles, fontsize=6.2)
    axL.set_ylabel("Failed HDDs", fontsize=6.5, fontweight="bold", labelpad=3)
    axL.set_ylim(0, 128)
    # Below the axes: the bars fill the lower area and the count labels fill the
    # upper one, so there is no room for a legend inside this panel. Labels are
    # clipped short here -- the Korean caption note spells both groups out.
    axL.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
               fontsize=5.8, frameon=False, handlelength=1.2, columnspacing=1.0)

    # (b) how often the rest were detected
    width = 0.2
    bxs = np.arange(len(BINS))
    for i, (code, title) in enumerate(MODELS):
        oc = per_model[code]["On time"]
        h = [int(((oc >= lo) & (oc <= hi)).sum()) for lo, hi, _ in BINS]
        off = (i - 1.5) * width
        axR.bar(bxs + off, h, width=width, color=MODEL_COLORS[code],
                alpha=BAR_ALPHA, edgecolor=BAR_EDGE, linewidth=0.4,
                zorder=3, label=title)
    # Per-bar value labels are dropped at this width: 24 bars across 3.4in
    # leaves ~3mm each, and the numbers would overlap their neighbours.
    n_any = [int((per_model[c]["On time"] > 0).sum()) for c, _ in MODELS]
    axR.set_title(f"(b) Detection count among the remaining "
                  f"{min(n_any)}–{max(n_any)} HDDs",
                  fontsize=7, fontweight="bold", loc="left", pad=5)
    axR.set_xticks(bxs)
    axR.set_xticklabels([b[2] for b in BINS], fontsize=6.2)
    axR.set_xlabel("Seeds detected On-time (out of 30)",
                   fontsize=6.5, fontweight="bold", labelpad=3)
    axR.set_ylabel("Failed HDDs", fontsize=6.5, fontweight="bold", labelpad=3)
    axR.set_ylim(0, 31)
    axR.legend(loc="upper right", fontsize=5.8, frameon=True, ncol=2,
               handlelength=1.2, columnspacing=1.0, borderpad=0.35,
               labelspacing=0.3)

    for ax in (axL, axR):
        ax.grid(True, axis="y", linestyle=":", alpha=0.3)
        ax.tick_params(axis="y", labelsize=6)
        sns.despine(ax=ax, top=True, right=True)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"hdd_detectability_2x1_{HDD_NAME}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"figure: {out}")


if __name__ == "__main__":
    main()