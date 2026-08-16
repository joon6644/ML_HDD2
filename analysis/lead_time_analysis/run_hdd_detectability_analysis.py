"""Per-HDD detectability across 30 training seeds (HGST, tau_op).

Row-level evaluation cannot ask this question at all: it loses disk identity, so
it cannot separate "the same drives are missed every time" from "a different
drive is missed each run". Here every failed test HDD is followed across all
30 seeds and 4 models, and classified by how often it was detected On-time.

Outputs
  - per-HDD detection counts (csv)
  - per-model summary: all-Missed / never-On-time / any-On-time / all-On-time
  - cross-model summary over all 120 runs
  - 2x2 figure: distribution of On-time detection count per HDD
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

BINS = [(0, 0, "0"), (1, 5, "1–5"), (6, 10, "6–10"), (11, 15, "11–15"),
        (16, 20, "16–20"), (21, 25, "21–25"), (26, 30, "26–30")]


def collect(model_code: str) -> pd.DataFrame:
    """One row per (seed, failed HDD) with its judgement category."""
    frames = []
    for seed in SEEDS:
        rep = load_alarm_report(HDD_NAME, model_code, seed=seed)
        f = rep[rep["has_failed"] == 1][["serial_number", "category"]].copy()
        f["seed"] = seed
        frames.append(f)
        print(f"  [{model_code}] seed {seed}: {len(f)} failed HDDs", flush=True)
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
        long_df = collect(code)
        piv = counts_per_hdd(long_df)
        per_model[code] = piv

        n = len(piv)
        all_missed = piv["Missed"] == len(SEEDS)
        any_on = piv["On time"] > 0
        never_on = ~any_on
        all_on = piv["On time"] == len(SEEDS)
        detected = piv.loc[any_on, "On time"]
        summary.append({
            "model": title, "n_failed": n,
            "all_Missed": int(all_missed.sum()),
            "alarmed_but_never_On_time": int((never_on & ~all_missed).sum()),
            "never_On_time": int(never_on.sum()),
            "any_On_time": int(any_on.sum()),
            "all_On_time": int(all_on.sum()),
            "median_On_time_count_among_detected": float(detected.median()) if len(detected) else np.nan,
        })

    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(REPORTS_DIR, f"detectability_per_model_{HDD_NAME}.csv"),
              index=False, encoding="utf-8-sig")

    # ---- cross-model: 120 runs -------------------------------------------
    on_all = sum(per_model[c]["On time"] for c, _ in MODELS)
    miss_all = sum(per_model[c]["Missed"] for c, _ in MODELS)
    n_runs = len(SEEDS) * len(MODELS)
    cross = pd.DataFrame({
        "on_time_runs": on_all, "missed_runs": miss_all,
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
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), dpi=300, sharey=True)
    fig.suptitle("Per-HDD On-time Detection Count across 30 Training Seeds "
                 "— HGST (HUH721212ALN604)",
                 fontsize=14.5, fontweight="bold", y=0.98)
    labels = ["(a)", "(b)", "(c)", "(d)"]
    for idx, (code, title) in enumerate(MODELS):
        ax = axes[idx // 2, idx % 2]
        piv = per_model[code]
        oc = piv["On time"]
        heights = [int(((oc >= lo) & (oc <= hi)).sum()) for lo, hi, _ in BINS]
        xs = np.arange(len(BINS))

        # The 0-column mixes two very different failure modes, so split it:
        # drives that never raised any alarm vs drives whose alarms were all Early.
        n_all_missed = int((piv["Missed"] == len(SEEDS)).sum())
        n_alarmed_never_on = heights[0] - n_all_missed
        base = [n_all_missed] + [0] * (len(BINS) - 1)
        top = [n_alarmed_never_on] + heights[1:]
        ax.bar(xs, base, color="#b0b0b0", edgecolor="#ffffff", linewidth=0.8, zorder=3,
               label="No alarm in any seed (Missed)")
        ax.bar(xs, top, bottom=base, color=MODEL_COLORS[code], edgecolor="#ffffff",
               linewidth=0.8, zorder=3, label="No On-time; Early when alarmed")
        ax.text(0, n_all_missed / 2, str(n_all_missed), ha="center", va="center",
                fontsize=9, color="#ffffff", fontweight="bold")
        ax.text(0, n_all_missed + n_alarmed_never_on / 2, str(n_alarmed_never_on),
                ha="center", va="center", fontsize=9, color="#ffffff", fontweight="bold")
        for x, h in zip(xs[1:], heights[1:]):
            if h:
                ax.text(x, h + 1.5, str(h), ha="center", fontsize=9, color="#333333")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9, frameon=True)
        never = int((oc == 0).sum())
        ax.set_title(f"{labels[idx]} {title}  (never On-time: {never}/{len(oc)} "
                     f"= {100*never/len(oc):.0f}%)",
                     fontsize=12, fontweight="bold", loc="left", pad=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([b[2] for b in BINS], fontsize=10)
        ax.grid(True, axis="y", linestyle=":", alpha=0.3)
        sns.despine(ax=ax, top=True, right=True)
        if idx // 2 == 1:
            ax.set_xlabel("Number of seeds detected On-time (out of 30)",
                          fontsize=11, fontweight="bold", labelpad=6)
        if idx % 2 == 0:
            ax.set_ylabel("Number of failed HDDs", fontsize=11, fontweight="bold", labelpad=6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(RESULTS_DIR, f"hdd_detectability_2x2_{HDD_NAME}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"figure: {out}")


if __name__ == "__main__":
    main()