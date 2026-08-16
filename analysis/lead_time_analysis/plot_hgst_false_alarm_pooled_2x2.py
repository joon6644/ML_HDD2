"""Pooled (30-seed) version of the operational false-alarm timing figure.

The single-seed dot plot carries only 7-18 events per model because the
HDD-level FAR constraint caps the numerator at roughly 977 x 1% ~= 10 disks.
Pooling the 30 training seeds raises this to 320-511 events, which is enough to
report the temporal distribution itself rather than a visual trend.

One event = one (seed, HDD) pair judged Censored Early. The same censored HDD
may contribute in several seeds; this is a distribution over runs, not over
distinct disks, and the caption must say so.
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
MODELS = [
    ("LGBM", "LightGBM"),
    ("XGB", "XGBoost"),
    ("LSTM", "LSTM"),
    ("GRU", "GRU"),
]

MODEL_COLORS = {
    "LGBM": "#2b5c8f",
    "XGB": "#d95f02",
    "LSTM": "#7570b3",
    "GRU": "#1b9e77",
}

BIN_WIDTH = 100
MAX_DAYS = 2600


def collect_pooled_events(model_code: str) -> pd.DataFrame:
    """First false-alarm timing for every (seed, HDD) judged Censored Early."""
    rows = []
    for seed in SEEDS:
        df = load_alarm_report(HDD_NAME, model_code, seed=seed)
        # HDD-level FAR population: censored HDDs only. Early alarms on failed
        # HDDs are penalized through Recall/Precision, not FAR.
        fa = df[(df["has_failed"] == 0) & (df["alarm_triggered"] == 1)]
        days = fa["days_since_observed"].dropna()
        days = days[days >= 0]
        for serial, d in zip(fa.loc[days.index, "serial_number"], days):
            rows.append({"seed": seed, "serial_number": serial, "days_since_observed": float(d)})
        print(f"  [{model_code}] seed {seed}: {len(days)} events", flush=True)
    return pd.DataFrame(rows)


def main():
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Calibri", "sans-serif"]
    plt.rcParams["axes.edgecolor"] = "#111111"
    plt.rcParams["axes.linewidth"] = 1.1
    sns.set_theme(style="ticks", palette="muted")

    pooled = {}
    for m_code, _ in MODELS:
        print(f"[collect] {m_code}", flush=True)
        pooled[m_code] = collect_pooled_events(m_code)

    combined = pd.concat(
        [df.assign(model=m) for m, df in pooled.items()], ignore_index=True
    )
    csv_path = os.path.join(REPORTS_DIR, f"pooled30_false_alarm_timing_{HDD_NAME}.csv")
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=300, sharex=True, sharey=True)
    fig.suptitle(
        "Timing of First Operational False Alarm, Pooled over 30 Seeds "
        "— HGST (HUH721212ALN604)",
        fontsize=15, fontweight="bold", y=0.98, color="#111111",
    )

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    labels = ["(a)", "(b)", "(c)", "(d)"]
    bins = np.arange(0, MAX_DAYS + BIN_WIDTH, BIN_WIDTH)

    summary = []
    for idx, (m_code, m_title) in enumerate(MODELS):
        r, c = positions[idx]
        ax = axes[r, c]
        days = pooled[m_code]["days_since_observed"].values
        n_total = len(days)
        n_disks = pooled[m_code]["serial_number"].nunique()
        med = float(np.median(days)) if n_total else float("nan")

        weights = np.full(n_total, 100.0 / n_total) if n_total else None
        ax.hist(
            days, bins=bins, weights=weights,
            color=MODEL_COLORS[m_code], edgecolor="#ffffff", linewidth=0.6, zorder=3,
        )
        ax.axvline(med, color="#c0392b", linestyle="--", linewidth=1.6, zorder=4,
                   label=f"Median: {med:.0f} days")

        ax.set_title(
            f"{labels[idx]} {m_title}  (n = {n_total} events, {n_disks} distinct HDDs)",
            fontsize=12.5, fontweight="bold", pad=10, loc="left", color="#111111",
        )
        ax.legend(loc="upper left", fontsize=10, frameon=True)
        ax.set_xlim(0, MAX_DAYS)
        ax.set_xticks(np.arange(0, MAX_DAYS + 1, 500))
        ax.grid(True, axis="y", linestyle=":", alpha=0.30, color="#888888")
        sns.despine(ax=ax, top=True, right=True)

        if r == 1:
            ax.set_xlabel("Days Since Observation Start (Days)", fontsize=11,
                          fontweight="bold", labelpad=6)
        if c == 0:
            ax.set_ylabel("Frequency (%)", fontsize=11, fontweight="bold", labelpad=6)

        summary.append({
            "model": m_title, "n_events": n_total, "n_distinct_hdd": n_disks,
            "median_days": round(med, 1),
            "pct_before_1400d": round(100.0 * np.mean(days < 1400), 1) if n_total else np.nan,
            "pct_within_60d": round(100.0 * np.mean(days <= 60), 1) if n_total else np.nan,
        })

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(RESULTS_DIR, f"pooled30_false_alarm_timing_{HDD_NAME}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(REPORTS_DIR, f"pooled30_false_alarm_summary_{HDD_NAME}.csv"),
              index=False, encoding="utf-8-sig")

    print("=" * 78)
    print(sm.to_string(index=False))
    print(f"figure : {out_path}")
    print(f"events : {csv_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()