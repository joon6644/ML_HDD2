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
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

HDD_NAME = "HGST_20HUH721212ALN604"
MODELS = [
    ("LGBM", "LightGBM"),
    ("XGB", "XGBoost"),
    ("LSTM", "LSTM"),
    ("GRU", "GRU")
]

STYLE_CONFIG = {
    "LGBM": {"fill": "#2b5c8f", "edge": "#000000"},
    "XGB":  {"fill": "#d95f02", "edge": "#000000"},
    "LSTM": {"fill": "#7570b3", "edge": "#000000"},
    "GRU":  {"fill": "#1b9e77", "edge": "#000000"}
}


def find_report_csv(model_code: str) -> str:
    for d in [REPORTS_DIR, RESULTS_DIR, ANALYSIS_DIR]:
        path = os.path.join(d, f"seed42_alarm_report_{HDD_NAME}_{model_code}.csv")
        if os.path.exists(path):
            return path
    return None


def load_operational_false_alarm_data(model_code: str) -> np.ndarray:
    csv_path = find_report_csv(model_code)
    if not (csv_path and os.path.exists(csv_path)):
        raise FileNotFoundError(
            f"[STRICT ERROR] Missing required alarm report CSV for dataset='{HDD_NAME}', model='{model_code}'. "
            f"Experiments must not proceed without valid inference report data."
        )
    df = pd.read_csv(csv_path)
    # 1. Healthy HDD False Alarms
    healthy_fa = df[(df['has_failed'] == 0) & (df['alarm_triggered'] == 1)]
    # 2. Failed HDD False Alarms (outside 30-day operational warning window)
    failed_early_fa = df[(df['has_failed'] == 1) & (df['alarm_triggered'] == 1) & (df['days_to_failure_at_alarm'] > 30)]
    
    combined_fa = pd.concat([healthy_fa, failed_early_fa], ignore_index=True)
    days_since = combined_fa['days_since_observed'].dropna().values
    return days_since[days_since >= 0]


def ensure_alarm_reports():
    missing = False
    for m_code, _ in MODELS:
        if find_report_csv(m_code) is None:
            missing = True
            break
    if missing:
        script_path = os.path.join(ANALYSIS_DIR, "run_alarm_timing_seed42_analysis.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"[STRICT ERROR] Generator script missing: {script_path}")
        print("\n[AUTO-GENERATE] Required alarm report CSVs missing. Triggering run_alarm_timing_seed42_analysis.py...")
        import subprocess
        res = subprocess.run([sys.executable, script_path], check=False)
        if res.returncode != 0:
            raise RuntimeError(
                f"[STRICT ERROR] Automatic report generation failed (exit code {res.returncode}). "
                f"Cannot generate visualization without complete model checkpoints and data."
            )


def main():
    ensure_alarm_reports()
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Calibri', 'sans-serif']
    plt.rcParams['axes.edgecolor'] = '#111111'
    plt.rcParams['axes.linewidth'] = 1.1

    sns.set_theme(style="ticks", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300, sharey=True)
    
    fig.suptitle(
        "False Alarm Occurrence Timing Since Observation Start — HGST (20HUH721212ALN604)",
        fontsize=16, fontweight="bold", y=0.98, color="#111111"
    )

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    labels = ["(a)", "(b)", "(c)", "(d)"]

    model_data = {}
    for m_code, _ in MODELS:
        days_since = load_operational_false_alarm_data(m_code)
        model_data[m_code] = days_since

    # Full observation period (0 to 2600 days)
    max_days = 2600
    bins = np.linspace(0, max_days, 36) # 35 bins

    max_pct = 0.0
    for m_code, _ in MODELS:
        days = model_data[m_code]
        if len(days) > 0:
            counts, _ = np.histogram(days, bins=bins)
            pcts = (counts / len(days)) * 100.0
            if len(pcts) > 0 and pcts.max() > max_pct:
                max_pct = pcts.max()

    ylim_top = min(100.0, max(20.0, np.ceil(max_pct + 4.0)))

    for idx, (m_code, m_title) in enumerate(MODELS):
        r, c = positions[idx]
        ax = axes[r, c]
        style = STYLE_CONFIG[m_code]
        days = model_data[m_code]
        n_disks = len(days)

        if n_disks > 0:
            sns.histplot(
                x=days,
                bins=bins,
                stat="percent",
                kde=False,
                color=style["fill"],
                edgecolor=style["edge"],
                alpha=0.72,
                linewidth=0.9,
                ax=ax
            )

        ax.set_title(
            f"{labels[idx]} {m_title}  (n = {n_disks} HDDs)",
            fontsize=13, fontweight="bold", pad=10, loc="left", color="#111111"
        )

        ax.set_xlabel("Days Since Observation Start (Days)", fontsize=11, fontweight="bold", labelpad=6)
        if c == 0:
            ax.set_ylabel("False Alarm Frequency (%)", fontsize=11, fontweight="bold", labelpad=6)
        else:
            ax.set_ylabel("")

        ax.set_xlim(0, max_days)
        ax.set_ylim(0, ylim_top)
        ax.set_xticks(np.arange(0, max_days + 1, 500))

        ax.label_outer()
        ax.set_xlabel("Days Since Observation Start (Days)", fontsize=11, fontweight="bold", labelpad=6)

        ax.grid(True, axis="y", linestyle=":", alpha=0.20, color="#666666")
        ax.grid(False, axis="x")
        sns.despine(ax=ax, top=True, right=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out2 = os.path.join(RESULTS_DIR, "seed42_healthy_hdd_false_alarm_timing_grid.png")
    
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()

    print("=" * 80)
    print(" [SUCCESS] Updated Operational False Alarm Timing Grid Generated!")
    print("  - Includes Healthy HDD False Alarms + Failed HDD Early False Alarms (>30d)")
    print(f"  - LightGBM: n = {len(model_data['LGBM'])}")
    print(f"  - XGBoost : n = {len(model_data['XGB'])}")
    print(f"  - LSTM    : n = {len(model_data['LSTM'])}")
    print(f"  - GRU     : n = {len(model_data['GRU'])}")
    print(f" Saved to: {out2}")
    print("=" * 80)


if __name__ == "__main__":
    main()
