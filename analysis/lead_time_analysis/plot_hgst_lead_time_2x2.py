import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Publication Quality Styling Settings
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Calibri', 'sans-serif']
plt.rcParams['axes.edgecolor'] = '#111111'
plt.rcParams['axes.linewidth'] = 1.1

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis", "lead_time_analysis")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

HDD_NAME = "HGST_20HUH721212ALN604"

MODELS = [
    ("LGBM", "LightGBM"),
    ("XGB", "XGBoost"),
    ("LSTM", "LSTM"),
    ("GRU", "GRU")
]

STYLE_CONFIG = {
    "LGBM": {
        "fill": "#2b5c8f",       # Modern Deep Blue
        "edge": "#000000",       # Black Border
        "median_line": "#d9534f" # Crimson accent
    },
    "XGB": {
        "fill": "#d95f02",       # Tone-down Orange
        "edge": "#000000",
        "median_line": "#d9534f"
    },
    "LSTM": {
        "fill": "#7570b3",       # Soft Purple/Slate
        "edge": "#000000",
        "median_line": "#d9534f"
    },
    "GRU": {
        "fill": "#1b9e77",       # Deep Teal
        "edge": "#000000",
        "median_line": "#d9534f"
    }
}


def find_report_csv(model_code: str) -> str:
    for d in [REPORTS_DIR, RESULTS_DIR, ANALYSIS_DIR]:
        path = os.path.join(d, f"seed42_alarm_report_{HDD_NAME}_{model_code}.csv")
        if os.path.exists(path):
            return path
    return None


def load_lead_time_data(model_code: str) -> np.ndarray:
    path = find_report_csv(model_code)
    if not (path and os.path.exists(path)):
        raise FileNotFoundError(
            f"[STRICT ERROR] Missing required lead time report CSV for dataset='{HDD_NAME}', model='{model_code}'. "
            f"Experiments must not proceed without valid inference report data."
        )
    df = pd.read_csv(path)
    hits = df[(df['has_failed'] == 1) & (df['is_hit'] == 1)]
    return hits['days_to_failure_at_alarm'].dropna().values


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
        print("\n[AUTO-GENERATE] Required lead time CSVs missing. Triggering run_alarm_timing_seed42_analysis.py...")
        import subprocess
        res = subprocess.run([sys.executable, script_path], check=False)
        if res.returncode != 0:
            raise RuntimeError(
                f"[STRICT ERROR] Automatic report generation failed (exit code {res.returncode}). "
                f"Cannot generate visualization without complete model checkpoints and data."
            )


def main():
    ensure_alarm_reports()
    sns.set_theme(style="ticks", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300, sharey=True)
    
    fig.suptitle(
        "First-Alarm Lead Time Distribution — HGST (20HUH721212ALN604)",
        fontsize=16, fontweight="bold", y=0.98, color="#111111"
    )

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    labels = ["(a)", "(b)", "(c)", "(d)"]
    
    # 35 bins across 0 to 180 days
    bins = np.linspace(0, 180, 36)

    # Pre-calculate data to ensure unified Y-axis limit
    model_data = {}
    max_density = 0.0

    for m_code, _ in MODELS:
        lead_times_full = load_lead_time_data(m_code)
        if len(lead_times_full) > 0:
            lead_times_disp = lead_times_full[lead_times_full <= 180]
            true_median = float(np.median(lead_times_full))
            counts, _ = np.histogram(lead_times_disp, bins=bins, density=True)
            if len(counts) > 0 and not np.isnan(counts.max()) and counts.max() > max_density:
                max_density = counts.max()
        else:
            lead_times_disp = np.array([], dtype=float)
            true_median = 0.0

        model_data[m_code] = {
            "full": lead_times_full,
            "disp": lead_times_disp,
            "median": true_median
        }

    y_limit = max(0.045, np.ceil(max_density * 100) / 100 + 0.005)

    for idx, (m_code, m_title) in enumerate(MODELS):
        r, c = positions[idx]
        ax = axes[r, c]
        style = STYLE_CONFIG[m_code]
        data = model_data[m_code]

        lead_times_full = data["full"]
        lead_times_display = data["disp"]
        true_median_lt = data["median"]

        if len(lead_times_full) > 0:
            # Solid Fill Histogram with Black Borders
            ax.hist(
                lead_times_display,
                bins=bins,
                density=True,
                facecolor=style["fill"],
                edgecolor=style["edge"],
                alpha=0.72,
                linewidth=0.9,
                label="Density Hist"
            )

            # Vertical Line for Median
            ax.axvline(
                true_median_lt,
                color=style["median_line"],
                linestyle="--",
                linewidth=2.4,
                zorder=5,
                label=f"Median: {true_median_lt:.1f} days"
            )
        else:
            ax.text(
                0.5, 0.5, "Data Pending\n(Run analysis script)",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='#777777', style='italic'
            )

        # Subplot Formatting
        ax.set_title(
            f"{labels[idx]} {m_title}  (n = {len(lead_times_full)})",
            fontsize=13, fontweight="bold", pad=10, loc="left", color="#111111"
        )
        
        ax.set_xlabel("Lead Time (Days)", fontsize=11, fontweight="bold", labelpad=6)
        if c == 0:
            ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold", labelpad=6)
        else:
            ax.set_ylabel("")
        
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 181, 30))
        ax.set_ylim(0, y_limit)
        
        # Apply label_outer for clean y-axis deduplication
        ax.label_outer()
        # Ensure x-label is displayed on all subplots for explicit clarity
        ax.set_xlabel("Lead Time (Days)", fontsize=11, fontweight="bold", labelpad=6)
        
        # Subtle horizontal grid line (alpha=0.20)
        ax.grid(True, axis="y", linestyle=":", alpha=0.20, color="#666666")
        ax.grid(False, axis="x")
        sns.despine(ax=ax, top=True, right=True)
        
        # Legend with Median info
        handles, labels_leg = ax.get_legend_handles_labels()
        sel = [i for i, l in enumerate(labels_leg) if "Median" in l]
        
        ax.legend(
            [handles[i] for i in sel],
            [labels_leg[i] for i in sel],
            loc="upper right",
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#cccccc",
            framealpha=0.95,
            fontsize=10.5
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path_results = os.path.join(RESULTS_DIR, "HGST_20HUH721212ALN604_4models_2x2_lead_time_density.png")
    
    plt.savefig(out_path_results, dpi=300, bbox_inches="tight")
    plt.close()

    print("=" * 80)
    print(" [SUCCESS] Clean Y-axis Label Outer 2x2 Density Plot Generated!")
    print("  - Applied label_outer(): Removed redundant Y-axis labels & tick labels on (b) & (d)")
    print(f" Saved to: {out_path_results}")
    print("=" * 80)


if __name__ == "__main__":
    main()
