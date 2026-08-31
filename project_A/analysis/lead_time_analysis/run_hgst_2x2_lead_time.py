import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis", "lead_time_analysis")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

if ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, ANALYSIS_DIR)
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

from analysis_data_loader import get_proposed_threshold
from data_loader import load_dataset
from checkpoint_utils import load_checkpoint
from evaluator import RollingEvaluator
import config

HDD_NAME = "HGST_20HUH721212ALN604"
SEED = 42

MODELS = [
    ("LGBM", "LightGBM"),
    ("XGB", "XGBoost"),
    ("LSTM", "LSTM"),
    ("GRU", "GRU")
]

STYLE_CONFIG = {
    "LGBM": {"fill": "#2b5c8f", "edge": "#000000", "median": "#d9534f"},
    "XGB":  {"fill": "#d95f02", "edge": "#000000", "median": "#d9534f"},
    "LSTM": {"fill": "#7570b3", "edge": "#000000", "median": "#d9534f"},
    "GRU":  {"fill": "#1b9e77", "edge": "#000000", "median": "#d9534f"}
}


def load_all_alarm_lead_times(model_code: str) -> np.ndarray:
    """
    Computes lead times for ALL daily alarms triggered on failed HDDs.
    If an HDD alarms on multiple days before failure, every daily alarm contributes a lead time value.
    """
    m_lower = model_code.lower()
    thr = get_proposed_threshold(HDD_NAME, m_lower, seed=SEED)
    data_path = os.path.join(PROJECT_ROOT, "data", "splitted", HDD_NAME)
    is_seq = m_lower in ['lstm', 'gru']
    w_size = config.WINDOW_SIZE if is_seq else 1

    _, _, test_df, features = load_dataset(data_path, model=m_lower)
    model = load_checkpoint(
        m_lower, "none", SEED, config.TARGET_LEAD_TIME, data_path,
        input_dim=len(features), features=features,
        window_size=w_size if is_seq else None
    )
    m_type = 'pytorch_class' if is_seq else m_lower
    evaluator = RollingEvaluator(
        model, features, window_size=w_size,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model_type=m_type, seed=SEED
    )
    raw_preds = evaluator.get_raw_predictions(test_df, lead_time=config.TARGET_LEAD_TIME)

    all_lead_times = []
    for d in raw_preds:
        probs = np.array(d['preds'])
        dates = pd.to_datetime(d['dates'])
        alarm_mask = probs >= thr
        if d['has_failed'] and d['failure_date'] is not None:
            fail_date = pd.to_datetime(d['failure_date'])
            alarm_dates = dates[alarm_mask]
            lts = (fail_date - alarm_dates).days.values
            all_lead_times.extend(lts)

    return np.array(all_lead_times)


def main():
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Calibri', 'sans-serif']
    plt.rcParams['axes.edgecolor'] = '#111111'
    plt.rcParams['axes.linewidth'] = 1.1

    sns.set_theme(style="ticks", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300, sharey=True)
    fig.suptitle("All-Alarms Lead Time Distribution — HGST (20HUH721212ALN604)", fontsize=16, fontweight="bold", y=0.98, color="#111111")

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    tags = ["(a)", "(b)", "(c)", "(d)"]
    bins = np.linspace(0, 360, 37)

    model_data = {}
    max_pct = 0.0

    for m_code, _ in MODELS:
        lead_times_full = load_all_alarm_lead_times(m_code)
        true_median = float(np.median(lead_times_full)) if len(lead_times_full) > 0 else 0.0

        counts, _ = np.histogram(lead_times_full, bins=bins)
        if len(lead_times_full) > 0:
            pct = counts / len(lead_times_full) * 100.0
            over_pct = (lead_times_full > 360).sum() / len(lead_times_full) * 100.0
            max_pct = max(max_pct, pct.max(), over_pct)

        model_data[m_code] = {"full": lead_times_full, "median": true_median}

    # Headroom above the tallest bar so the upper-right legend never overlaps
    # the overflow bar at the axis edge.
    y_limit = np.ceil(max_pct) + 4.0

    for idx, (m_code, m_title) in enumerate(MODELS):
        r, c = positions[idx]
        ax = axes[r, c]
        style = STYLE_CONFIG[m_code]
        data = model_data[m_code]

        lead_times_full = data["full"]
        true_median_lt = data["median"]

        if len(lead_times_full) > 0:
            counts, edges = np.histogram(lead_times_full, bins=bins)
            pct = counts / len(lead_times_full) * 100.0
            over_pct = (lead_times_full > 360).sum() / len(lead_times_full) * 100.0
            ax.bar(
                edges[:-1], pct, width=np.diff(edges), align="edge",
                facecolor=style["fill"], edgecolor=style["edge"],
                alpha=0.72, linewidth=0.9
            )
            # Lead times beyond the axis are pooled into one hatched overflow
            # bar, so displayed bars always sum to 100% of the samples that the
            # median line is computed on.
            ax.bar(
                366, over_pct, width=18, align="edge",
                facecolor=style["fill"], edgecolor=style["edge"],
                alpha=0.45, linewidth=0.9, hatch="//"
            )

        ax.axvline(
            true_median_lt,
            color=style["median"],
            linestyle="--",
            linewidth=2.0,
            zorder=5,
            label=f"Median (per-alarm): {true_median_lt:.1f} days"
        )
        ax.axvline(30, color="#444444", linestyle=":", linewidth=1.6, zorder=4, label="H = 30 days")

        ax.set_title(f"{tags[idx]} {m_title}  (n = {len(lead_times_full)})", fontsize=13, fontweight="bold", pad=10, loc="left", color="#111111")
        if c == 0:
            ax.set_ylabel("Frequency (%)", fontsize=11, fontweight="bold", labelpad=6)
        else:
            ax.set_ylabel("")

        ax.set_xlim(0, 390)
        ax.set_xticks(list(np.arange(0, 301, 60)) + [375])
        ax.set_xticklabels([str(v) for v in np.arange(0, 301, 60)] + [">360"])
        ax.set_ylim(0, y_limit)

        ax.label_outer()
        ax.set_xlabel("Lead Time (Days)", fontsize=11, fontweight="bold", labelpad=6)

        ax.grid(True, axis="y", linestyle=":", alpha=0.20, color="#666666")
        ax.grid(False, axis="x")
        sns.despine(ax=ax, top=True, right=True)

        ax.legend(
            loc="upper right",
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#cccccc",
            framealpha=0.95,
            fontsize=10.5
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_img1 = os.path.join(RESULTS_DIR, "HGST_20HUH721212ALN604_4models_2x2_lead_time.png")
    plt.savefig(out_img1, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 80)
    print(" [SUCCESS] All-Alarms Lead Time Plot Generated!")
    print(f"  - LightGBM: n = {len(model_data['LGBM']['full'])}")
    print(f"  - XGBoost : n = {len(model_data['XGB']['full'])}")
    print(f"  - LSTM    : n = {len(model_data['LSTM']['full'])}")
    print(f"  - GRU     : n = {len(model_data['GRU']['full'])}")
    print(f" Saved to:\n  -> {out_img1}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
