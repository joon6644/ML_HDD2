import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import torch
except ImportError:
    torch = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, ANALYSIS_DIR)

import config
from data_loader import load_dataset
from checkpoint_utils import load_checkpoint
from evaluator import RollingEvaluator
from analysis_data_loader import load_threshold_map

SEED = 42
HDD_NAME = "HGST_20HUH721212ALN604"
MODELS = ['lgbm', 'xgb', 'lstm', 'gru']

MODEL_TITLES = {
    'lgbm': 'LightGBM',
    'xgb': 'XGBoost',
    'lstm': 'LSTM',
    'gru': 'GRU'
}

STYLE_CONFIG = {
    "lgbm": {"fill": "#2b5c8f", "edge": "#000000"},
    "xgb":  {"fill": "#d95f02", "edge": "#000000"},
    "lstm": {"fill": "#7570b3", "edge": "#000000"},
    "gru":  {"fill": "#1b9e77", "edge": "#000000"}
}


def evaluate_one(dataset: str, model_name: str, threshold: float = None):
    data_path = os.path.join(PROJECT_ROOT, "data", "splitted", dataset)
    is_sequence_model = model_name in ['lstm', 'gru']
    window_size = config.WINDOW_SIZE if is_sequence_model else 1

    _, val_df, test_df, features = load_dataset(data_path, model=model_name)

    model = load_checkpoint(
        model_name, "none", SEED, config.TARGET_LEAD_TIME, data_path,
        input_dim=len(features), features=features,
        window_size=window_size if is_sequence_model else None
    )
    if model is None:
        raise FileNotFoundError(
            f"[STRICT ERROR] Checkpoint missing for model='{model_name}' on dataset='{dataset}' (seed={SEED}). "
            f"Experiments must not proceed without valid trained model weights."
        )

    model_type = 'pytorch_class' if is_sequence_model else model_name
    evaluator = RollingEvaluator(
        model=model, features=features, window_size=window_size,
        device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
        model_type=model_type, seed=SEED
    )

    if threshold is None or threshold <= 0:
        print(f"[Optimal Threshold Search] Searching best threshold for model='{model_name}' dataset='{dataset}' via RollingEvaluator.find_best_threshold...")
        val_raw_preds = evaluator.get_raw_predictions(val_df, lead_time=config.TARGET_LEAD_TIME)
        threshold, max_val_recall = evaluator.find_best_threshold(val_raw_preds, max_far=config.MAX_FAR, lead_time=config.TARGET_LEAD_TIME)
        print(f"[Optimal Threshold Found] {model_name} optimal threshold = {threshold:.4f} (Max Val Recall @ FAR <= {config.MAX_FAR:.2%}: {max_val_recall:.4%})")

    raw_preds = evaluator.get_raw_predictions(test_df, lead_time=config.TARGET_LEAD_TIME)
    _, report_df = evaluator.evaluate_proposed_level(raw_preds, threshold=threshold)

    first_seen = {d['serial_number']: pd.to_datetime(d['dates']).min() for d in raw_preds}
    report_df['first_seen_date'] = report_df['serial_number'].map(first_seen)
    report_df['days_since_observed'] = (
        pd.to_datetime(report_df['first_alarm_date']) - report_df['first_seen_date']
    ).dt.days

    return report_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Operational False Alarm Timing Analysis")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Overwrite cached alarm reports with new threshold evaluation")
    args = parser.parse_args()

    analysis_dir = os.path.join(PROJECT_ROOT, "analysis", "lead_time_analysis")
    results_dir = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
    reports_dir = os.path.join(results_dir, "reports")
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    threshold_map = load_threshold_map(SEED)

    print("=" * 80)
    print(f" OPERATIONAL FALSE ALARM TIMING ANALYSIS - HGST 2x2 (SEED={SEED}) ")
    print("=" * 80)

    model_data = {}
    for model_name in MODELS:
        thresh = threshold_map.get((HDD_NAME, model_name.upper()))
        
        fname = f"seed42_alarm_report_{HDD_NAME}_{model_name.upper()}.csv"
        csv_path = os.path.join(reports_dir, fname)

        if not args.overwrite and os.path.exists(csv_path):
            report_df = pd.read_csv(csv_path)
        else:
            print(f"[Evaluating with New Threshold] {HDD_NAME} | {model_name.upper()} | threshold={thresh}")
            report_df = evaluate_one(HDD_NAME, model_name, thresh)
            if report_df is None:
                continue
            report_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
        # User defined operational false alarms:
        # 1) Healthy HDDs with alarm (has_failed == 0 & alarm_triggered == 1)
        # 2) Failed HDDs with early alarm > 30d before failure (has_failed == 1 & alarm_triggered == 1 & days_to_failure_at_alarm > 30)
        h_fa = report_df[(report_df['has_failed'] == 0) & (report_df['alarm_triggered'] == 1)]
        f_fa = report_df[(report_df['has_failed'] == 1) & (report_df['alarm_triggered'] == 1) & (report_df['days_to_failure_at_alarm'] > 30)]
        
        combined_fa = pd.concat([h_fa, f_fa], ignore_index=True)
        days_since = combined_fa['days_since_observed'].dropna().values
        model_data[model_name] = days_since[days_since >= 0]

    # Publication Quality Styling
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

    max_days = 2600
    bins = np.linspace(0, max_days, 36)

    max_pct = 0.0
    for m in MODELS:
        days = model_data.get(m, np.array([], dtype=float))
        if len(days) > 0:
            counts, _ = np.histogram(days, bins=bins)
            pcts = (counts / len(days)) * 100.0
            if len(pcts) > 0 and pcts.max() > max_pct:
                max_pct = pcts.max()

    ylim_top = min(100.0, max(20.0, np.ceil(max_pct + 4.0)))

    for idx, m in enumerate(MODELS):
        r, c = positions[idx]
        ax = axes[r, c]
        style = STYLE_CONFIG[m]
        m_title = MODEL_TITLES[m]
        days = model_data.get(m, np.array([], dtype=float))
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

    out2 = os.path.join(results_dir, "seed42_healthy_hdd_false_alarm_timing_grid.png")

    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 80)
    print(" [SUCCESS] Operational False Alarm Timing HGST 2x2 Grid Saved!")
    print(f" Saved to:\n  -> {out2}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
