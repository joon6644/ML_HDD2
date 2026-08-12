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



def extract_alarm_counts_by_window(hdd_name: str, model_name: str, threshold: float):
    hdd_path = os.path.join(PROJECT_ROOT, "data", "splitted", hdd_name)
    model_upper = model_name.upper()
    lookup_key = "LGBM" if model_name.lower() == "lgbm" else ("XGB" if model_name.lower() == "xgb" else model_name.upper())

    # 1. Fast Path: Reuse existing cached Report CSV if available (0.01s load time)
    reports_dir = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis", "reports")
    report_csv = os.path.join(reports_dir, f"seed42_alarm_report_{hdd_name}_{lookup_key}.csv")
    
    if os.path.exists(report_csv):
        print(f"[CACHE HIT] Instant load from report CSV -> {report_csv}")
        df = pd.read_csv(report_csv)
        hits = df[(df['has_failed'] == 1) & (df['alarm_triggered'] == 1)]
        arr = hits['days_to_failure_at_alarm'].dropna().values
        arr = arr[arr >= 0]
    else:
        print(f"[Processing] Running inference for Model: {model_upper} | Threshold: {threshold:.4f}")

        train_df, val_df, test_df, features = load_dataset(hdd_path, model=model_name.lower())

        is_sequence_model = (model_name.lower() in ['lstm', 'gru'])
        ckpt_window_size = config.WINDOW_SIZE if is_sequence_model else None

        model = load_checkpoint(
            model_name.lower(), "none", config.SEED, config.TARGET_LEAD_TIME, hdd_path,
            input_dim=len(features), features=features, window_size=ckpt_window_size
        )

        if model is None:
            raise FileNotFoundError(
                f"[STRICT ERROR] Checkpoint missing for model '{model_name}' on HDD '{hdd_name}'. "
                f"Experiments must not proceed without valid trained model weights."
            )

        model_type = 'pytorch_class' if is_sequence_model or model_name.lower() == 'mlp' else model_name.lower()

        evaluator = RollingEvaluator(
            model=model,
            features=features,
            window_size=config.WINDOW_SIZE if is_sequence_model else 1,
            device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
            model_type=model_type,
            seed=config.SEED
        )

        raw_preds = evaluator.get_raw_predictions(test_df, lead_time=config.TARGET_LEAD_TIME)

        days_to_failure_list = []
        for disk in raw_preds:
            if not disk['has_failed']:
                continue
            failure_date = disk['failure_date']
            dates = disk['dates']
            preds = disk['preds']

            alarm_indices = np.where(preds >= threshold)[0]
            for idx in alarm_indices:
                alarm_date = pd.to_datetime(dates[idx])
                dtf = (failure_date - alarm_date).days
                if dtf >= 0:
                    days_to_failure_list.append(dtf)

        arr = np.array(days_to_failure_list)

    tot = len(arr)

    c_0_10 = int(np.sum((arr >= 0) & (arr <= 10)))
    c_11_30 = int(np.sum((arr >= 11) & (arr <= 30)))
    c_31_60 = int(np.sum((arr >= 31) & (arr <= 60)))
    c_gt_60 = int(np.sum(arr > 60))

    return {
        'model': model_name,
        'model_label': 'LightGBM' if model_name.lower()=='lgbm' else ('XGBoost' if model_name.lower()=='xgb' else model_name.upper()),
        'total_alarms': tot,
        '0_10d': c_0_10,
        '11_30d': c_11_30,
        '31_60d': c_31_60,
        'gt_60d': c_gt_60,
        'pct_0_10d': (c_0_10 / tot * 100.0) if tot > 0 else 0.0,
        'pct_11_30d': (c_11_30 / tot * 100.0) if tot > 0 else 0.0,
        'pct_31_60d': (c_31_60 / tot * 100.0) if tot > 0 else 0.0,
        'pct_gt_60d': (c_gt_60 / tot * 100.0) if tot > 0 else 0.0,
    }


def main():
    hdd_name = "HGST_20HUH721212ALN604"
    threshold_map = load_threshold_map(seed=config.SEED)
    models = ["lgbm", "xgb", "lstm", "gru"]

    results_dir = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
    reports_dir = os.path.join(results_dir, "reports")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 80)
    print(f"  HGST_20HUH721212ALN604 - 100% STACKED BAR ALARM SUMMARY ANALYSIS  ")
    print("=" * 80)

    rows = []
    for m in models:
        lookup_key = "LGBM" if m.lower() == "lgbm" else ("XGB" if m.lower() == "xgb" else m.upper())
        thresh = threshold_map[(hdd_name, lookup_key)]
        res = extract_alarm_counts_by_window(hdd_name, m, thresh)
        rows.append(res)

    df_summary = pd.DataFrame(rows)
    csv_path = os.path.join(reports_dir, f"{hdd_name}_alarm_temporal_summary_table.csv")
    df_summary.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # Plot 100% Horizontal Stacked Bar Chart
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(12, 6))

    model_labels = df_summary['model_label'].values[::-1] # Reverse order so LightGBM is top
    pct_0_10 = df_summary['pct_0_10d'].values[::-1]
    pct_11_30 = df_summary['pct_11_30d'].values[::-1]
    pct_31_60 = df_summary['pct_31_60d'].values[::-1]
    pct_gt_60 = df_summary['pct_gt_60d'].values[::-1]

    # Color palette matching operational urgency
    colors = {
        '0_10d': '#d62728',   # Red (Imminent)
        '11_30d': '#ff7f0e',  # Orange (Short-term)
        '31_60d': '#17becf',  # Cyan/Teal (Mid-term)
        'gt_60d': '#1f77b4',  # Deep Blue (>60d Early)
    }

    y_pos = np.arange(len(model_labels))
    bar_height = 0.55

    # Cumulative left positions for stacking
    left_11_30 = pct_0_10
    left_31_60 = left_11_30 + pct_11_30
    left_gt_60 = left_31_60 + pct_31_60

    b1 = ax.barh(y_pos, pct_0_10, height=bar_height, color=colors['0_10d'], edgecolor='white', label='0 ~ 10 Days (Imminent)')
    b2 = ax.barh(y_pos, pct_11_30, left=left_11_30, height=bar_height, color=colors['11_30d'], edgecolor='white', label='11 ~ 30 Days (Short-term)')
    b3 = ax.barh(y_pos, pct_31_60, left=left_31_60, height=bar_height, color=colors['31_60d'], edgecolor='white', label='31 ~ 60 Days (Mid-term)')
    b4 = ax.barh(y_pos, pct_gt_60, left=left_gt_60, height=bar_height, color=colors['gt_60d'], edgecolor='white', label='> 60 Days (Early Warning)')

    # Add text annotations inside bar segments
    for idx in range(len(model_labels)):
        # Segment 1: 0-10d
        val1 = pct_0_10[idx]
        if val1 > 4.0:
            ax.text(val1 / 2.0, y_pos[idx], f"{val1:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10.5)

        # Segment 2: 11-30d
        val2 = pct_11_30[idx]
        if val2 > 4.0:
            ax.text(left_11_30[idx] + val2 / 2.0, y_pos[idx], f"{val2:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10.5)

        # Segment 3: 31-60d
        val3 = pct_31_60[idx]
        if val3 > 4.0:
            ax.text(left_31_60[idx] + val3 / 2.0, y_pos[idx], f"{val3:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10.5)

        # Segment 4: >60d
        val4 = pct_gt_60[idx]
        if val4 > 4.0:
            ax.text(left_gt_60[idx] + val4 / 2.0, y_pos[idx], f"{val4:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_labels, fontsize=12, fontweight='bold')
    ax.set_xlabel("Relative Alarm Frequency (%)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 10))
    ax.set_xticklabels([f"{x}%" for x in range(0, 101, 10)], fontsize=10.5)
    ax.set_title("100% Stacked Bar Summary of Alarm Concentration Before Failure - HGST (20HUH721212ALN604)", fontsize=14, fontweight="bold", pad=15)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=11, frameon=True)
    sns.despine(top=True, right=True)

    plt.tight_layout()

    out_img = os.path.join(results_dir, "HGST_20HUH721212ALN604_100pct_stacked_bar_alarm_summary.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 80)
    print(f" [SUCCESS] 100% Stacked Bar Summary Image saved to:\n  {out_img}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
