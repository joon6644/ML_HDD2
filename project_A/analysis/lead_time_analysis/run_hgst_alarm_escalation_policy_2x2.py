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
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, ANALYSIS_DIR)

import config
from data_loader import load_dataset
from evaluator import RollingEvaluator
from analysis_data_loader import load_threshold_map, load_analysis_model

MODEL_TITLES = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "lstm": "LSTM",
    "gru": "GRU"
}


def collect_alarm_escalation_policy_data(hdd_name: str, model_name: str, threshold: float, max_m: int = 20):
    hdd_path = os.path.join(PROJECT_ROOT, "data", "splitted", hdd_name)
    train_df, val_df, test_df, features = load_dataset(hdd_path, model=model_name)
    is_seq = model_name in ['lstm', 'gru']

    model = load_analysis_model(
        dataset=hdd_name,
        model_name=model_name,
        seed=config.SEED,
        features=features
    )

    evaluator = RollingEvaluator(
        model=model,
        features=features,
        window_size=config.WINDOW_SIZE if is_seq else 1,
        device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
        model_type='pytorch_class' if is_seq else model_name,
        seed=config.SEED
    )

    raw_preds = evaluator.get_raw_predictions(test_df, lead_time=config.TARGET_LEAD_TIME)

    total_failed = sum(1 for d in raw_preds if d['has_failed'])
    total_healthy = sum(1 for d in raw_preds if not d['has_failed'])

    records = []
    for M in range(1, max_m + 1):
        tp_cnt = 0
        fp_cnt = 0
        for disk in raw_preds:
            n_alarms = np.sum(disk['preds'] >= threshold)
            if n_alarms >= M:
                if disk['has_failed']:
                    tp_cnt += 1
                else:
                    fp_cnt += 1

        recall_pct = (tp_cnt / total_failed * 100) if total_failed > 0 else 0
        precision_pct = (tp_cnt / (tp_cnt + fp_cnt) * 100) if (tp_cnt + fp_cnt) > 0 else 0

        records.append({
            'M': M,
            'TP': tp_cnt,
            'FP': fp_cnt,
            'Recall_pct': recall_pct,
            'Precision_pct': precision_pct,
            'Total_Failed': total_failed,
            'Total_Healthy': total_healthy
        })

    return pd.DataFrame(records)


def plot_alarm_escalation_policy_2x2(model_data_map: dict, hdd_name: str, output_paths: list, max_m: int = 20):
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Calibri', 'sans-serif']
    plt.rcParams['axes.edgecolor'] = '#2D3748'
    plt.rcParams['axes.linewidth'] = 1.2

    sns.set_theme(style="ticks", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2), dpi=300)

    fig.suptitle(
        f"Operational Alarm Escalation Policy Analysis — Persistence vs. Detection & Precision — {hdd_name}",
        fontsize=15, fontweight="bold", y=0.98, color="#1A202C"
    )

    models_order = ["lgbm", "xgb", "lstm", "gru"]
    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    subplot_labels = [
        "(a) LightGBM",
        "(b) XGBoost",
        "(c) LSTM",
        "(d) GRU"
    ]

    color_recall = "#1E40AF"     # Deep Royal Blue for Recall / TP
    color_precision = "#059669"  # Emerald Green for Precision
    color_fp = "#DC2626"         # Crimson Red for False Alarm Count (FP)

    for idx, m_key in enumerate(models_order):
        r, c = subplot_positions[idx]
        ax1 = axes[r, c]
        df_p = model_data_map[m_key]

        m_indices = df_p['M'].values
        rec_vals = df_p['Recall_pct'].values
        prec_vals = df_p['Precision_pct'].values
        tp_vals = df_p['TP'].values
        fp_vals = df_p['FP'].values
        tot_failed = df_p['Total_Failed'].iloc[0]

        # Primary Y-axis: Recall (%) & TP Count
        ax1.plot(
            m_indices, rec_vals, color=color_recall, linewidth=2.5,
            marker='o', markersize=5, label='Disk Recall (%) [TP]'
        )
        ax1.set_ylabel("Disk Recall (%)", fontsize=11, fontweight="bold", color=color_recall)
        ax1.set_ylim(0, 45)
        ax1.tick_params(axis='y', labelcolor=color_recall)

        # Secondary Y-axis: Precision (%)
        ax2 = ax1.twinx()
        ax2.plot(
            m_indices, prec_vals, color=color_precision, linewidth=2.2,
            linestyle='--', marker='s', markersize=4.5, label='Disk Precision (%)'
        )
        ax2.set_ylabel("Disk Precision (%)", fontsize=11, fontweight="bold", color=color_precision)
        ax2.set_ylim(50, 100)
        ax2.tick_params(axis='y', labelcolor=color_precision)

        # Key Policy Annotations (M=1 vs M=5 vs M=10)
        m1_tp, m1_fp, m1_rec, m1_prec = tp_vals[0], fp_vals[0], rec_vals[0], prec_vals[0]
        m5_tp, m5_fp, m5_rec, m5_prec = tp_vals[4], fp_vals[4], rec_vals[4], prec_vals[4]
        m10_tp, m10_fp, m10_rec, m10_prec = tp_vals[9], fp_vals[9], rec_vals[9], prec_vals[9]

        annot_text = (
            f"• M=1 Alarm (Immediate Replacement):\n"
            f"   Recall: {m1_rec:.1f}% ({m1_tp}/{tot_failed}) | Prec: {m1_prec:.1f}% (FP={m1_fp})\n"
            f"• M=5 Alarms (Moderate Persistence):\n"
            f"   Recall: {m5_rec:.1f}% ({m5_tp}/{tot_failed}) | Prec: {m5_prec:.1f}% (FP={m5_fp})\n"
            f"• M=10 Alarms (High Persistence):\n"
            f"   Recall: {m10_rec:.1f}% ({m10_tp}/{tot_failed}) | Prec: {m10_prec:.1f}% (FP={m10_fp})"
        )

        ax1.text(
            0.04, 0.42, annot_text,
            transform=ax1.transAxes, fontsize=8.5, color='#1E293B',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8FAFC', edgecolor='#CBD5E1', alpha=0.95)
        )

        ax1.set_title(
            subplot_labels[idx],
            fontsize=12.5, fontweight='bold', pad=9, loc='left', color="#1A202C"
        )

        ax1.set_xlim(0.5, max_m + 0.5)
        ax1.set_xticks(np.arange(1, max_m + 1, 2 if max_m <= 20 else 5))

        if r == 1:
            ax1.set_xlabel("Alarm Escalation Policy M (Require ≥ M Alarms Before Replacement)", fontsize=11, fontweight="bold", labelpad=6)
        else:
            ax1.set_xlabel("")

        ax1.grid(True, axis="y", linestyle="--", alpha=0.3, color="#CBD5E1")
        ax1.grid(False, axis="x")

        # Legends
        if r == 0 and c == 0:
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(
                lines1 + lines2, labels1 + labels2,
                fontsize=9, loc='upper right', frameon=True,
                facecolor='#FFFFFF', edgecolor='#CBD5E1', framealpha=0.95
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    for path in output_paths:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Saved 2x2 Alarm Escalation Policy plot -> {output_paths[0]}")


def main():
    hdd_name = "HGST_20HUH721212ALN604"
    models = ["lgbm", "xgb", "lstm", "gru"]
    threshold_map = load_threshold_map(seed=config.SEED)

    results_dir = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(" GENERATING 2X2 ALARM ESCALATION POLICY PLOT (HGST) ")
    print("=" * 80)

    model_data_map = {}
    for m in models:
        lookup_key = "LGBM" if m.lower() == "lgbm" else ("XGB" if m.lower() == "xgb" else m.upper())
        thresh = threshold_map[(hdd_name, lookup_key)]
        print(f"\n[Processing] Model: {MODEL_TITLES[m]} | Threshold: {thresh:.4f}")

        df_p = collect_alarm_escalation_policy_data(hdd_name, m, thresh, max_m=20)
        model_data_map[m] = df_p

    out1 = os.path.join(results_dir, "HGST_20HUH721212ALN604_alarm_escalation_policy_2x2.png")

    plot_alarm_escalation_policy_2x2(model_data_map, hdd_name, [out1], max_m=20)


if __name__ == "__main__":
    main()
