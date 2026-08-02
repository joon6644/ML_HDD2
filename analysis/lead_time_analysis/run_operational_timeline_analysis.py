import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

try:
    import torch
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except ImportError:
    torch = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

import config
from data_loader import load_dataset
from checkpoint_utils import load_checkpoint
from evaluator import RollingEvaluator

config.PIPELINE_VERSION = "v2"

DEFAULT_THRESHOLDS = {
    ("HGST_20HUH721212ALN604", "LGBM"): 0.99,
    ("HGST_20HUH721212ALN604", "XGB"): 0.46,
    ("HGST_20HUH721212ALN604", "LSTM"): 0.11,
    ("HGST_20HUH721212ALN604", "GRU"): 0.16,
}


def load_threshold_map() -> dict:
    threshold_map = DEFAULT_THRESHOLDS.copy()
    master_csv = os.path.join(PROJECT_ROOT, "results", "master_experiment_results.csv")
    if os.path.exists(master_csv):
        try:
            df = pd.read_csv(master_csv, encoding='utf-8-sig')
            for _, row in df.iterrows():
                hdd = str(row['데이터']).strip()
                model_name = str(row['Model']).upper()
                thresh = float(row['Threshold'])
                threshold_map[(hdd, model_name)] = thresh
            print(f"[Threshold Loader] Loaded thresholds from master CSV -> {master_csv}")
        except Exception as e:
            print(f"[Threshold Loader] Warning: Could not read master CSV ({e}). Using defaults.")
    return threshold_map


def select_best_quadrant_samples(raw_preds, threshold: float):
    tp_candidates = [d for d in raw_preds if d['has_failed'] and np.any(d['preds'] >= threshold)]
    fn_candidates = [d for d in raw_preds if d['has_failed'] and not np.any(d['preds'] >= threshold)]
    fp_candidates = [d for d in raw_preds if not d['has_failed'] and np.any(d['preds'] >= threshold)]
    tn_candidates = [d for d in raw_preds if not d['has_failed'] and not np.any(d['preds'] >= threshold)]

    tp_sample = sorted(tp_candidates, key=lambda x: np.max(x['preds']), reverse=True)[0] if tp_candidates else None
    fn_sample = sorted(fn_candidates, key=lambda x: np.max(x['preds']), reverse=True)[0] if fn_candidates else None
    fp_sample = sorted(fp_candidates, key=lambda x: np.max(x['preds']), reverse=True)[0] if fp_candidates else None
    tn_sample = tn_candidates[0] if tn_candidates else None

    return {
        'TP': tp_sample,
        'FN': fn_sample,
        'FP': fp_sample,
        'TN': tn_sample
    }


def plot_operational_timelines(samples: dict, model_title: str, threshold: float, hdd_name: str, output_path: str):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f"Operational Life-Cycle Timelines across 4 Confusion Matrix Quadrants - {model_title} ({hdd_name})", fontsize=15, fontweight="bold", y=0.98)

    quadrants = [
        ('TP', (0, 0), "True Positive (TP): Early Alarm on Failed Disk", "Failed Disk with Successful Early Warning"),
        ('FN', (0, 1), "False Negative (FN): Missed Detection on Failed Disk", "Failed Disk without Reaching Threshold"),
        ('FP', (1, 0), "False Positive (FP): False Alarm on Healthy Disk", "Healthy Disk with Transient False Alert"),
        ('TN', (1, 1), "True Negative (TN): Normal Operation on Healthy Disk", "Healthy Disk Remaining Quiet")
    ]

    for idx, (key, (r, c), title, subtitle) in enumerate(quadrants):
        ax = axes[r, c]
        disk_data = samples.get(key)

        if disk_data is None:
            ax.text(0.5, 0.5, f"No sample found for {key}", ha='center', va='center')
            continue

        serial = disk_data['serial_number']
        dates = pd.to_datetime(disk_data['dates'])
        preds = disk_data['preds']
        has_failed = disk_data['has_failed']
        failure_date = disk_data['failure_date'] if has_failed else None

        # Truncate view to last 365 days for clean visualization
        max_days_plot = 365
        if len(dates) > max_days_plot:
            dates = dates[-max_days_plot:]
            preds = preds[-max_days_plot:]

        # Plot continuous Prediction Probability curve
        ax.plot(dates, preds, color='#1f77b4', linewidth=1.8, label='Prediction Probability P(t)')

        # Highlight Alarm Triggers (P >= threshold) with small red dots
        alarm_mask = (preds >= threshold)
        if np.any(alarm_mask):
            alarm_dates = dates[alarm_mask]
            alarm_preds = preds[alarm_mask]
            ax.scatter(alarm_dates, alarm_preds, color='red', s=10, zorder=5, label=f'Alarm Triggered (P ≥ {threshold:.2f})')
            ax.fill_between(dates, threshold, preds, where=(preds >= threshold), color='red', alpha=0.20)

        # Decision Threshold Line
        ax.axhline(threshold, color='darkred', linestyle='--', linewidth=1.5, label=f'Decision Threshold ({threshold:.2f})')

        # Clean vertical dashed line for Failure Event (No black X marker)
        if has_failed and failure_date is not None:
            ax.axvline(failure_date, color='black', linestyle='--', linewidth=1.8, label='Actual Failure Event')

        # TN specific clean annotation
        if key == 'TN':
            max_p = np.max(preds) if len(preds) > 0 else 0.0
            ax.text(
                0.50, 0.55, f"No Alarm Triggered\n(Max P = {max_p:.4f})",
                transform=ax.transAxes,
                fontsize=11, fontweight='bold', color='#555555', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.85)
            )

        ax.set_title(f"({chr(65 + idx)}) {title}\n[Disk Serial: {serial}]", fontsize=12, fontweight='bold', pad=8)
        ax.set_ylabel("Prediction Probability", fontsize=11)
        ax.set_xlabel("Time (Observation Date)", fontsize=11)

        # Standardized Y-axis range across ALL subplots [0.0, 1.0]
        ax.set_ylim(-0.02, 1.05)

        # X-axis range: Shift failure line slightly left by providing a clean ~10 day post-failure window
        start_date = dates[0]
        if has_failed and failure_date is not None:
            end_date = max(dates[-1], failure_date) + pd.Timedelta(days=10)
        else:
            end_date = dates[-1] + pd.Timedelta(days=10)
        ax.set_xlim(start_date, end_date)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=9.5)
        ax.legend(fontsize=9.5, loc='upper left', frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[Plot Saved] Clean Operational 4-Quadrant Timeline Image -> {output_path}")


def main():
    hdd_name = "HGST_20HUH721212ALN604"
    hdd_path = os.path.join(PROJECT_ROOT, "data", "splitted", hdd_name)
    threshold_map = load_threshold_map()

    target_models = ["gru", "lgbm"]

    results_dir = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(f"  OPERATIONAL LIFE-CYCLE TIMELINE ANALYSIS (POST-FAILURE MARGIN & TN ANNOTATION)  ")
    print("=" * 80)

    for m in target_models:
        model_upper = "LightGBM" if m == "lgbm" else m.upper()
        lookup_key = "LGBM" if m == "lgbm" else model_upper
        thresh = threshold_map.get((hdd_name, lookup_key), DEFAULT_THRESHOLDS.get((hdd_name, lookup_key), 0.16))

        train_df, val_df, test_df, features = load_dataset(hdd_path, model=m)
        is_sequence_model = (m in ['lstm', 'gru'])
        ckpt_window_size = config.WINDOW_SIZE if is_sequence_model else None

        model = load_checkpoint(
            m, "none", config.SEED, config.TARGET_LEAD_TIME, hdd_path,
            input_dim=len(features), extra_tag="cw0_focal0", features=features, window_size=ckpt_window_size
        )

        if model is None:
            print(f"Warning: Checkpoint not found for model '{m}' on '{hdd_name}'. Skipping.")
            continue

        model_type = 'pytorch_class' if is_sequence_model or m == 'mlp' else m

        evaluator = RollingEvaluator(
            model=model,
            features=features,
            window_size=config.WINDOW_SIZE if is_sequence_model else 1,
            device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
            model_type=model_type,
            seed=config.SEED
        )

        print(f"\n[Running Inference] Model: {model_upper} | Threshold: {thresh:.4f}")
        raw_preds = evaluator.get_raw_predictions(test_df, lead_time=config.TARGET_LEAD_TIME)

        samples = select_best_quadrant_samples(raw_preds, thresh)

        out_img_path = os.path.join(results_dir, f"operational_timeline_4quadrants_{model_upper}.png")
        plot_operational_timelines(samples, model_upper, thresh, hdd_name, out_img_path)


if __name__ == "__main__":
    main()
