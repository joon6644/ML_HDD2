import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, ANALYSIS_DIR)

try:
    import torch
except ImportError:
    torch = None

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

# Standard academic blue for prediction curve
STANDARD_LINE_COLOR = "#1f77b4"


def select_best_operational_samples(raw_preds, threshold: float, lead_time: int = None):
    if lead_time is None:
        lead_time = config.TARGET_LEAD_TIME
    ontime_candidates = []
    early_candidates = []
    cens_early_candidates = []
    missed_candidates = []

    for disk in raw_preds:
        has_failed = disk['has_failed']
        failure_date = pd.to_datetime(disk['failure_date']) if (has_failed and disk['failure_date'] is not None) else None
        dates = pd.to_datetime(disk['dates'])
        preds = disk['preds']

        alarm_mask = (preds >= threshold)
        alarm_indices = np.where(alarm_mask)[0]

        if len(alarm_indices) > 0:
            first_alarm_idx = alarm_indices[0]
            first_alarm_date = dates[first_alarm_idx]
            if has_failed and failure_date is not None:
                days_to_fail = (failure_date - first_alarm_date).days
                if 0 <= days_to_fail <= lead_time:
                    ontime_candidates.append((disk, days_to_fail, np.max(preds)))
                elif days_to_fail > lead_time:
                    early_candidates.append((disk, days_to_fail, np.max(preds)))
            else:
                cens_early_candidates.append((disk, None, np.max(preds)))
        else:
            if has_failed:
                missed_candidates.append((disk, None, np.max(preds)))

    # Select representative samples based on peak score & operational characteristics
    ontime_sample = sorted(ontime_candidates, key=lambda x: x[2], reverse=True)[0][0] if ontime_candidates else None
    early_sample = sorted(early_candidates, key=lambda x: x[2], reverse=True)[0][0] if early_candidates else None
    cens_early_sample = sorted(cens_early_candidates, key=lambda x: x[2], reverse=True)[0][0] if cens_early_candidates else None
    missed_sample = sorted(missed_candidates, key=lambda x: x[2], reverse=True)[0][0] if missed_candidates else None

    return {
        'On-time': ontime_sample,
        'Early': early_sample,
        'Censored Early': cens_early_sample,
        'Missed': missed_sample
    }


def plot_operational_timelines(samples: dict, model_key: str, threshold: float, hdd_name: str, output_paths: list):
    m_title = MODEL_TITLES.get(model_key, model_key.upper())

    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Calibri', 'sans-serif']
    plt.rcParams['axes.edgecolor'] = '#111111'
    plt.rcParams['axes.linewidth'] = 1.1

    sns.set_theme(style="ticks", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2), dpi=300, sharey=True)

    fig.suptitle(
        f"Operational Life-Cycle Timelines across 4 Alarm Categories — {m_title} ({hdd_name})",
        fontsize=15, fontweight="bold", y=0.98, color="#111111"
    )

    categories = [
        ('On-time', (0, 0), "(a) On-time Alarm"),
        ('Early', (0, 1), "(b) Early Alarm"),
        ('Censored Early', (1, 0), "(c) Censored Early Alarm"),
        ('Missed', (1, 1), "(d) Missed Failure")
    ]

    max_days_plot = 365

    for key, (r, c), title in categories:
        ax = axes[r, c]
        disk_data = samples.get(key)

        if disk_data is None:
            ax.text(0.5, 0.5, f"No sample found for category: {key}", ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=12.5, fontweight='bold', pad=9, loc='left', color='#111111')
            ax.set_ylim(-0.02, 1.05)
            ax.set_ylabel("Prediction Probability" if c == 0 else "")
            sns.despine(ax=ax, top=True, right=True)
            continue

        dates = pd.to_datetime(disk_data['dates'])
        preds = disk_data['preds']
        has_failed = disk_data['has_failed']
        failure_date = pd.to_datetime(disk_data['failure_date']) if (has_failed and disk_data['failure_date'] is not None) else None

        # Truncate view to last 365 days for clean, standardized visualization
        if len(dates) > max_days_plot:
            dates = dates[-max_days_plot:]
            preds = preds[-max_days_plot:]

        # 1. Prediction Probability Curve P(t)
        ax.plot(
            dates, preds,
            color=STANDARD_LINE_COLOR, linewidth=1.8,
            label=r"Prediction Probability $P(t)$"
        )

        # 2. Shading above threshold for alarm regions
        alarm_mask = (preds >= threshold)
        alarm_indices = np.where(alarm_mask)[0]
        if len(alarm_indices) > 0:
            ax.fill_between(
                dates, threshold, preds,
                where=(preds >= threshold),
                color="#e41a1c", alpha=0.20, interpolate=True
            )
            # Mark First Alarm Event
            first_alarm_idx = alarm_indices[0]
            first_alarm_date = dates[first_alarm_idx]
            ax.axvline(
                first_alarm_date, color="#e41a1c", linestyle=":", linewidth=1.5,
                label=f"First Alarm Event ({first_alarm_date.strftime('%Y-%m-%d')})"
            )

        # 3. Decision Threshold Line
        ax.axhline(
            threshold, color='#c00000', linestyle='--', linewidth=1.4,
            label=f"Decision Threshold ({threshold:.3f})"
        )

        # 4. Actual Failure Event Line (For Failed Disks: On-time, Early, Missed)
        if has_failed and failure_date is not None:
            ax.axvline(
                failure_date, color='#000000', linestyle='--', linewidth=1.5,
                label=f"Actual Failure Event ({failure_date.strftime('%Y-%m-%d')})"
            )

        # Subplot Title
        ax.set_title(
            title,
            fontsize=12.5, fontweight='bold', pad=9, loc='left', color='#111111'
        )

        ax.set_ylim(-0.02, 1.05)

        # Set 5 evenly spaced date ticks across the time range
        tick_indices = np.linspace(0, len(dates) - 1, 5, dtype=int)
        tick_dates = [dates[i] for i in tick_indices]
        ax.set_xticks(tick_dates)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        ax.tick_params(axis='x', which='both', labelbottom=True, labelsize=9.5)
        ax.set_xlabel("")

        if c == 0:
            ax.set_ylabel("Prediction Probability", fontsize=11, fontweight="bold", labelpad=6)
        else:
            ax.set_ylabel("")

        ax.grid(True, axis="y", linestyle=":", alpha=0.25, color="#666666")
        ax.grid(False, axis="x")
        sns.despine(ax=ax, top=True, right=True)

        if r == 0 and c == 0:
            ax.legend(fontsize=9.0, loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#cccccc', framealpha=0.90)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.28)

    for out_path in output_paths:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Saved updated operational category timeline image -> {output_paths[0]}")


def main():
    hdd_name = "HGST_20HUH721212ALN604"
    hdd_path = os.path.join(PROJECT_ROOT, "data", "splitted", hdd_name)
    threshold_map = load_threshold_map(seed=config.SEED)

    target_models = ["gru", "lgbm", "xgb", "lstm"]

    results_dir = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(f" UPDATING OPERATIONAL LIFE-CYCLE TIMELINES (4 OPERATIONAL CATEGORIES) ")
    print("=" * 80)

    for m in target_models:
        model_upper = MODEL_TITLES[m]
        lookup_key = "LGBM" if m == "lgbm" else m.upper()
        thresh = threshold_map[(hdd_name, lookup_key)]

        train_df, val_df, test_df, features = load_dataset(hdd_path, model=m)
        is_sequence_model = (m in ['lstm', 'gru'])

        model = load_analysis_model(
            dataset=hdd_name,
            model_name=m,
            seed=config.SEED,
            features=features
        )

        model_type = 'pytorch_class' if is_sequence_model else m

        evaluator = RollingEvaluator(
            model=model,
            features=features,
            window_size=config.WINDOW_SIZE if is_sequence_model else 1,
            device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
            model_type=model_type,
            seed=config.SEED
        )

        print(f"\n[Inference & Plotting] Model: {model_upper} | Proposed Threshold: {thresh:.4f}")
        raw_preds = evaluator.get_raw_predictions(test_df, lead_time=config.TARGET_LEAD_TIME)

        samples = select_best_operational_samples(raw_preds, thresh, lead_time=config.TARGET_LEAD_TIME)

        out1 = os.path.join(results_dir, f"operational_timeline_4quadrants_{model_upper}.png")
        plot_operational_timelines(samples, m, thresh, hdd_name, [out1])


if __name__ == "__main__":
    main()
