import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def extract_all_first_alarms(hdd_name: str, model_name: str, threshold: float):
    hdd_path = os.path.join(PROJECT_ROOT, "data", "splitted", hdd_name)
    model_upper = model_name.upper()

    print(f"\n[Processing] Model: {model_upper} | Threshold: {threshold:.4f}")

    train_df, val_df, test_df, features = load_dataset(hdd_path, model=model_name.lower())

    is_sequence_model = (model_name.lower() in ['lstm', 'gru'])
    ckpt_window_size = config.WINDOW_SIZE if is_sequence_model else None
    ckpt_tag = "cw0_focal0"

    model = load_checkpoint(
        model_name.lower(), "none", config.SEED, config.TARGET_LEAD_TIME, hdd_path,
        input_dim=len(features), extra_tag=ckpt_tag, features=features, window_size=ckpt_window_size
    )

    if model is None:
        raise FileNotFoundError(f"Checkpoint missing for model '{model_name}' on HDD '{hdd_name}'")

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

    records = []
    total_failed_disks = 0

    for disk in raw_preds:
        has_failed = disk['has_failed']
        if not has_failed:
            continue
        
        total_failed_disks += 1
        serial = disk['serial_number']
        failure_date = disk['failure_date']
        dates = disk['dates']
        preds = disk['preds']

        alarm_indices = np.where(preds >= threshold)[0]
        if len(alarm_indices) > 0:
            first_alarm_idx = alarm_indices[0]
            first_alarm_date = pd.to_datetime(dates[first_alarm_idx])
            days_to_failure = (failure_date - first_alarm_date).days
            alarm_score = float(preds[first_alarm_idx])

            if days_to_failure >= 0:
                records.append({
                    'serial_number': serial,
                    'hdd': hdd_name,
                    'model': model_upper,
                    'threshold': threshold,
                    'actual_failure_date': failure_date,
                    'first_alarm_date': first_alarm_date,
                    'lead_time_days': days_to_failure,
                    'alarm_score': alarm_score
                })

    df_records = pd.DataFrame(records)
    print(f" -> Total Failed: {total_failed_disks} | First Alarm Triggered Disks: {len(df_records)} ({len(df_records)/total_failed_disks:.1%})")
    return df_records, total_failed_disks


def main():
    hdd_name = "HGST_20HUH721212ALN604"
    threshold_map = load_threshold_map()
    models = ["lgbm", "xgb", "lstm", "gru"]
    model_titles = {
        "lgbm": "LightGBM",
        "xgb": "XGBoost",
        "lstm": "LSTM",
        "gru": "GRU"
    }

    results_dir = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(f"  HGST_20HUH721212ALN604 - CLEAN 2x2 LEAD TIME DISTRIBUTION  ")
    print("=" * 80)

    model_data = {}
    for m in models:
        model_upper = model_titles[m].upper()
        lookup_key = "LGBM" if model_upper == "LIGHTGBM" else ("XGB" if model_upper == "XGBOOST" else model_upper)
        thresh = threshold_map.get((hdd_name, lookup_key), DEFAULT_THRESHOLDS.get((hdd_name, lookup_key), 0.5))
        df_records, total_failed = extract_all_first_alarms(hdd_name, m, thresh)
        
        csv_path = os.path.join(results_dir, f"lead_time_{hdd_name}_{lookup_key}_all_alarms.csv")
        df_records.to_csv(csv_path, index=False, encoding='utf-8-sig')

        model_data[m] = {
            'title': model_titles[m],
            'lookup_key': lookup_key,
            'threshold': thresh,
            'df_records': df_records,
            'total_failed': total_failed
        }

    # Generate Clean 2x2 Grid Image Plot (No white box, no threshold in titles)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("First Alarm Lead Time Distribution - HGST (20HUH721212ALN604)", fontsize=16, fontweight="bold", y=0.98)

    color_dict = {
        "lgbm": "#1f77b4",
        "xgb": "#ff7f0e",
        "lstm": "#2ca02c",
        "gru": "#d62728"
    }

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, m in enumerate(models):
        r, c = positions[idx]
        ax = axes[r, c]

        info = model_data[m]
        df_rec = info['df_records']
        m_title = info['title']

        lead_times = df_rec['lead_time_days'].values if len(df_rec) > 0 else np.array([])
        
        mean_lt = float(np.mean(lead_times)) if len(lead_times) > 0 else 0.0
        median_lt = float(np.median(lead_times)) if len(lead_times) > 0 else 0.0

        lead_times_plot = lead_times[lead_times <= 180]
        bins = np.arange(-0.5, 180.5, 3.0)

        counts, edges, patches = ax.hist(
            lead_times_plot,
            bins=bins,
            color=color_dict[m],
            edgecolor="black",
            alpha=0.75,
            linewidth=1.0
        )

        if len(lead_times) > 0:
            ax.axvline(median_lt, color="darkred", linestyle="--", linewidth=2.5, label=f"Median Lead Time: {median_lt:.1f} days")
            ax.axvline(mean_lt, color="darkorange", linestyle="-.", linewidth=2.5, label=f"Mean Lead Time: {mean_lt:.1f} days")

        # Subplot Title (Clean, NO threshold)
        ax.set_title(f"({chr(65+idx)}) {m_title}", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Lead Time (Days from First Alarm to Failure)", fontsize=11)
        ax.set_ylabel("Disk Count", fontsize=11)
        ax.set_xlim(-2, 182)
        ax.set_xticks(range(0, 181, 15))
        ax.legend(fontsize=10.5, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_img_path = os.path.join(results_dir, "HGST_20HUH721212ALN604_4models_2x2_lead_time.png")
    plt.savefig(output_img_path, dpi=300)
    plt.close()

    print("\n" + "=" * 80)
    print(f" [SUCCESS] Clean 2x2 Grid Image saved to:\n  {output_img_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
