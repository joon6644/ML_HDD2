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


def extract_alarm_counts_per_hdd(hdd_name: str, model_name: str, threshold: float):
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

    for disk in raw_preds:
        serial = disk['serial_number']
        has_failed = disk['has_failed']
        preds = disk['preds']

        alarm_mask = (preds >= threshold)
        total_alarm_count = int(np.sum(alarm_mask))
        obs_days = len(preds)

        records.append({
            'serial_number': serial,
            'hdd': hdd_name,
            'model': model_upper,
            'threshold': threshold,
            'has_failed': int(has_failed),
            'total_obs_days': obs_days,
            'total_alarm_count': total_alarm_count,
            'alarm_ratio': total_alarm_count / obs_days if obs_days > 0 else 0.0
        })

    df_records = pd.DataFrame(records)
    failed_df = df_records[df_records['has_failed'] == 1]
    healthy_df = df_records[df_records['has_failed'] == 0]

    print(f" -> Failed Disks Mean Alarms: {failed_df['total_alarm_count'].mean():.2f} (Median: {failed_df['total_alarm_count'].median():.1f})")
    print(f" -> Healthy Disks Mean Alarms: {healthy_df['total_alarm_count'].mean():.2f} (Median: {healthy_df['total_alarm_count'].median():.1f})")

    return df_records


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
    print(f"  HGST_20HUH721212ALN604 - TOTAL ALARM COUNT PER HDD ANALYSIS (4 MODELS 2x2)  ")
    print("=" * 80)

    all_records_list = []
    model_data = {}

    for m in models:
        model_upper = model_titles[m].upper()
        lookup_key = "LGBM" if model_upper == "LIGHTGBM" else ("XGB" if model_upper == "XGBOOST" else model_upper)
        thresh = threshold_map.get((hdd_name, lookup_key), DEFAULT_THRESHOLDS.get((hdd_name, lookup_key), 0.5))

        df_rec = extract_alarm_counts_per_hdd(hdd_name, m, thresh)
        all_records_list.append(df_rec)
        model_data[m] = {
            'title': model_titles[m],
            'lookup_key': lookup_key,
            'threshold': thresh,
            'df_records': df_rec
        }

    # Save CSVs
    full_df = pd.concat(all_records_list, ignore_index=True)
    full_csv_path = os.path.join(results_dir, f"{hdd_name}_alarm_count_per_hdd_all_models.csv")
    full_df.to_csv(full_csv_path, index=False, encoding='utf-8-sig')

    # Summary table
    summary_rows = []
    for m in models:
        m_key = "LGBM" if m == "lgbm" else ("XGB" if m == "xgb" else m.upper())
        df_m = full_df[full_df['model'] == m_key]
        failed_m = df_m[df_m['has_failed'] == 1]
        healthy_m = df_m[df_m['has_failed'] == 0]

        summary_rows.append({
            'HDD': hdd_name,
            'Model': model_titles[m],
            'Threshold': model_data[m]['threshold'],
            'Failed_HDD_Mean_Alarms': failed_m['total_alarm_count'].mean(),
            'Failed_HDD_Median_Alarms': failed_m['total_alarm_count'].median(),
            'Failed_HDD_Max_Alarms': failed_m['total_alarm_count'].max(),
            'Healthy_HDD_Mean_Alarms': healthy_m['total_alarm_count'].mean(),
            'Healthy_HDD_Median_Alarms': healthy_m['total_alarm_count'].median(),
            'Healthy_HDD_Max_Alarms': healthy_m['total_alarm_count'].max(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(results_dir, f"{hdd_name}_alarm_count_per_hdd_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')

    # Generate 2x2 Grid Image Plot
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Total Alarm Count Distribution per HDD Unit - HGST (20HUH721212ALN604)", fontsize=16, fontweight="bold", y=0.98)

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

        failed_counts = df_rec[df_rec['has_failed'] == 1]['total_alarm_count'].values
        failed_alarm_triggered = failed_counts[failed_counts > 0]

        mean_alarms = float(np.mean(failed_alarm_triggered)) if len(failed_alarm_triggered) > 0 else 0.0
        median_alarms = float(np.median(failed_alarm_triggered)) if len(failed_alarm_triggered) > 0 else 0.0

        # Hist of total alarm counts per failed disk (up to 150 alarms)
        alarm_counts_plot = failed_alarm_triggered[failed_alarm_triggered <= 150]
        bins = np.arange(-0.5, 150.5, 3.0)

        counts, edges, patches = ax.hist(
            alarm_counts_plot,
            bins=bins,
            color=color_dict[m],
            edgecolor="black",
            alpha=0.75,
            linewidth=1.0
        )

        if len(failed_alarm_triggered) > 0:
            ax.axvline(median_alarms, color="darkred", linestyle="--", linewidth=2.5, label=f"Median Alarms/Disk: {median_alarms:.1f}")
            ax.axvline(mean_alarms, color="darkorange", linestyle="-.", linewidth=2.5, label=f"Mean Alarms/Disk: {mean_alarms:.1f}")

        ax.set_title(f"({chr(65+idx)}) {m_title}", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Total Alarm Count per Failed HDD Unit", fontsize=11)
        ax.set_ylabel("Disk Count", fontsize=11)
        ax.set_xlim(-2, 152)
        ax.set_xticks(range(0, 151, 15))
        ax.legend(fontsize=10.5, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_img_path = os.path.join(results_dir, "HGST_20HUH721212ALN604_alarm_count_per_hdd_2x2.png")
    plt.savefig(output_img_path, dpi=300)
    plt.close()

    print("\n" + "=" * 80)
    print(f" [SUCCESS] Total Alarm Count per HDD 2x2 Grid Image saved to:\n  {output_img_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
