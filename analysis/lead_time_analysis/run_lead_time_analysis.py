import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import torch
    # Patch torch.load to default weights_only=False for PyTorch 2.6+ compatibility with saved checkpoints
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except ImportError:
    torch = None

# Ensure experiments directory is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

import config
from data_loader import load_dataset
from checkpoint_utils import load_checkpoint
from evaluator import RollingEvaluator

# Force PIPELINE_VERSION to match saved checkpoint version (pv2)
config.PIPELINE_VERSION = "v2"

# Manufacturer friendly name mapping
MANUFACTURER_MAP = {
    "ST12000NM0007": "Seagate (ST12000NM0007)",
    "HGST_20HUH721212ALN604": "HGST (20HUH721212ALN604)",
    "TOSHIBA_20MG07ACA14TA": "Toshiba (20MG07ACA14TA)"
}

# Operational thresholds from validation-set tuning
DEFAULT_THRESHOLDS = {
    ("ST12000NM0007", "LGBM"): 0.08,
    ("ST12000NM0007", "XGB"): 0.11,
    ("ST12000NM0007", "LSTM"): 0.30,
    ("ST12000NM0007", "GRU"): 0.32,
    ("HGST_20HUH721212ALN604", "LGBM"): 0.99,
    ("HGST_20HUH721212ALN604", "XGB"): 0.46,
    ("HGST_20HUH721212ALN604", "LSTM"): 0.11,
    ("HGST_20HUH721212ALN604", "GRU"): 0.16,
    ("TOSHIBA_20MG07ACA14TA", "LGBM"): 0.13,
    ("TOSHIBA_20MG07ACA14TA", "XGB"): 0.13,
    ("TOSHIBA_20MG07ACA14TA", "LSTM"): 0.18,
    ("TOSHIBA_20MG07ACA14TA", "GRU"): 0.13,
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


def extract_all_first_alarms_for_hdd(hdd_name: str, model_name: str, threshold_map: dict):
    """
    Extracts ALL first alarms on failed HDDs without filtering hit horizon.
    Calculates lead_time_days = (actual_failure_date - first_alarm_date).days.
    """
    hdd_path = os.path.join(PROJECT_ROOT, "data", "splitted", hdd_name)
    mfr_name = MANUFACTURER_MAP.get(hdd_name, hdd_name)
    model_upper = model_name.upper()
    thresh = threshold_map.get((hdd_name, model_upper), DEFAULT_THRESHOLDS.get((hdd_name, model_upper), 0.5))

    print(f"\n[Processing] HDD Dataset: {hdd_name} ({mfr_name}) | Model: {model_upper} | Threshold: {thresh:.4f}")

    # 1. Load dataset
    train_df, val_df, test_df, features = load_dataset(hdd_path, model=model_name.lower())

    # 2. Load Checkpoint
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

    # 3. Rolling Inference Evaluator
    evaluator = RollingEvaluator(
        model=model,
        features=features,
        window_size=config.WINDOW_SIZE if is_sequence_model else 1,
        device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
        model_type=model_type,
        seed=config.SEED
    )

    raw_preds = evaluator.get_raw_predictions(test_df, lead_time=config.TARGET_LEAD_TIME)

    # 4. Extract ALL first alarm dates for failed HDDs
    records = []
    total_failed_disks = 0
    disks_with_alarm = 0

    for disk in raw_preds:
        has_failed = disk['has_failed']
        if not has_failed:
            continue
        
        total_failed_disks += 1
        serial = disk['serial_number']
        failure_date = disk['failure_date']
        dates = disk['dates']
        preds = disk['preds']

        alarm_indices = np.where(preds >= thresh)[0]
        if len(alarm_indices) > 0:
            disks_with_alarm += 1
            first_alarm_idx = alarm_indices[0]
            first_alarm_date = pd.to_datetime(dates[first_alarm_idx])
            days_to_failure = (failure_date - first_alarm_date).days
            alarm_score = float(preds[first_alarm_idx])

            records.append({
                'serial_number': serial,
                'hdd': hdd_name,
                'manufacturer': mfr_name,
                'model': model_upper,
                'threshold': thresh,
                'actual_failure_date': failure_date,
                'first_alarm_date': first_alarm_date,
                'lead_time_days': days_to_failure,
                'alarm_score': alarm_score
            })

    df_records = pd.DataFrame(records)
    print(f" -> Failed Disks Total : {total_failed_disks}")
    print(f" -> Disks with Alarm  : {disks_with_alarm} ({disks_with_alarm / total_failed_disks:.1%})")

    return df_records, total_failed_disks, thresh


def plot_discrete_histogram(df_records: pd.DataFrame, hdd_name: str, model_name: str, total_failed: int, thresh: float, output_dir: str):
    """
    Plots a discrete 1-day unit histogram for lead time (0 to 30 days).
    """
    mfr_name = MANUFACTURER_MAP.get(hdd_name, hdd_name)
    model_upper = model_name.upper()

    # Filter lead times for the 0..30 days horizon display
    lead_times = df_records['lead_time_days'].values
    lead_times_in_range = lead_times[(lead_times >= 0) & (lead_times <= 30)]

    plt.figure(figsize=(11, 6))
    sns.set_theme(style="whitegrid")

    color_map = {
        "ST12000NM0007": "#2b5c8f",
        "HGST_20HUH721212ALN604": "#d9534f",
        "TOSHIBA_20MG07ACA14TA": "#2e7d32"
    }
    plot_color = color_map.get(hdd_name, "#1f77b4")

    lead_times_all = df_records['lead_time_days'].values if len(df_records) > 0 else np.array([])
    mean_lt = float(np.mean(lead_times_all)) if len(lead_times_all) > 0 else 0.0
    median_lt = float(np.median(lead_times_all)) if len(lead_times_all) > 0 else 0.0
    max_lt = int(np.max(lead_times_all)) if len(lead_times_all) > 0 else 0
    min_lt = int(np.min(lead_times_all)) if len(lead_times_all) > 0 else 0

    lead_times_plot = lead_times_all[lead_times_all <= 180]
    bins = np.arange(-0.5, 180.5, 3.0)

    counts, edges, patches = plt.hist(
        lead_times_plot,
        bins=bins,
        color=plot_color,
        edgecolor="black",
        alpha=0.75,
        linewidth=1.2
    )

    if len(lead_times_all) > 0:
        plt.axvline(median_lt, color="darkred", linestyle="--", linewidth=2.5, label=f"Median Lead Time: {median_lt:.1f} days")
        plt.axvline(mean_lt, color="darkorange", linestyle="-.", linewidth=2.5, label=f"Mean Lead Time: {mean_lt:.1f} days")

    # Metadata textbox
    stats_text = (
        f"HDD Dataset : {hdd_name}\n"
        f"Model       : {model_upper}\n"
        f"Threshold   : {thresh:.4f}\n"
        f"Failed Disks: {total_failed}\n"
        f"Alarm Disks : {len(lead_times_all)} ({len(lead_times_all)/total_failed:.1%})\n"
        f"----------------------------------\n"
        f"Median Lead Time : {median_lt:.1f} days\n"
        f"Mean Lead Time   : {mean_lt:.1f} days\n"
        f"Min / Max Range  : {min_lt}d / {max_lt}d"
    )
    plt.gca().text(
        0.45, 0.93, stats_text,
        transform=plt.gca().transAxes,
        fontsize=10.5,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#bbbbbb", alpha=0.92)
    )

    plt.title(f"First Alarm Lead Time Distribution - {mfr_name} ({model_upper})", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Lead Time (Days from First Alarm to Actual Failure)", fontsize=11, labelpad=10)
    plt.ylabel("Disk Count", fontsize=11, labelpad=10)
    plt.xlim(-2, 182)
    plt.xticks(range(0, 181, 15), fontsize=9.5)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10.5, loc="upper right")
    plt.tight_layout()
    plt.xlim(-0.7, 30.7)
    plt.ylim(0, max(counts) * 1.15 if len(counts) > 0 and max(counts) > 0 else 10)
    plt.xticks(range(0, 31, 1), fontsize=9)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10.5, loc="upper right")
    plt.tight_layout()

    out_image_path = os.path.join(output_dir, f"lead_time_{hdd_name}_{model_upper}.png")
    plt.savefig(out_image_path, dpi=300)
    plt.close()

    print(f"[Plot Saved] -> {out_image_path}")
    return out_image_path


def main():
    parser = argparse.ArgumentParser(description="Generate 1-day unit lead time histograms for GRU model across 3 datasets")
    parser.add_argument('--model', type=str, default='gru', help='Target model (default: gru)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for lead time results (default: results/lead_time_analysis)')
    args = parser.parse_args()

    model_name = args.model.lower()
    hdd_list = ['ST12000NM0007', 'HGST_20HUH721212ALN604', 'TOSHIBA_20MG07ACA14TA']
    threshold_map = load_threshold_map()

    if args.output_dir:
        results_dir = args.output_dir
    else:
        results_dir = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
    
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("    SECTION 5.4.1 LEAD TIME ANALYSIS (1-DAY STEP DISCRETE HISTOGRAMS)    ")
    print(f" Model                     : {model_name.upper()}")
    print(f" Datasets                  : {hdd_list}")
    print(f" Output Results Directory  : {results_dir}")
    print("=" * 80)

    generated_images = []
    for hdd in hdd_list:
        df_records, total_failed, thresh = extract_all_first_alarms_for_hdd(hdd, model_name, threshold_map)
        
        # Save raw CSV to results directory
        csv_path = os.path.join(results_dir, f"lead_time_{hdd}_{model_name.upper()}_all_alarms.csv")
        df_records.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # Plot 1-day discrete histogram to results directory
        img_path = plot_discrete_histogram(df_records, hdd, model_name, total_failed, thresh, results_dir)
        generated_images.append(img_path)

    print("\n" + "=" * 80)
    print("                   SUCCESSFULLY GENERATED ALL 3 HISTOGRAMS               ")
    print("=" * 80)
    for idx, path in enumerate(generated_images):
        print(f" Image [{idx+1}/3]: {path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
