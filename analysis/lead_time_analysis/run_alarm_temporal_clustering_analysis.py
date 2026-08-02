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


def extract_all_alarm_events(hdd_name: str, model_name: str, threshold: float):
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

    alarm_events = []
    total_failed = 0

    for disk in raw_preds:
        if not disk['has_failed']:
            continue
        total_failed += 1
        serial = disk['serial_number']
        failure_date = disk['failure_date']
        dates = disk['dates']
        preds = disk['preds']

        alarm_indices = np.where(preds >= threshold)[0]
        for idx in alarm_indices:
            alarm_date = pd.to_datetime(dates[idx])
            days_to_failure = (failure_date - alarm_date).days
            if days_to_failure >= 0:
                alarm_events.append({
                    'serial_number': serial,
                    'hdd': hdd_name,
                    'model': model_upper,
                    'threshold': threshold,
                    'alarm_date': alarm_date,
                    'failure_date': failure_date,
                    'days_to_failure': days_to_failure,
                    'alarm_score': float(preds[idx])
                })

    df_events = pd.DataFrame(alarm_events)
    print(f" -> Total Failed HDDs: {total_failed} | Total Alarm Events Collected: {len(df_events)}")
    return df_events, total_failed


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
    print(f"  HGST_20HUH721212ALN604 - NORMALIZED ALARM FREQUENCY ANALYSIS (4 MODELS 2x2)  ")
    print("=" * 80)

    model_events = {}
    all_events_list = []

    for m in models:
        model_upper = model_titles[m].upper()
        lookup_key = "LGBM" if model_upper == "LIGHTGBM" else ("XGB" if model_upper == "XGBOOST" else model_upper)
        thresh = threshold_map.get((hdd_name, lookup_key), DEFAULT_THRESHOLDS.get((hdd_name, lookup_key), 0.5))

        df_ev, total_failed = extract_all_alarm_events(hdd_name, m, thresh)
        all_events_list.append(df_ev)

        model_events[m] = {
            'title': model_titles[m],
            'lookup_key': lookup_key,
            'threshold': thresh,
            'df_events': df_ev,
            'total_failed': total_failed
        }

    # Combine all events
    if len(all_events_list) > 0:
        full_df = pd.concat(all_events_list, ignore_index=True)
        full_csv = os.path.join(results_dir, f"{hdd_name}_all_alarm_events_temporal_clustering.csv")
        full_df.to_csv(full_csv, index=False, encoding='utf-8-sig')
    else:
        full_df = pd.DataFrame()

    # Generate 2x2 Plot with Seaborn histplot using stat='percent'
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Normalized Temporal Alarm Distribution Before Failure - HGST (20HUH721212ALN604)", fontsize=16, fontweight="bold", y=0.98)

    color_dict = {
        "lgbm": "#1f77b4",
        "xgb": "#ff7f0e",
        "lstm": "#2ca02c",
        "gru": "#d62728"
    }

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    bins = np.arange(-0.5, 180.5, 5.0) # 5-day step binning for clean % representation

    # Find max percentage to standardize y-axis limits across all 4 subplots
    max_pct = 0.0
    plot_data_dict = {}

    for m in models:
        df_m = full_df[(full_df['model'] == model_data_key(m)) & (full_df['days_to_failure'] <= 180)]
        days_m = df_m['days_to_failure'].values
        n_tot = len(days_m)
        if n_tot > 0:
            counts, _ = np.histogram(days_m, bins=bins)
            pcts = (counts / n_tot) * 100.0
            if len(pcts) > 0:
                max_pct = max(max_pct, np.max(pcts))
        plot_data_dict[m] = df_m

    ylim_top = min(100.0, max(15.0, np.ceil(max_pct + 2.0)))

    for idx, m in enumerate(models):
        r, c = positions[idx]
        ax = axes[r, c]

        info = model_events[m]
        df_m = plot_data_dict[m]
        m_title = info['title']

        if len(df_m) > 0:
            sns.histplot(
                data=df_m,
                x='days_to_failure',
                bins=bins,
                stat='percent',
                kde=True,
                color=color_dict[m],
                edgecolor="black",
                alpha=0.70,
                linewidth=1.0,
                ax=ax
            )

        # Vertical reference lines
        ax.axvline(10, color="red", linestyle="--", linewidth=2.0, label="Imminent Zone (0-10d)")
        ax.axvline(30, color="goldenrod", linestyle="-.", linewidth=2.0, label="Operational Window (30d)")

        ax.set_title(f"({chr(65+idx)}) {m_title}", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Days Remaining to Failure (Days to Failure)", fontsize=11)
        ax.set_ylabel("Normalized Alarm Frequency (%)", fontsize=11)
        ax.set_xlim(-2, 182)
        ax.set_ylim(0, ylim_top)
        ax.set_xticks(range(0, 181, 15))
        ax.legend(fontsize=10.5, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_img_path = os.path.join(results_dir, "HGST_20HUH721212ALN604_4models_2x2_alarm_temporal_clustering.png")
    plt.savefig(output_img_path, dpi=300)
    plt.close()

    print("\n" + "=" * 80)
    print(f" [SUCCESS] Normalized Temporal Alarm Clustering 2x2 Grid Image saved to:\n  {output_img_path}")
    print("=" * 80 + "\n")


def model_data_key(m: str) -> str:
    m_u = m.upper()
    return "LGBM" if m_u == "LIGHTGBM" else ("XGB" if m_u == "XGBOOST" else m_u)


if __name__ == "__main__":
    main()
