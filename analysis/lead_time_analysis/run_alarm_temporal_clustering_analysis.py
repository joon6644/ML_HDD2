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

# Trendy Academic Palette (Option B)
STYLE_CONFIG = {
    "lgbm": {"fill": "#2b5c8f", "edge": "#000000", "title": "LightGBM"},
    "xgb":  {"fill": "#d95f02", "edge": "#000000", "title": "XGBoost"},
    "lstm": {"fill": "#7570b3", "edge": "#000000", "title": "LSTM"},
    "gru":  {"fill": "#1b9e77", "edge": "#000000", "title": "GRU"}
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


def model_data_key(m: str) -> str:
    m_u = m.upper()
    return "LGBM" if m_u == "LIGHTGBM" else ("XGB" if m_u == "XGBOOST" else m_u)


def main():
    hdd_name = "HGST_20HUH721212ALN604"
    results_dir = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
    analysis_dir = os.path.join(PROJECT_ROOT, "analysis", "lead_time_analysis")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, f"{hdd_name}_all_alarm_events_temporal_clustering.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(analysis_dir, f"{hdd_name}_all_alarm_events_temporal_clustering.csv")

    models = ["lgbm", "xgb", "lstm", "gru"]
    model_titles = {
        "lgbm": "LightGBM",
        "xgb": "XGBoost",
        "lstm": "LSTM",
        "gru": "GRU"
    }

    if os.path.exists(csv_path):
        print(f"[Data Loader] Loading pre-extracted temporal clustering events from {csv_path}")
        full_df = pd.read_csv(csv_path)
    else:
        threshold_map = load_threshold_map()
        all_events_list = []
        for m in models:
            model_upper = model_titles[m].upper()
            lookup_key = "LGBM" if model_upper == "LIGHTGBM" else ("XGB" if model_upper == "XGBOOST" else model_upper)
            thresh = threshold_map.get((hdd_name, lookup_key), DEFAULT_THRESHOLDS.get((hdd_name, lookup_key), 0.5))

            df_ev, _ = extract_all_alarm_events(hdd_name, m, thresh)
            all_events_list.append(df_ev)

        full_df = pd.concat(all_events_list, ignore_index=True)
        out_csv = os.path.join(results_dir, f"{hdd_name}_all_alarm_events_temporal_clustering.csv")
        full_df.to_csv(out_csv, index=False, encoding='utf-8-sig')

    # Publication Quality Styling Settings
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Calibri', 'sans-serif']
    plt.rcParams['axes.edgecolor'] = '#111111'
    plt.rcParams['axes.linewidth'] = 1.1

    sns.set_theme(style="ticks", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300, sharey=True)
    fig.suptitle(
        "Normalized Temporal Alarm Distribution Before Failure — HGST (20HUH721212ALN604)",
        fontsize=16, fontweight="bold", y=0.98, color="#111111"
    )

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    labels = ["(a)", "(b)", "(c)", "(d)"]
    bins = np.linspace(0, 180, 36) # 35 bins across 0 to 180 days

    # Pre-calculate data to standardize y-axis limits across all 4 subplots
    plot_data_dict = {}
    max_pct = 0.0

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
        style = STYLE_CONFIG[m]
        df_m = plot_data_dict[m]
        m_title = style['title']
        n_events = len(df_m)

        if n_events > 0:
            sns.histplot(
                data=df_m,
                x='days_to_failure',
                bins=bins,
                stat='percent',
                kde=False,              # 1. KDE line removed
                color=style["fill"],     # 2. Trendy Academic Solid Fill
                edgecolor=style["edge"], # Crisp Black Border
                alpha=0.72,
                linewidth=0.9,
                ax=ax
            )

        # 3. Vertical lines and legends completely removed
        ax.set_title(
            f"{labels[idx]} {m_title}  (n = {n_events})",
            fontsize=13, fontweight="bold", pad=10, loc="left", color="#111111"
        )
        
        ax.set_xlabel("Days Remaining to Failure (Days)", fontsize=11, fontweight="bold", labelpad=6)
        if c == 0:
            ax.set_ylabel("Normalized Alarm Frequency (%)", fontsize=11, fontweight="bold", labelpad=6)
        else:
            ax.set_ylabel("")

        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 181, 30))
        ax.set_ylim(0, ylim_top)

        # Deduplicate y-axis labels and tick numbers
        ax.label_outer()
        ax.set_xlabel("Days Remaining to Failure (Days)", fontsize=11, fontweight="bold", labelpad=6)

        ax.grid(True, axis="y", linestyle=":", alpha=0.20, color="#666666")
        ax.grid(False, axis="x")
        sns.despine(ax=ax, top=True, right=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_img_path = os.path.join(results_dir, "HGST_20HUH721212ALN604_4models_2x2_alarm_temporal_clustering.png")
    
    plt.savefig(output_img_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 80)
    print(" [SUCCESS] Temporal Alarm Clustering 2x2 Grid Image Updated!")
    print("  - Removed KDE curve line")
    print("  - Removed vertical reference lines and legends completely")
    print("  - Matched Trendy Academic styling (Option B colors, black borders, 35 bins, label_outer)")
    print(f" Saved to:\n  -> {output_img_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
