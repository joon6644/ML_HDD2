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
from checkpoint_utils import CHECKPOINT_DIR
from evaluator import RollingEvaluator

config.PIPELINE_VERSION = "v2"

DEFAULT_THRESHOLDS = {
    ("HGST_20HUH721212ALN604", "LGBM"): 0.998,
    ("HGST_20HUH721212ALN604", "XGB"): 0.478,
    ("HGST_20HUH721212ALN604", "LSTM"): 0.327,
    ("HGST_20HUH721212ALN604", "GRU"): 0.662,
}

MODEL_TITLES = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "lstm": "LSTM",
    "gru": "GRU"
}

# Standard academic blue for prediction curve
STANDARD_LINE_COLOR = "#1f77b4"


def _read_master_csv(path: str) -> pd.DataFrame:
    for encoding in ('utf-8-sig', 'cp949'):
        for sep in (',', '\t'):
            try:
                df = pd.read_csv(path, encoding=encoding, sep=sep)
                if df.shape[1] > 1:
                    return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
    raise ValueError(f"Could not parse master CSV: {path}")


def load_threshold_map(seed: int = 42) -> dict:
    threshold_map = DEFAULT_THRESHOLDS.copy()
    db_path = os.path.join(PROJECT_ROOT, "results", "experiments.db")
    
    if os.path.exists(db_path):
        try:
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql("SELECT 데이터, Model, [Threshold (Proposed-Opt)] FROM master_proposed_threshold_results WHERE Seed=?", conn, params=(seed,))
                for _, row in df.iterrows():
                    hdd = str(row['데이터']).strip()
                    model_name = str(row['Model']).upper()
                    thresh = float(row['Threshold (Proposed-Opt)'])
                    threshold_map[(hdd, model_name)] = thresh
            print(f"[Threshold Loader] Successfully loaded Seed {seed} proposed thresholds from SQLite -> {db_path}")
            return threshold_map
        except Exception as e:
            print(f"[Threshold Loader] Warning loading from SQLite ({e}). Trying CSV...")
            
    master_csv = os.path.join(PROJECT_ROOT, "results", "master_proposed_threshold_results.csv")
    if os.path.exists(master_csv):
        try:
            df = _read_master_csv(master_csv)
            if 'Seed' in df.columns:
                df = df[df['Seed'] == seed]
            for _, row in df.iterrows():
                hdd = str(row['데이터']).strip()
                model_name = str(row['Model']).upper()
                thresh_col = 'Threshold (Proposed-Opt)' if 'Threshold (Proposed-Opt)' in df.columns else 'Threshold'
                thresh = float(row[thresh_col])
                threshold_map[(hdd, model_name)] = thresh
            print(f"[Threshold Loader] Loaded thresholds from CSV -> {master_csv}")
        except Exception as e:
            print(f"[Threshold Loader] Warning: Could not read master CSV ({e}). Using defaults.")
    return threshold_map


def load_checkpoint_flexible(model_name: str, seed: int, lead_time: int, dataset_name: str, input_dim: int):
    clean_ds = os.path.basename(dataset_name.rstrip('/\\'))
    candidates = [
        f"{model_name.lower()}_none_cw0_focal0_lead{lead_time}_seed{seed}_{clean_ds}_pv2.ckpt",
        f"{model_name.lower()}_none_cw0_focal0_lead{lead_time}_seed{seed}_{clean_ds}_pv3.ckpt",
        f"{model_name.lower()}_none_lead{lead_time}_seed{seed}_{clean_ds}.ckpt",
        f"{model_name.lower()}_none_cw0_focal0_lead{lead_time}_seed{seed}_{clean_ds}.ckpt"
    ]
    for filename in candidates:
        full_path = os.path.join(CHECKPOINT_DIR, filename)
        if os.path.exists(full_path):
            print(f"[Checkpoint Manager] Found checkpoint: {filename}")
            payload = torch.load(full_path, map_location='cpu', weights_only=False)
            ckpt_type = payload.get("type")
            if ckpt_type == "pytorch":
                cls_name = payload["class_name"]
                st_dict = payload["state_dict"]
                arch_kwargs = {k: v for k, v in payload.get("arch_kwargs", {}).items() if v is not None}
                if cls_name == "LSTMClass":
                    from models.lstm import LSTMClass
                    m = LSTMClass(input_dim=input_dim, **arch_kwargs)
                elif cls_name == "GRUClass":
                    from models.gru import GRUClass
                    m = GRUClass(input_dim=input_dim, **arch_kwargs)
                else:
                    raise ValueError(f"Unknown PyTorch model class: {cls_name}")
                m.load_state_dict(st_dict)
                if torch.cuda.is_available():
                    m = m.cuda()
                return m
            elif ckpt_type == "sklearn_or_tree":
                return payload["model_obj"]
    return None


def select_best_operational_samples(raw_preds, threshold: float, lead_time: int = 30):
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
        thresh = threshold_map.get((hdd_name, lookup_key), DEFAULT_THRESHOLDS.get((hdd_name, lookup_key), 0.16))

        train_df, val_df, test_df, features = load_dataset(hdd_path, model=m)
        is_sequence_model = (m in ['lstm', 'gru'])

        model = load_checkpoint_flexible(
            model_name=m,
            seed=config.SEED,
            lead_time=config.TARGET_LEAD_TIME,
            dataset_name=hdd_name,
            input_dim=len(features)
        )

        if model is None:
            print(f"Warning: Checkpoint not found for model '{m}' on '{hdd_name}'. Skipping.")
            continue

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
