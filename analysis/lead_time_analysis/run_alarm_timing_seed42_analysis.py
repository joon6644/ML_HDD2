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

SEED = 42
DATASETS = ['ST12000NM0007', 'HGST_20HUH721212ALN604', 'TOSHIBA_20MG07ACA14TA']
MODELS = ['lgbm', 'xgb', 'lstm', 'gru']

MANUFACTURER_MAP = {
    "ST12000NM0007": "Seagate (ST12000NM0007)",
    "HGST_20HUH721212ALN604": "HGST (20HUH721212ALN604)",
    "TOSHIBA_20MG07ACA14TA": "Toshiba (20MG07ACA14TA)"
}


def _read_master_csv(path: str) -> pd.DataFrame:
    """Robustly reads the master results CSV, which is normally written as
    utf-8-sig/comma but can end up as cp949/tab-delimited after being opened
    and re-saved in Excel."""
    for encoding in ('utf-8-sig', 'cp949'):
        for sep in (',', '\t'):
            try:
                df = pd.read_csv(path, encoding=encoding, sep=sep)
                if df.shape[1] > 1:
                    return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
    raise ValueError(f"Could not parse master CSV with known encodings/separators: {path}")


def load_threshold_map(seed: int) -> dict:
    """Reads (dataset, model) -> threshold from master_proposed_threshold_results.csv for the given seed."""
    master_csv = os.path.join(PROJECT_ROOT, "results", "master_proposed_threshold_results.csv")
    threshold_map = {}
    df = _read_master_csv(master_csv)
    df = df[df['Seed'].astype(int) == seed]
    for _, row in df.iterrows():
        threshold_map[(str(row['데이터']).strip(), str(row['Model']).upper())] = float(row['Threshold (Proposed-Opt)'])
    return threshold_map


def evaluate_one(dataset: str, model_name: str, threshold: float):
    data_path = os.path.join(PROJECT_ROOT, "data", "splitted", dataset)
    is_sequence_model = model_name in ['lstm', 'gru']
    window_size = config.WINDOW_SIZE if is_sequence_model else 1

    _, _, test_df, features = load_dataset(data_path, model=model_name)

    model = load_checkpoint(
        model_name, "none", SEED, config.TARGET_LEAD_TIME, data_path,
        input_dim=len(features), features=features,
        window_size=window_size if is_sequence_model else None
    )
    if model is None:
        raise FileNotFoundError(f"Checkpoint missing for model='{model_name}' dataset='{dataset}' seed={SEED}")

    model_type = 'pytorch_class' if is_sequence_model else model_name
    evaluator = RollingEvaluator(
        model=model, features=features, window_size=window_size,
        device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
        model_type=model_type, seed=SEED
    )

    raw_preds = evaluator.get_raw_predictions(test_df, lead_time=config.TARGET_LEAD_TIME)
    _, report_df = evaluator.evaluate_proposed_level(raw_preds, threshold=threshold)

    # Attach each disk's first-observed date in the test window (needed to express
    # false-alarm timing as "days since disk entered observation", since a censored
    # disk has no failure date to anchor against).
    first_seen = {d['serial_number']: pd.to_datetime(d['dates']).min() for d in raw_preds}
    report_df['first_seen_date'] = report_df['serial_number'].map(first_seen)
    report_df['days_since_observed'] = (
        pd.to_datetime(report_df['first_alarm_date']) - report_df['first_seen_date']
    ).dt.days

    return report_df


def plot_facet_grid(all_reports: dict, results_dir: str):
    sns.set_theme(style="whitegrid")

    # Figure 1: Failed disks - lead time from first alarm to actual failure (days)
    fig1, axes1 = plt.subplots(len(DATASETS), len(MODELS), figsize=(20, 12), sharex=True)
    # Figure 2: Healthy disks - false alarm timing since observation start (days).
    # Not sharex: each dataset spans a very different day range on the log axis.
    fig2, axes2 = plt.subplots(len(DATASETS), len(MODELS), figsize=(20, 12), sharex=False)

    for i, dataset in enumerate(DATASETS):
        for j, model_name in enumerate(MODELS):
            key = (dataset, model_name.upper())
            report_df = all_reports[key]
            mfr = MANUFACTURER_MAP.get(dataset, dataset)

            # --- Failed HDD: first-alarm lead time (hits only) ---
            # Display window is capped at 180d for readability; the cap is applied
            # BEFORE computing bins/median so the median line always matches what's drawn.
            hits = report_df[(report_df['has_failed'] == 1) & (report_df['is_hit'] == 1)]
            lead_times_full = hits['days_to_failure_at_alarm'].dropna().values
            lead_times = lead_times_full[lead_times_full <= 180]
            n_clipped_1 = len(lead_times_full) - len(lead_times)
            ax1 = axes1[i, j]
            if len(lead_times) > 0:
                bins = np.arange(-0.5, lead_times.max() + 3.5, 3.0)
                ax1.hist(lead_times, bins=bins, color="#2b5c8f", edgecolor="black", alpha=0.8)
                ax1.axvline(np.median(lead_times), color="darkred", linestyle="--", linewidth=1.8,
                            label=f"Median {np.median(lead_times):.0f}d")
                ax1.legend(fontsize=8)
            title_suffix_1 = f" (+{n_clipped_1} >180d)" if n_clipped_1 > 0 else ""
            ax1.set_title(f"{mfr}\n{model_name.upper()} (n={len(lead_times_full)}{title_suffix_1})", fontsize=9)
            if i == len(DATASETS) - 1:
                ax1.set_xlabel("Days from First Alarm to Failure", fontsize=8)
            if j == 0:
                ax1.set_ylabel("Disk Count", fontsize=8)

            # --- Healthy HDD: false alarm timing since observation start ---
            # Disk histories here span years, so a linear axis buries everything near
            # zero. Log1p-scale the x-axis instead of clipping, so the median line
            # (computed on the exact same values that are drawn) always lines up.
            fa = report_df[(report_df['has_failed'] == 0) & (report_df['is_false_alarm'] == 1)]
            days_since = fa['days_since_observed'].dropna().values
            days_since = days_since[days_since >= 0]
            ax2 = axes2[i, j]
            if len(days_since) > 0:
                log_vals = np.log1p(days_since)
                bins2 = np.linspace(0, max(log_vals.max(), 1.0), 25)
                ax2.hist(log_vals, bins=bins2, color="#d9534f", edgecolor="black", alpha=0.8)
                med = np.median(days_since)
                ax2.axvline(np.log1p(med), color="darkblue", linestyle="--", linewidth=1.8,
                            label=f"Median {med:.0f}d")
                ax2.legend(fontsize=8)
                tick_days = [0, 1, 7, 30, 90, 365, 1095, 3650]
                tick_days = [d for d in tick_days if d <= days_since.max() * 1.05]
                ax2.set_xticks(np.log1p(tick_days))
                ax2.set_xticklabels([str(d) for d in tick_days], fontsize=7)
            ax2.set_title(f"{mfr}\n{model_name.upper()} (n={len(days_since)})", fontsize=9)
            if i == len(DATASETS) - 1:
                ax2.set_xlabel("Days Since Observation Start (log scale)", fontsize=8)
            if j == 0:
                ax2.set_ylabel("Disk Count", fontsize=8)

    fig1.suptitle(f"Failed HDD: First-Alarm Lead Time Before Actual Failure (Seed={SEED})", fontsize=14, fontweight="bold")
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = os.path.join(results_dir, f"seed{SEED}_failed_hdd_first_alarm_leadtime_grid.png")
    fig1.savefig(out1, dpi=300)
    plt.close(fig1)

    fig2.suptitle(f"Healthy HDD: False Alarm Timing Since Observation Start (Seed={SEED})", fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    out2 = os.path.join(results_dir, f"seed{SEED}_healthy_hdd_false_alarm_timing_grid.png")
    fig2.savefig(out2, dpi=300)
    plt.close(fig2)

    print(f"[Plot Saved] -> {out1}")
    print(f"[Plot Saved] -> {out2}")
    return out1, out2


def main():
    results_dir = os.path.join(PROJECT_ROOT, "analysis", "lead_time_analysis")
    os.makedirs(results_dir, exist_ok=True)

    threshold_map = load_threshold_map(SEED)

    print("=" * 80)
    print(f" ALARM TIMING ANALYSIS (SEED={SEED}) - Failed-HDD First Alarm & Healthy-HDD False Alarms")
    print(f" Datasets : {DATASETS}")
    print(f" Models   : {MODELS}")
    print("=" * 80)

    all_reports = {}
    for dataset in DATASETS:
        for model_name in MODELS:
            thresh = threshold_map.get((dataset, model_name.upper()))
            if thresh is None:
                raise KeyError(f"No Seed={SEED} threshold found for ({dataset}, {model_name.upper()}) in master_proposed_threshold_results.csv")
            print(f"\n[Processing] {dataset} | {model_name.upper()} | threshold={thresh}")
            report_df = evaluate_one(dataset, model_name, thresh)
            all_reports[(dataset, model_name.upper())] = report_df

            csv_path = os.path.join(results_dir, f"seed{SEED}_alarm_report_{dataset}_{model_name.upper()}.csv")
            report_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    plot_facet_grid(all_reports, results_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
