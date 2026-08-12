"""
Sweeps the decision threshold for each (dataset, model) combination and records how
Proposed Disk-Level lead time (median/std/mean) and recall/FAR/precision shift as the
threshold changes. Inference is run once per (dataset, model); only the cheap
threshold-dependent aggregation (evaluator.evaluate_proposed_level) is repeated per
threshold value, so the sweep itself is fast.

Output: one CSV per (dataset, model) with one row per threshold, plus a combined
facet-grid plot of Median Lead Time (+/- Std) vs Threshold.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import torch
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

DATASETS = ['ST12000NM0007', 'HGST_20HUH721212ALN604', 'TOSHIBA_20MG07ACA14TA']
MODELS = ['lgbm', 'xgb', 'lstm', 'gru']

MANUFACTURER_MAP = {
    "ST12000NM0007": "Seagate (ST12000NM0007)",
    "HGST_20HUH721212ALN604": "HGST (20HUH721212ALN604)",
    "TOSHIBA_20MG07ACA14TA": "Toshiba (20MG07ACA14TA)"
}


def get_raw_preds(dataset: str, model_name: str, seed: int):
    data_path = os.path.join(PROJECT_ROOT, "data", "splitted", dataset)
    is_sequence_model = model_name in ['lstm', 'gru']
    window_size = config.WINDOW_SIZE if is_sequence_model else 1

    _, _, test_df, features = load_dataset(data_path, model=model_name)

    model = load_checkpoint(
        model_name, "none", seed, config.TARGET_LEAD_TIME, data_path,
        input_dim=len(features), features=features,
        window_size=window_size if is_sequence_model else None
    )
    if model is None:
        raise FileNotFoundError(f"Checkpoint missing for model='{model_name}' dataset='{dataset}' seed={seed}")

    model_type = 'pytorch_class' if is_sequence_model else model_name
    evaluator = RollingEvaluator(
        model=model, features=features, window_size=window_size,
        device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
        model_type=model_type, seed=seed
    )
    raw_preds = evaluator.get_raw_predictions(test_df, lead_time=config.TARGET_LEAD_TIME)
    return evaluator, raw_preds


def sweep_thresholds(evaluator: RollingEvaluator, raw_preds: list, thresholds: np.ndarray) -> pd.DataFrame:
    records = []
    for thresh in thresholds:
        metrics, _ = evaluator.evaluate_proposed_level(raw_preds, threshold=float(thresh))
        records.append({
            'threshold': round(float(thresh), 4),
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'far': metrics['far'],
            'mean_lead_time': metrics['mean_lead_time'],
            'median_lead_time': metrics['median_lead_time'],
            'std_lead_time': metrics['std_lead_time'],
            'edr_15': metrics['edr_15'],
            'hits': metrics['tp'],
            'false_alarms': metrics['fp'],
            'misses': metrics['fn'],
            'correct_rejections': metrics['tn'],
        })
    return pd.DataFrame(records)


def plot_facet_grid(all_sweeps: dict, results_dir: str, seed: int):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(len(DATASETS), len(MODELS), figsize=(20, 12), sharex=True)

    for i, dataset in enumerate(DATASETS):
        for j, model_name in enumerate(MODELS):
            key = (dataset, model_name.upper())
            df = all_sweeps[key]
            mfr = MANUFACTURER_MAP.get(dataset, dataset)
            ax = axes[i, j]

            valid = df.dropna(subset=['median_lead_time'])
            valid = valid[valid['hits'] > 0]
            if len(valid) > 0:
                x = valid['threshold'].values
                y = valid['median_lead_time'].values
                n = valid['hits'].values.astype(float)

                ax.plot(x, y, color="#0a1f38", linewidth=2.6, alpha=1.0, label="Median LT", zorder=3)

                ax.fill_between(
                    x, y - valid['std_lead_time'].values, y + valid['std_lead_time'].values,
                    color="#2b5c8f", alpha=0.10, label="+/- Std", zorder=1
                )

                # Secondary axis: n (hit count) backing each threshold, so low-n
                # (statistically unreliable) stretches are visible rather than implied.
                ax2 = ax.twinx()
                ax2.plot(x, n, color="#c0392b", linestyle="--", linewidth=1.3, alpha=1.0, label="n (hits)", zorder=2)
                ax2.set_ylim(bottom=0)
                ax2.grid(False)
                ax2.tick_params(axis='y', labelsize=7, colors="#c0392b")
                if j == len(MODELS) - 1:
                    ax2.set_ylabel("n (hits) [dashed -> right axis]", fontsize=8, color="#c0392b")

                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6.5, loc="upper right")
            ax.set_title(f"{mfr}\n{model_name.upper()}", fontsize=9)
            if i == len(DATASETS) - 1:
                ax.set_xlabel("Threshold", fontsize=8)
            if j == 0:
                ax.set_ylabel("Lead Time (days) [solid -> left axis]", fontsize=8)

    fig.suptitle(f"Proposed Disk-Level Lead Time vs Threshold (Seed={seed})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(results_dir, f"seed{seed}_leadtime_vs_threshold_grid.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Plot Saved] -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Sweep threshold and record lead time median/std per (dataset, model)")
    parser.add_argument('--seed', type=int, default=config.SEED, help='Seed to evaluate (must have a saved checkpoint)')
    parser.add_argument('--datasets', type=str, nargs='+', default=DATASETS)
    parser.add_argument('--models', type=str, nargs='+', default=MODELS)
    parser.add_argument('--step', type=float, default=0.01, help='Threshold step size')
    args = parser.parse_args()

    results_dir = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
    os.makedirs(results_dir, exist_ok=True)

    thresholds = np.round(np.arange(0.01, 1.00, args.step), 4)

    print("=" * 80)
    print(f" THRESHOLD -> LEAD TIME SWEEP (SEED={args.seed})")
    print(f" Datasets   : {args.datasets}")
    print(f" Models     : {args.models}")
    print(f" Thresholds : {thresholds[0]} .. {thresholds[-1]} (step={args.step}, n={len(thresholds)})")
    print("=" * 80)

    all_sweeps = {}
    for dataset in args.datasets:
        for model_name in args.models:
            print(f"\n[Processing] {dataset} | {model_name.upper()}")
            evaluator, raw_preds = get_raw_preds(dataset, model_name, args.seed)
            sweep_df = sweep_thresholds(evaluator, raw_preds, thresholds)
            all_sweeps[(dataset, model_name.upper())] = sweep_df

            csv_path = os.path.join(results_dir, f"seed{args.seed}_threshold_sweep_{dataset}_{model_name.upper()}.csv")
            sweep_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"[CSV Saved] -> {csv_path}")

    plot_facet_grid(all_sweeps, results_dir, args.seed)
    print("\nDone.")


if __name__ == "__main__":
    main()
