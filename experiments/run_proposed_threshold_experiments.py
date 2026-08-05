import os
import sys
import gc
import argparse
import time
from datetime import datetime
import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

import config
from data_loader import load_dataset, create_binary_target, build_sequences
from imbalance import apply_imbalance_treatment, validate_config_compatibility
from checkpoint_utils import load_checkpoint, save_checkpoint
from models import train_rf_model, train_lgbm_model, train_xgb_model, train_mlp_model, train_lstm_model, train_gru_model
from evaluator import (
    RollingEvaluator,
    save_experiment_result_to_csv
)

# ------------------------------------------------------------------------------
# In-Memory Dataset Cache for Zero-IO Overhead
# ------------------------------------------------------------------------------
_DATASET_CACHE = {}

def get_cached_dataset(data_path: str, drop_failure_day: bool, model: str):
    cache_key = (data_path, drop_failure_day, model)
    if cache_key not in _DATASET_CACHE:
        print(f"\n[Data Loader Cache] Loading dataset into RAM: {data_path}...")
        train_df, val_df, test_df, features = load_dataset(data_path, drop_failure_day_in_train=drop_failure_day, model=model)
        _DATASET_CACHE[cache_key] = (train_df, val_df, test_df, features)
    return _DATASET_CACHE[cache_key]


def run_proposed_threshold_experiments():
    parser = argparse.ArgumentParser(description="Script 2: Proposed Disk-Level Threshold Search & Evaluation (3 Datasets x 4 Models x 5 Seeds)")
    parser.add_argument('--seeds', type=int, nargs='+', default=config.ALL_SEEDS, help='Target seeds')
    parser.add_argument('--datasets', type=str, nargs='+', default=config.ALL_DATASETS, help='Target dataset names')
    parser.add_argument('--models', type=str, nargs='+', default=config.ALL_MODELS, help='Target model names')
    parser.add_argument('--imbalance', type=str, default='none', help='Imbalance strategy')
    parser.add_argument('--drop-failure-day', action='store_true', default=config.DROP_FAILURE_DAY_IN_TRAIN, help='Drop failure day in train')
    parser.add_argument('--dry-run', action='store_true', help='Print planned tasks without execution')
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    master_csv_path = os.path.join(project_dir, "results", "master_proposed_threshold_results.csv")
    key_cols = ['Model', '데이터', 'Seed']

    # 1. Build planned task list (Loop Order: DATASET -> MODEL -> SEED for RAM caching & minimal I/O)
    tasks = []
    for dataset in args.datasets:
        if os.path.isabs(dataset) or os.path.exists(dataset):
            data_path = dataset
        else:
            data_path = os.path.join(project_dir, "data", "splitted", dataset)

        for model in args.models:
            for seed in args.seeds:
                tasks.append({
                    'dataset': dataset,
                    'data_path': data_path,
                    'model': model,
                    'seed': seed
                })

    print("=" * 80)
    print(" SCRIPT 2: PROPOSED DISK-LEVEL INDEPENDENT THRESHOLD SEARCH & EVALUATION RUNNER")
    print(f" Datasets ({len(args.datasets)}) : {args.datasets}")
    print(f" Models ({len(args.models)})   : {args.models}")
    print(f" Seeds ({len(args.seeds)})    : {args.seeds}")
    print(f" Total Planned Runs  : {len(tasks)}")
    print(f" Threshold Search    : Proposed Disk-Level Val Set (FAR <= 1% Constraint -> Max Disk Recall)")
    print(f" Lead Time Sampled   : First positive prediction across ALL failed disks")
    print(f" Master CSV Output   : {master_csv_path}")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN MODE] Printing planned execution sequence:")
        for idx, t in enumerate(tasks, 1):
            print(f"  {idx:02d}. DATA={t['dataset']} | MODEL={t['model'].upper()} | SEED={t['seed']}")
        print("\nDry run completed. No model training or evaluation was performed.")
        return

    start_time = time.time()
    successful_runs = 0
    failed_runs = 0

    # 2. Iterate per dataset (Outer Loop)
    unique_datasets = list(dict.fromkeys([t['dataset'] for t in tasks]))

    for ds in unique_datasets:
        ds_tasks = [t for t in tasks if t['dataset'] == ds]
        print("\n" + "#" * 80)
        print(f" [START DATASET BATCH: DATASET = {ds}] ({len(ds_tasks)} experiments planned)")
        print("#" * 80)

        for idx, t in enumerate(ds_tasks, 1):
            data_path = t['data_path']
            model_name = t['model']
            seed = t['seed']
            imbalance = args.imbalance

            print(f"\n>>> [Dataset: {ds} | Task {idx}/{len(ds_tasks)}] MODEL={model_name.upper()} | SEED={seed}")
            print("-" * 80)

            try:
                # A. Load cached dataset
                train_df, val_df, test_df, features = get_cached_dataset(data_path, args.drop_failure_day, model_name)

                is_sequence_model = model_name in ['lstm', 'gru']
                window_size = config.WINDOW_SIZE if is_sequence_model else 1

                # B. Checkpoint reload or model training
                cached_model = load_checkpoint(model_name, imbalance, seed, config.TARGET_LEAD_TIME, data_path, input_dim=len(features), features=features, window_size=window_size if is_sequence_model else None)

                if is_sequence_model:
                    X_val_seq, y_val_seq = build_sequences(val_df, features, window_size=window_size, lead_time=config.TARGET_LEAD_TIME)
                    X_test_seq, y_test_seq = build_sequences(test_df, features, window_size=window_size, lead_time=config.TARGET_LEAD_TIME)
                    if cached_model is None:
                        X_train_seq, y_train_seq = build_sequences(train_df, features, window_size=window_size, lead_time=config.TARGET_LEAD_TIME)
                else:
                    y_val = create_binary_target(val_df, lead_time=config.TARGET_LEAD_TIME)
                    X_val_2d = val_df[features].values
                    if cached_model is None:
                        y_train = create_binary_target(train_df, lead_time=config.TARGET_LEAD_TIME)
                        X_train_2d = train_df[features].values

                if cached_model is not None:
                    model = cached_model
                    model_type = 'pytorch_class' if is_sequence_model else model_name
                else:
                    print(f"Training {model_name.upper()} model (Seed={seed})...")
                    if is_sequence_model:
                        X_tr_proc, y_tr_proc = apply_imbalance_treatment(X_train_seq, y_train_seq, strategy=imbalance, seed=seed, dataset_path=data_path, lead_time=config.TARGET_LEAD_TIME, drop_failure_day=args.drop_failure_day)
                        if model_name == 'lstm':
                            model = train_lstm_model(X_tr_proc, y_tr_proc, X_val_seq, y_val_seq, seed=seed)
                        else:
                            model = train_gru_model(X_tr_proc, y_tr_proc, X_val_seq, y_val_seq, seed=seed)
                        model_type = 'pytorch_class'
                    else:
                        X_tr_proc, y_tr_proc = apply_imbalance_treatment(X_train_2d, y_train, strategy=imbalance, seed=seed, dataset_path=data_path, lead_time=config.TARGET_LEAD_TIME, drop_failure_day=args.drop_failure_day)
                        if model_name == 'lgbm':
                            model = train_lgbm_model(X_tr_proc, y_tr_proc, X_val_2d, y_val, seed=seed, use_gpu=config.USE_GPU)
                            model_type = 'lgbm'
                        elif model_name == 'xgb':
                            model = train_xgb_model(X_tr_proc, y_tr_proc, X_val_2d, y_val, seed=seed, use_gpu=config.USE_GPU)
                            model_type = 'xgb'

                    if config.SAVE_MODEL_WEIGHTS:
                        save_checkpoint(model, model_name, imbalance, seed, config.TARGET_LEAD_TIME, data_path, features=features, window_size=window_size if is_sequence_model else None)

                # C. Single Sequential Inference (Run exactly once!)
                evaluator = RollingEvaluator(
                    model=model,
                    features=features,
                    window_size=window_size,
                    device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
                    model_type=model_type,
                    seed=seed
                )

                print("Performing single sequential inference on Validation set...")
                raw_preds_val = evaluator.get_raw_predictions(val_df)
                print("Performing single sequential inference on Test set...")
                raw_preds_test = evaluator.get_raw_predictions(test_df)

                # D. Search Independent Optimal Threshold on Validation Set (Proposed Disk-Level)
                th_proposed, max_val_rec = evaluator.find_best_threshold_proposed_level(raw_preds_val, max_far=config.MAX_FAR)
                print(f"[Proposed Threshold Search] Best Threshold on Val Set: {th_proposed:.4f} (Max Val Disk Recall @ FAR <= 1%: {max_val_rec:.4%})")

                # E. Evaluate Test Set using th_proposed
                # Proposed Disk-level evaluation on Test Set using th_proposed
                proposed_metrics, report_df = evaluator.evaluate_proposed_level(raw_preds_test, threshold=th_proposed)

                # F. Record to Master CSV
                row_data = {
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Model': model_name.upper(),
                    '데이터': os.path.basename(ds),
                    'Seed': seed,
                    'Threshold (Proposed-Opt)': round(th_proposed, 4),

                    # Proposed Disk-level Metrics (evaluated at th_proposed)
                    'Disk Precision': round(proposed_metrics['precision'], 4),
                    'Disk Recall': round(proposed_metrics['recall'], 4),
                    'Disk F1': round(proposed_metrics['f1'], 4),
                    'Disk FAR (%)': round(proposed_metrics['far'] * 100, 2),
                    'Mean Lead Time': round(proposed_metrics['mean_lead_time'], 2),
                    'Median Lead Time': round(proposed_metrics['median_lead_time'], 2),
                    'Std Lead Time': round(proposed_metrics['std_lead_time'], 2),
                    'EDR@15': round(proposed_metrics['edr_15'], 4)
                }

                save_experiment_result_to_csv(master_csv_path, row_data, key_cols)
                successful_runs += 1

            except Exception as e:
                failed_runs += 1
                print(f"[ERROR] Task failed (DATA={ds}, MODEL={model_name}, SEED={seed}): {e}")
                import traceback
                traceback.print_exc()

            finally:
                # Memory Cleanup
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(" SCRIPT 2 COMPLETED!")
    print(f" Total Elapsed Time : {elapsed / 60:.2f} minutes")
    print(f" Successful Runs    : {successful_runs} / {len(tasks)}")
    print(f" Failed Runs        : {failed_runs} / {len(tasks)}")
    print(f" Master CSV Path    : {master_csv_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_proposed_threshold_experiments()
