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
import prediction_cache
from evaluator import (
    RollingEvaluator,
    find_best_threshold_row_level,
    calculate_row_level_metrics,
    save_run_results,
    read_results_table,
    export_results_table_to_csv,
    csv_export_path,
    get_db_path,
    THRESHOLD_GRID,
    RESULT_TABLES,
    RUNS_TABLE,
)

# ------------------------------------------------------------------------------
# In-Memory Dataset Cache for Zero-IO Overhead
# ------------------------------------------------------------------------------
_DATASET_CACHE = {}

def get_cached_dataset(data_path: str, drop_failure_day: bool, model: str):
    cache_key = (data_path, drop_failure_day, model)
    if cache_key not in _DATASET_CACHE:
        _DATASET_CACHE.clear()
        gc.collect()
        print(f"\n[Data Loader Cache] Loading dataset into RAM: {data_path}...")
        train_df, val_df, test_df, features = load_dataset(data_path, drop_failure_day_in_train=drop_failure_day, model=model)
        _DATASET_CACHE[cache_key] = (train_df, val_df, test_df, features)
    return _DATASET_CACHE[cache_key]


import sqlite3

def is_experiment_already_completed(results_dir: str, model_name: str, dataset_name: str, seed: int) -> bool:
    """
    Checks whether a run is already recorded in the authoritative SQLite DB. The CSV
    exports are derived and are never consulted here -- one source of truth decides
    what has been run.

    A missing DB/table means 'nothing recorded yet'; a malformed one propagates
    rather than being silently treated as 'not completed' (which would quietly
    retrain and overwrite results).
    """
    db_path = get_db_path(results_dir)
    model_norm = model_name.upper()
    ds_norm = os.path.basename(dataset_name)

    for table in RESULT_TABLES:
        df = read_results_table(db_path, table)
        if df.empty:
            return False
        match = (
            (df['model'].astype(str).str.upper() == model_norm)
            & (df['dataset'].astype(str) == ds_norm)
            & (df['seed'].astype(int) == int(seed))
        )
        if not match.any():
            return False

    return True


def reset_master_results(results_dir: str):
    """
    Drops every result table and its derived CSV export so results accumulate from
    scratch. Prediction caches are left alone -- they are keyed by run and are
    overwritten as each run is recomputed.
    """
    db_path = get_db_path(results_dir)

    print("\n" + "=" * 80)
    print(" [RESET] Dropping every result table and its CSV export")
    # A reset that only half-succeeds leaves stale rows that later look like real
    # results, so any failure here stops the run instead of printing a warning.
    for table in RESULT_TABLES:
        path = csv_export_path(results_dir, table)
        if os.path.exists(path):
            os.remove(path)
            print(f"  -> Removed derived CSV export: {path}")

    if os.path.exists(db_path):
        with sqlite3.connect(db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            for table in RESULT_TABLES:
                cursor.execute(f"DROP TABLE IF EXISTS [{table}]")
            conn.commit()
            print(f"  -> Dropped tables {list(RESULT_TABLES)} from {db_path}")
    print("=" * 80 + "\n")


def run_unified_threshold_experiments():
    parser = argparse.ArgumentParser(description="UNIFIED THRESHOLD EXPERIMENT RUNNER: Dual Threshold Search & Evaluation (3 Datasets x 4 Models x 5 Seeds)")
    parser.add_argument('--seeds', type=int, nargs='+', default=config.ALL_SEEDS, help='Target seeds')
    parser.add_argument('--datasets', type=str, nargs='+', default=config.ALL_DATASETS, help='Target dataset names')
    parser.add_argument('--models', type=str, nargs='+', default=config.ALL_MODELS, help='Target model names')
    parser.add_argument('--imbalance', type=str, default='none', help='Imbalance strategy')
    parser.add_argument('--drop-failure-day', action='store_true', default=config.DROP_FAILURE_DAY_IN_TRAIN, help='Drop failure day in train')
    parser.add_argument('--reset', action='store_true', default=False, help='Drop every result table and its CSV export before running')
    parser.add_argument('--dry-run', action='store_true', help='Print planned tasks without execution')
    parser.add_argument('--eval-only', '--inference-only', action='store_true', default=False, help='Run inference and threshold evaluation only using pre-saved model checkpoints (do not retrain and do not skip existing results)')
    parser.add_argument('--overwrite', '--force-eval', action='store_true', default=False, help='Force re-evaluation and overwrite existing results for specified dataset/model/seed without running full reset')
    parser.add_argument('--keep-going', action='store_true', default=False, help='Continue the batch after a task fails (default: abort immediately on the first failure)')
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    if torch is None and any(m in ['lstm', 'gru'] for m in args.models):
        print("\n" + "!" * 80)
        print("[WARNING] PyTorch (torch) is not installed in the currently active Python environment.")
        print(" -> LSTM / GRU models require PyTorch (with GPU/CUDA support).")
        print(r" -> Please run using the project virtual environment:")
        print(r"    .\venv\Scripts\python.exe experiments\run_unified_threshold_experiments.py")
        print("!" * 80 + "\n")

    # Reset existing results ONLY if explicitly requested via --reset flag
    if args.reset and not args.dry_run:
        reset_master_results(results_dir)

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
    print(" UNIFIED DUAL THRESHOLD SEARCH & EVALUATION RUNNER")
    print(f" Datasets ({len(args.datasets)}) : {args.datasets}")
    print(f" Models ({len(args.models)})   : {args.models}")
    print(f" Seeds ({len(args.seeds)})    : {args.seeds}")
    print(f" Eval Only / Overwrite: eval_only={args.eval_only}, overwrite={args.overwrite}")
    print(f" Total Planned Runs  : {len(tasks)}")
    print(f" Result DB           : {get_db_path(results_dir)}")
    print(f" Result tables       : {list(RESULT_TABLES)}")
    print(f" Prediction cache    : {prediction_cache.CACHE_ROOT}")
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

            # Skip if task is already completed and saved in results (unless reset/eval_only/overwrite)
            skip_completed = not (args.reset or args.eval_only or args.overwrite)
            if skip_completed and is_experiment_already_completed(results_dir, model_name, ds, seed):
                print(f"[SKIP] DATA={ds} | MODEL={model_name.upper()} | SEED={seed} already completed in results. Skipping inference.")
                successful_runs += 1
                continue

            print(f"\n>>> [Dataset: {ds} | Task {idx}/{len(ds_tasks)}] MODEL={model_name.upper()} | SEED={seed}")
            print("-" * 80)

            try:
                # A. Load cached dataset
                train_df, val_df, test_df, features = get_cached_dataset(data_path, args.drop_failure_day, model_name)

                is_sequence_model = model_name in ['lstm', 'gru']
                window_size = config.WINDOW_SIZE if is_sequence_model else 1

                # B. Checkpoint reload or model training
                cached_model = load_checkpoint(model_name, imbalance, seed, config.TARGET_LEAD_TIME, data_path, input_dim=len(features), features=features, window_size=window_size if is_sequence_model else None)

                if cached_model is None and args.eval_only:
                    print(f"[ERROR] No saved checkpoint found for DATA={ds} | MODEL={model_name.upper()} | SEED={seed}. Skipping (Eval-Only Mode).")
                    failed_runs += 1
                    continue

                if cached_model is None:
                    if is_sequence_model:
                        X_val_seq, y_val_seq = build_sequences(val_df, features, window_size=window_size, lead_time=config.TARGET_LEAD_TIME)
                        X_train_seq, y_train_seq = build_sequences(train_df, features, window_size=window_size, lead_time=config.TARGET_LEAD_TIME)
                    else:
                        y_val = create_binary_target(val_df, lead_time=config.TARGET_LEAD_TIME)
                        X_val_2d = val_df[features].values
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
                            model = train_lgbm_model(X_tr_proc, y_tr_proc, X_val_2d, y_val, seed=seed, use_gpu=config.LGBM_USE_GPU)
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

                splits = {}
                for split_name, split_df in (('val', val_df), ('test', test_df)):
                    print(f"Performing single sequential inference on {split_name.capitalize()} set...")
                    raw = evaluator.get_raw_predictions(split_df)
                    splits[split_name] = {
                        'raw': raw,
                        'probs': np.concatenate([d['preds'] for d in raw]),
                        'y_true': np.concatenate([d['y_true'] for d in raw]),
                    }

                # Cache the sufficient statistics so re-deriving thresholds, metrics or a
                # different horizon H never needs inference again.
                for split_name, s in splits.items():
                    written = prediction_cache.save_all_layers(
                        s['raw'], s['y_true'], s['probs'], THRESHOLD_GRID,
                        ds, model_name, seed, split_name)
                    print(f"[Cache] {split_name}: " + ", ".join(sorted(written)))

                # D1. Row-unit threshold: max Recall subject to row FAR <= MAX_FAR
                th_row, val_row_recall = find_best_threshold_row_level(
                    splits['val']['y_true'], splits['val']['probs'], max_far=config.MAX_FAR)
                val_row_at_th_row = calculate_row_level_metrics(
                    splits['val']['y_true'], splits['val']['probs'], threshold=th_row)
                print(f"[Row-Level Search] Threshold {th_row:.4f} "
                      f"(val Recall {val_row_recall:.4%} @ val FAR {val_row_at_th_row['far']:.4%})")

                # D2. Disk-unit threshold: max Recall subject to disk FAR <= MAX_DISK_FAR
                th_disk, val_disk_recall = evaluator.find_best_threshold_disk_level(
                    splits['val']['raw'], max_far=config.MAX_DISK_FAR)
                val_disk_at_th_disk, _ = evaluator.evaluate_proposed_level(
                    splits['val']['raw'], threshold=th_disk)
                print(f"[Disk-Level Search] Threshold {th_disk:.4f} "
                      f"(val Recall {val_disk_recall:.4%} @ val FAR {val_disk_at_th_disk['far']:.4%})")

                thresholds = {'row_opt': th_row, 'disk_opt': th_disk}

                # Evaluate BOTH units at BOTH thresholds on BOTH splits.
                row_level_rows, disk_level_rows = [], []
                for split_name, s in splits.items():
                    for kind, thr in thresholds.items():
                        rm = calculate_row_level_metrics(s['y_true'], s['probs'], threshold=thr)
                        dm, _ = evaluator.evaluate_proposed_level(s['raw'], threshold=thr)
                        common = {
                            'dataset': os.path.basename(ds),
                            'model': model_name.upper(),
                            'seed': seed,
                            'split': split_name,
                            'threshold_kind': kind,
                            'threshold': round(thr, 4),
                        }
                        row_level_rows.append({
                            **common,
                            'precision': round(rm['precision'], 6),
                            'recall': round(rm['recall'], 6),
                            'f1': round(rm['f1'], 6),
                            'far': round(rm['far'], 6),
                            'auroc': round(rm['auroc'], 6),
                            'tp': rm['tp'], 'fp': rm['fp'], 'fn': rm['fn'], 'tn': rm['tn'],
                        })
                        disk_level_rows.append({
                            **common,
                            'N_on_time': dm['N_ontime'],
                            'N_early': dm['N_early'],
                            'N_missed': dm['N_missed'],
                            'N_censored_early': dm['N_cens_early'],
                            'N_censored_no_alarm': dm['N_cens_no_alarm'],
                            'N_failed': dm['N_failed'],
                            'N_censored': dm['N_censored'],
                            'N_alarmed': dm['N_alarmed'],
                            'precision': round(dm['precision'], 6),
                            'recall': round(dm['recall'], 6),
                            'f1': round(dm['f1'], 6),
                            'far': round(dm['far'], 6),
                            'on_time_share': round(dm['on_time_share'], 6),
                            'mean_lead_time': round(dm['mean_lead_time'], 2),
                            'median_lead_time': round(dm['median_lead_time'], 2),
                            'std_lead_time': round(dm['std_lead_time'], 2),
                        })

                run_row = {
                    'dataset': os.path.basename(ds),
                    'model': model_name.upper(),
                    'seed': seed,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'lead_time_H': config.TARGET_LEAD_TIME,
                    'n_features': len(features),
                    'threshold_row_opt': round(th_row, 4),
                    'threshold_disk_opt': round(th_disk, 4),
                    'max_far_row': config.MAX_FAR,
                    'max_far_disk': config.MAX_DISK_FAR,
                    'val_far_row_opt': round(val_row_at_th_row['far'], 6),
                    'val_far_disk_opt': round(val_disk_at_th_disk['far'], 6),
                    # Whether each constraint was actually satisfiable on validation.
                    'row_constraint_met': int(val_row_at_th_row['far'] <= config.MAX_FAR),
                    'disk_constraint_met': int(val_disk_at_th_disk['far'] <= config.MAX_DISK_FAR),
                }

                save_run_results(results_dir, run_row, row_level_rows, disk_level_rows)

                successful_runs += 1

            except Exception:
                failed_runs += 1
                print(f"\n[ERROR] Task failed (DATA={ds}, MODEL={model_name}, SEED={seed}).")
                if not args.keep_going:
                    # Fail fast by default: a partially-populated results table is
                    # worse than a stopped batch, because the missing rows are
                    # invisible in the exported CSV.
                    print("[ERROR] Aborting the batch. Re-run with --keep-going to skip failures instead.")
                    raise
                import traceback
                traceback.print_exc()

            finally:
                # Memory Cleanup
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(" UNIFIED THRESHOLD EXPERIMENT RUNNER COMPLETED!")
    print(f" Total Elapsed Time   : {elapsed / 60:.2f} minutes")
    print(f" Successful Runs      : {successful_runs} / {len(tasks)}")
    print(f" Failed Runs          : {failed_runs} / {len(tasks)}")
    print(f" Result DB            : {get_db_path(results_dir)}")
    for table in RESULT_TABLES:
        print(f"   {table:12s} -> {csv_export_path(results_dir, table)}")
    print("=" * 80 + "\n")

    if failed_runs > 0:
        # Reached only under --keep-going; exit non-zero so an incomplete batch
        # cannot be mistaken for a successful one by a caller or a shell script.
        sys.exit(1)


if __name__ == "__main__":
    run_unified_threshold_experiments()
