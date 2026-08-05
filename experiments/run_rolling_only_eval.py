import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
try:
    import torch
except ImportError:
    torch = None

import config
from data_loader import load_dataset, build_sequences
from imbalance import validate_config_compatibility, EasyEnsembleHandler, EasyEnsembleModelWrapper
from evaluator import RollingEvaluator, _normalize_key_val
from checkpoint_utils import load_checkpoint, save_checkpoint
from models import (
    train_rf_model,
    train_lgbm_model,
    train_xgb_model,
    train_mlp_model,
    train_lstm_model,
    train_gru_model
)

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_DATASETS = [
    'ST12000NM0007',
    'HGST_20HUH721212ALN604',
    'TOSHIBA_20MG07ACA14TA'
]
DEFAULT_MODELS = ['lgbm', 'xgb', 'lstm', 'gru']
DEFAULT_IMBALANCE_STRATEGIES = ['none']

MASTER_ROLLING_CSV_COLS = [
    'Timestamp', 'Model', '데이터', '불균형 처리', 'Seed', 'Pipeline_Version',
    'Disk_Threshold', 'Disk_TP', 'Disk_FP', 'Disk_FN', 'Disk_TN',
    'Disk_Precision', 'Disk_Recall', 'Disk_F1', 'Disk_FAR (%)',
    'Hit_Median_Lead_Time', 'Hit_Mean_Lead_Time',
    'All_Alarm_Failed_Median_Lead_Time', 'All_Alarm_Failed_Mean_Lead_Time',
    'EDR@15 (%)'
]


def is_compatible(model: str, strategy: str) -> bool:
    if strategy == 'focal_loss' and model not in ['mlp', 'lstm', 'gru']:
        return False
    if strategy == 'smote' and model in ['lstm', 'gru']:
        return False
    return True


def analyze_rolling_report(report_df: pd.DataFrame, threshold: float) -> dict:
    if len(report_df) == 0:
        return {}
    total_disks = len(report_df)
    failed_disks = int(report_df['has_failed'].sum())
    censored_disks = int(total_disks - failed_disks)

    hits = int(report_df['is_hit'].sum())
    false_alarms = int(report_df['is_false_alarm'].sum())
    misses = int(report_df['is_miss'].sum())
    correct_rejections = int(report_df['is_correct_rejection'].sum())

    recall = float(hits / failed_disks) if failed_disks > 0 else 0.0
    far = float(false_alarms / censored_disks) if censored_disks > 0 else 0.0
    precision = float(hits / (hits + false_alarms)) if (hits + false_alarms) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # 1. Lead time calculation among HIT disks only (days_to_failure >= 0)
    hit_rows = report_df[report_df['is_hit'] == 1]
    hit_lead_times = hit_rows['days_to_failure_at_alarm'].dropna().values
    hit_mean_lt = float(np.mean(hit_lead_times)) if len(hit_lead_times) > 0 else 0.0
    hit_median_lt = float(np.median(hit_lead_times)) if len(hit_lead_times) > 0 else 0.0

    # 2. Lead time calculation among ALL failed HDDs that triggered an alarm (Hit or Miss/Late)
    all_alarm_failed_rows = report_df[(report_df['has_failed'] == 1) & (report_df['alarm_triggered'] == 1)]
    all_alarm_lead_times = all_alarm_failed_rows['days_to_failure_at_alarm'].dropna().values
    all_alarm_mean_lt = float(np.mean(all_alarm_lead_times)) if len(all_alarm_lead_times) > 0 else 0.0
    all_alarm_median_lt = float(np.median(all_alarm_lead_times)) if len(all_alarm_lead_times) > 0 else 0.0

    # EDR@15 calculation
    edr_15_count = sum(1 for lt in hit_lead_times if lt >= 15)
    edr_15 = float(edr_15_count / failed_disks) if failed_disks > 0 else 0.0

    return {
        'threshold': threshold,
        'total_disks': total_disks,
        'failed_disks': failed_disks,
        'censored_disks': censored_disks,
        'tp': hits,
        'fp': false_alarms,
        'fn': misses,
        'tn': correct_rejections,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'far': far,
        'hit_median_lt': hit_median_lt,
        'hit_mean_lt': hit_mean_lt,
        'all_alarm_median_lt': all_alarm_median_lt,
        'all_alarm_mean_lt': all_alarm_mean_lt,
        'edr_15': edr_15
    }


def append_rolling_master_result(row_dict: dict, master_csv_path: str):
    os.makedirs(os.path.dirname(master_csv_path), exist_ok=True)
    df_row = pd.DataFrame([row_dict])
    
    if os.path.exists(master_csv_path):
        try:
            df_master = pd.read_csv(master_csv_path, encoding='utf-8-sig').dropna(how='all')
            key_cols = ['Model', '데이터', '불균형 처리', 'Seed']
            if all(c in df_master.columns for c in key_cols):
                new_key = df_row.iloc[0]
                mask = pd.Series(True, index=df_master.index)
                for c in key_cols:
                    target_norm = _normalize_key_val(new_key[c])
                    col_norm = df_master[c].apply(_normalize_key_val)
                    mask &= (col_norm == target_norm)
                df_master = df_master[~mask]
            df_final = pd.concat([df_master, df_row], ignore_index=True)
        except Exception:
            df_final = df_row
    else:
        df_final = df_row

    df_final.to_csv(master_csv_path, index=False, encoding='utf-8-sig')
    print(f"[Master Log Saved] Append result -> {master_csv_path}")


def get_completed_tasks(master_csv_path: str) -> set:
    completed = set()
    if not os.path.exists(master_csv_path):
        return completed
    try:
        df = pd.read_csv(master_csv_path, encoding='utf-8-sig')
        req_cols = ['Model', '데이터', '불균형 처리', 'Seed']
        if not all(col in df.columns for col in req_cols):
            return completed
        for _, row in df.iterrows():
            m = str(row['Model']).strip().lower()
            ds = os.path.basename(str(row['데이터']).strip().rstrip('/\\'))
            strat = str(row['불균형 처리']).strip().lower()
            try:
                seed = int(row['Seed'])
                completed.add((ds, m, strat, seed))
            except (ValueError, TypeError):
                continue
    except Exception as e:
        print(f"[Warning] Could not read master CSV for task skipping: {e}")
    return completed


def run_rolling_only_experiments():
    parser = argparse.ArgumentParser(description="Rolling-Only Evaluation Pipeline with Disk-Level Threshold Optimization")
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS, help='List of random seeds')
    parser.add_argument('--datasets', type=str, nargs='+', default=DEFAULT_DATASETS, help='List of target HDD dataset names')
    parser.add_argument('--models', type=str, nargs='+', default=DEFAULT_MODELS, help='List of model architectures')
    parser.add_argument('--imbalance', type=str, nargs='+', default=DEFAULT_IMBALANCE_STRATEGIES, help='List of imbalance strategies')
    parser.add_argument('--dry-run', action='store_true', help='Print planned tasks without executing')
    parser.add_argument('--force-rerun', action='store_true', help='Force re-run even if recorded in master CSV')
    parser.add_argument('--sample-size', type=int, default=config.SAMPLE_SIZE, help='Limit val/test disks for fast testing')
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    master_csv_path = os.path.join(project_dir, "results", "master_rolling_only_results.csv")
    completed_tasks = set() if args.force_rerun else get_completed_tasks(master_csv_path)

    tasks = []
    skipped_count = 0
    for dataset in args.datasets:
        if os.path.isabs(dataset) or os.path.exists(dataset):
            data_path = dataset
        else:
            data_path = os.path.join(project_dir, "data", "splitted", dataset)

        for model in args.models:
            for strategy in args.imbalance:
                for seed in args.seeds:
                    if is_compatible(model, strategy):
                        tasks.append({
                            'dataset': dataset,
                            'data_path': data_path,
                            'model': model,
                            'strategy': strategy,
                            'seed': seed
                        })
                    else:
                        skipped_count += 1

    print("=" * 80)
    print(" DISK-LEVEL ROLLING-ONLY EVALUATION RUNNER")
    print(f" Execution Order                : DATASET -> MODEL -> IMBALANCE -> SEED")
    print(f" Threshold Optimization Policy  : DISK-LEVEL Recall Maximization under Disk FAR <= 1% (Val Set)")
    print(f" Lead Time Calculation Policy   : Including Missed Failure Disks as 0.0 Days")
    print(f" Target Datasets ({len(args.datasets)})          : {args.datasets}")
    print(f" Target Models ({len(args.models)})            : {args.models}")
    print(f" Target Seeds ({len(args.seeds)})             : {args.seeds}")
    print(f" Total Planned Model Runs       : {len(tasks)}")
    print(f" Completed Runs in Master CSV   : {len(completed_tasks)}")
    print(f" Master Results File            : {master_csv_path}")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN MODE] Planned Tasks:")
        for idx, t in enumerate(tasks, 1):
            ds_base = os.path.basename(t['dataset'].rstrip('/\\'))
            task_key = (ds_base, t['model'].lower(), t['strategy'].lower(), t['seed'])
            status = "[ALREADY DONE]" if task_key in completed_tasks else "[WILL RUN]"
            print(f"  {idx:02d}. {status} DATA={t['dataset']} | MODEL={t['model'].upper()} | SEED={t['seed']}")
        print("\nDry run completed. No code executed.")
        return

    start_time = time.time()
    successful_runs = 0
    skipped_runs = 0
    failed_runs = 0

    unique_datasets = list(dict.fromkeys([t['dataset'] for t in tasks]))

    for ds in unique_datasets:
        ds_tasks = [t for t in tasks if t['dataset'] == ds]
        print("\n" + "#" * 80)
        print(f" [START DATASET BATCH: DATASET = {ds}] ({len(ds_tasks)} experiments planned)")
        print("#" * 80)

        for idx, t in enumerate(ds_tasks, 1):
            data_path = t['data_path']
            model_name = t['model']
            strat = t['strategy']
            seed = t['seed']

            ds_base = os.path.basename(ds.rstrip('/\\'))
            task_key = (ds_base, model_name.lower(), strat.lower(), seed)

            if task_key in completed_tasks:
                print(f"\n[SKIP COMPLETED] Task {idx}/{len(ds_tasks)}: DATA={ds} | MODEL={model_name.upper()} | SEED={seed} (Found in master_rolling_only_results.csv)")
                skipped_runs += 1
                continue

            print(f"\n>>> [Dataset {ds} | Task {idx}/{len(ds_tasks)}] MODEL={model_name.upper()} | IMBALANCE={strat.upper()} | SEED={seed}")
            print("-" * 80)

            try:
                # 1. Load Dataset
                train_df, val_df, test_df, features = load_dataset(data_path, drop_failure_day_in_train=config.DROP_FAILURE_DAY_IN_TRAIN, model=model_name)
                is_sequence_model = model_name in ['lstm', 'gru']

                # 2. Check / Load / Train Checkpoint
                is_cost_sensitive = (strat == 'class_weight') or getattr(config, 'USE_CLASS_WEIGHT', False)
                use_focal_loss = (strat == 'focal_loss')
                ckpt_tag = f"cw{int(is_cost_sensitive)}_focal{int(use_focal_loss)}"
                ckpt_window_size = config.WINDOW_SIZE if is_sequence_model else None

                cached_model = load_checkpoint(
                    model_name, strat, seed, config.TARGET_LEAD_TIME, data_path,
                    input_dim=len(features), extra_tag=ckpt_tag, features=features, window_size=ckpt_window_size
                )

                if cached_model is not None:
                    model = cached_model
                    model_type = 'ensemble' if strat == 'easyensemble' else ('pytorch_class' if is_sequence_model or model_name == 'mlp' else model_name)
                else:
                    print(f"[Training] Checkpoint not found for {model_name} (seed {seed}). Training new model...")
                    if is_sequence_model:
                        window_size = config.WINDOW_SIZE
                        X_train_seq, y_train_seq = build_sequences(train_df, features, window_size=window_size, lead_time=config.TARGET_LEAD_TIME)
                        X_val_seq, y_val_seq = build_sequences(val_df, features, window_size=window_size, lead_time=config.TARGET_LEAD_TIME)
                        if model_name == 'lstm':
                            model = train_lstm_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, seed=seed, is_cost_sensitive=is_cost_sensitive, use_focal_loss=use_focal_loss)
                        else:
                            model = train_gru_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, seed=seed, is_cost_sensitive=is_cost_sensitive, use_focal_loss=use_focal_loss)
                        model_type = 'pytorch_class'
                    else:
                        y_train = (train_df['RUL'] <= config.TARGET_LEAD_TIME).astype(int).values
                        y_val = (val_df['RUL'] <= config.TARGET_LEAD_TIME).astype(int).values
                        X_train_2d = train_df[features].values
                        X_val_2d = val_df[features].values
                        if model_name == 'rf':
                            model = train_rf_model(X_train_2d, y_train, X_val_2d, y_val, seed=seed, is_cost_sensitive=is_cost_sensitive)
                            model_type = 'rf'
                        elif model_name == 'lgbm':
                            model = train_lgbm_model(X_train_2d, y_train, X_val_2d, y_val, seed=seed, is_cost_sensitive=is_cost_sensitive, use_gpu=config.USE_GPU)
                            model_type = 'lgbm'
                        elif model_name == 'xgb':
                            model = train_xgb_model(X_train_2d, y_train, X_val_2d, y_val, seed=seed, is_cost_sensitive=is_cost_sensitive, use_gpu=config.USE_GPU)
                            model_type = 'xgb'
                        elif model_name == 'mlp':
                            model = train_mlp_model(X_train_2d, y_train, X_val_2d, y_val, seed=seed, is_cost_sensitive=is_cost_sensitive, use_focal_loss=use_focal_loss)
                            model_type = 'pytorch_class'

                    save_checkpoint(model, model_name, strat, seed, config.TARGET_LEAD_TIME, data_path, extra_tag=ckpt_tag, features=features, window_size=ckpt_window_size)

                # 3. Instantiate Rolling Evaluator
                evaluator = RollingEvaluator(
                    model=model,
                    features=features,
                    window_size=config.WINDOW_SIZE if is_sequence_model else 1,
                    device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
                    model_type=model_type,
                    seed=seed
                )

                # 4. Find Disk-Level Optimal Threshold on Validation Set
                print(f"\n[Val Threshold Search] Finding Disk-Level optimal decision threshold (FAR <= 1%)...")
                val_raw_preds = evaluator.get_raw_predictions(val_df, sample_size=args.sample_size)
                opt_threshold, val_disk_recall = evaluator.find_best_threshold(val_raw_preds, max_far=getattr(config, 'MAX_FAR', 0.01))
                print(f"[Optimal Disk Threshold] Found on Val Set: {opt_threshold:.4f} (Val Disk Recall @ FAR <= 1%: {val_disk_recall:.4%})")

                # 5. Evaluate Test Set using Disk-Level Optimal Threshold
                print(f"\n[Test Rolling Evaluation] Evaluating Test Set with threshold = {opt_threshold:.4f}...")
                test_raw_preds = evaluator.get_raw_predictions(test_df, sample_size=args.sample_size)
                test_report_df = evaluator.build_report_from_raw_predictions(test_raw_preds, threshold=opt_threshold)
                metrics = analyze_rolling_report(test_report_df, threshold=opt_threshold)

                # Print Brief Result Summary
                print(f"\n>>> RESULT SUMMARY: MODEL={model_name.upper()} | DATA={ds_base} | SEED={seed}")
                print(f"    Threshold                 : {opt_threshold:.4f}")
                print(f"    Disk TP/FP/FN/TN          : {metrics['tp']:,} / {metrics['fp']:,} / {metrics['fn']:,} / {metrics['tn']:,}")
                print(f"    Disk Precision / Recall / F1: {metrics['precision']:.4f} / {metrics['recall']:.4f} / {metrics['f1']:.4f}")
                print(f"    Disk FAR (%)              : {metrics['far']*100:.2f}%")
                print(f"    Hit Median Lead Time              : {metrics['hit_median_lt']:.2f} days (Mean: {metrics['hit_mean_lt']:.2f}d)")
                print(f"    All Alarm Failed Median Lead Time : {metrics['all_alarm_median_lt']:.2f} days (Mean: {metrics['all_alarm_mean_lt']:.2f}d)")
                print(f"    EDR@15                            : {metrics['edr_15']*100:.2f}%")

                # 6. Save to Master Rolling CSV (If not sample_size test mode)
                if args.sample_size is None:
                    row_data = {
                        'Timestamp': time.strftime("%Y-%m-%d %H:%M"),
                        'Model': model_name.upper(),
                        '데이터': ds_base,
                        '불균형 처리': strat,
                        'Seed': seed,
                        'Pipeline_Version': getattr(config, 'PIPELINE_VERSION', 'v3'),
                        'Disk_Threshold': f"{metrics['threshold']:.4f}",
                        'Disk_TP': metrics['tp'],
                        'Disk_FP': metrics['fp'],
                        'Disk_FN': metrics['fn'],
                        'Disk_TN': metrics['tn'],
                        'Disk_Precision': f"{metrics['precision']:.4f}",
                        'Disk_Recall': f"{metrics['recall']:.4f}",
                        'Disk_F1': f"{metrics['f1']:.4f}",
                        'Disk_FAR (%)': f"{metrics['far']*100:.2f}",
                        'Hit_Median_Lead_Time': f"{metrics['hit_median_lt']:.2f}",
                        'Hit_Mean_Lead_Time': f"{metrics['hit_mean_lt']:.2f}",
                        'All_Alarm_Failed_Median_Lead_Time': f"{metrics['all_alarm_median_lt']:.2f}",
                        'All_Alarm_Failed_Mean_Lead_Time': f"{metrics['all_alarm_mean_lt']:.2f}",
                        'EDR@15 (%)': f"{metrics['edr_15']*100:.2f}"
                    }
                    append_rolling_master_result(row_data, master_csv_path)

                successful_runs += 1

            except Exception as e:
                failed_runs += 1
                print(f"[Error] Execution failed for {model_name} on {ds} (seed {seed}): {e}")
                import traceback
                traceback.print_exc()

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\n" + "=" * 80)
    print(" ROLLING-ONLY EVALUATION PIPELINE COMPLETED!")
    print(f" Total Time Elapsed          : {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f" Successfully Executed       : {successful_runs} / {len(tasks)}")
    print(f" Skipped (Already Completed) : {skipped_runs} / {len(tasks)}")
    print(f" Failed Runs                 : {failed_runs} / {len(tasks)}")
    print(f" Results Saved to            : {master_csv_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_rolling_only_experiments()
