import os
import sys
import argparse
import subprocess
import time

import pandas as pd

# ------------------------------------------------------------------------------
# 1. 기본 실험 설정 (시드 42~46 반복, 총 5개 시드)
# ------------------------------------------------------------------------------
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_DATASETS = [
    'ST12000NM0007',
    'HGST_20HUH721212ALN604',
    'TOSHIBA_20MG07ACA14TA'
]
DEFAULT_MODELS = ['lgbm', 'xgb', 'lstm', 'gru']
DEFAULT_IMBALANCE_STRATEGIES = ['none']


def is_compatible(model: str, strategy: str) -> bool:
    """모델 architecture와 불균형 처리 방식 간 호환성 검증"""
    if strategy == 'focal_loss' and model not in ['mlp', 'lstm', 'gru']:
        return False
    if strategy == 'smote' and model in ['lstm', 'gru']:
        return False
    return True


def get_completed_tasks(project_dir: str) -> set:
    """마스터 CSV에서 오늘(최신 파이프라인 v3) 실행 완료된 (dataset_base, model, strategy, seed) 세트 파싱"""
    master_path = os.path.join(project_dir, "results", "master_experiment_results.csv")
    completed = set()
    if not os.path.exists(master_path):
        return completed
    try:
        df = pd.read_csv(master_path)
        req_cols = ['Timestamp', 'Model', '데이터', '불균형 처리', 'Seed']
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


def run_seed_experiments():
    parser = argparse.ArgumentParser(description="Multi-Seed Automated Batch Experiment Framework (Seeds 42~46)")
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS, help='List of random seeds (default: 42 43 44 45 46)')
    parser.add_argument('--datasets', type=str, nargs='+', default=DEFAULT_DATASETS, help='List of target HDD dataset names')
    parser.add_argument('--models', type=str, nargs='+', default=DEFAULT_MODELS, help='List of model architectures')
    parser.add_argument('--imbalance', type=str, nargs='+', default=DEFAULT_IMBALANCE_STRATEGIES, help='List of imbalance strategies')
    parser.add_argument('--dry-run', action='store_true', help='Print planned tasks without executing')
    parser.add_argument('--force-rerun', action='store_true', help='Force re-run even if task is already in master CSV')
    args = parser.parse_args()

    python_executable = sys.executable
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(exp_dir)
    run_exp_script = os.path.join(exp_dir, "run_experiment.py")

    completed_tasks = set() if args.force_rerun else get_completed_tasks(project_dir)

    # 1. 전체 실행 계획 수립 (데이터셋 최외곽 루프로 배치하여 캐시 재활용 극대화)
    total_planned = 0
    skipped_count = 0
    tasks = []

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
    print(" MULTI-SEED REPEATED EXPERIMENT RUNNER")
    print(f" Execution Order                : DATASET -> MODEL -> IMBALANCE -> SEED (Dataset-cached)")
    print(f" Target Datasets ({len(args.datasets)})          : {args.datasets}")
    print(f" Target Models ({len(args.models)})            : {args.models}")
    print(f" Imbalance Strategies ({len(args.imbalance)})   : {args.imbalance}")
    print(f" Target Seeds ({len(args.seeds)})             : {args.seeds}")
    print(f" Total Planned Model Runs       : {len(tasks)}")
    print(f" Completed Runs in Master CSV   : {len(completed_tasks)}")
    print(f" Skipped Incompatible Tasks     : {skipped_count}")
    print(f" Threshold Policy               : ONE threshold per (model, dataset, seed) from val-set")
    print(f" Lead Time Policy               : Unrestricted failure lead time (days_to_failure >= 0)")
    print(f" Evaluations per run            : Row-level / Disk-level rolling (unified threshold)")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN MODE] Printing planned execution tasks:")
        for idx, t in enumerate(tasks, 1):
            ds_base = os.path.basename(t['dataset'].rstrip('/\\'))
            task_key = (ds_base, t['model'].lower(), t['strategy'].lower(), t['seed'])
            status = "[ALREADY DONE]" if task_key in completed_tasks else "[WILL RUN]"
            print(f"  {idx:02d}. {status} DATA={t['dataset']} | MODEL={t['model'].upper()} | IMBALANCE={t['strategy'].upper()} | SEED={t['seed']}")
        print("\nDry run completed. No code was executed.")
        return

    start_time = time.time()
    successful_runs = 0
    failed_runs = 0
    skipped_completed = 0

    # 2. 데이터셋 그룹별 실험 수행
    unique_datasets = list(dict.fromkeys([t['dataset'] for t in tasks]))

    for ds in unique_datasets:
        ds_tasks = [t for t in tasks if t['dataset'] == ds]
        print("\n" + "#" * 80)
        print(f" [START DATASET BATCH: DATASET = {ds}] ({len(ds_tasks)} experiments planned)")
        print("#" * 80)

        for idx, t in enumerate(ds_tasks, 1):
            data_path = t['data_path']
            model = t['model']
            strat = t['strategy']
            seed = t['seed']

            ds_base = os.path.basename(ds.rstrip('/\\'))
            task_key = (ds_base, model.lower(), strat.lower(), seed)

            if task_key in completed_tasks:
                print(f"\n[SKIP COMPLETED] Task {idx}/{len(ds_tasks)}: DATA={ds} | MODEL={model.upper()} | SEED={seed} already present in master CSV. Skipping.")
                skipped_completed += 1
                continue

            print(f"\n>>> [Dataset {ds} | Task {idx}/{len(ds_tasks)}] MODEL={model.upper()} | IMBALANCE={strat.upper()} | SEED={seed}")
            print("-" * 80)

            # Fast In-Process Execution (reuses _DATASET_CACHE in RAM)
            orig_argv = sys.argv
            try:
                sys.argv = [
                    'run_experiment.py',
                    '--data', data_path,
                    '--model', model,
                    '--imbalance', strat,
                    '--seed', str(seed)
                ]
                import run_experiment
                run_experiment.main()
                successful_runs += 1
                print(f"[Success] DATA={ds} | MODEL={model.upper()} | SEED={seed}")
            except Exception as e:
                failed_runs += 1
                print(f"[Error] Execution failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                sys.argv = orig_argv

        print(f"\n[Dataset {ds}] All {len(ds_tasks)} experiments completed. Results saved to master_experiment_results.csv")

    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\n" + "=" * 80)
    print(" ALL MULTI-SEED EXPERIMENTS COMPLETED!")
    print(f" Total Time Elapsed          : {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f" Newly Executed Successful   : {successful_runs} / {len(tasks)}")
    print(f" Skipped (Already Completed) : {skipped_completed} / {len(tasks)}")
    print(f" Failed Runs                 : {failed_runs} / {len(tasks)}")
    print(" Check results in            : results/master_experiment_results.csv")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_seed_experiments()
