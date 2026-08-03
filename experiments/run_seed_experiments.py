import os
import sys
import argparse
import subprocess
import time

# ------------------------------------------------------------------------------
# 1. 기본 실험 설정 (시드 43~45 반복)
# ------------------------------------------------------------------------------
DEFAULT_SEEDS = [43, 44, 45, 46]
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


def run_seed_experiments():
    parser = argparse.ArgumentParser(description="Multi-Seed Automated Batch Experiment Framework (Seeds 43~45)")
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS, help='List of random seeds (default: 43 44 45 46)')
    parser.add_argument('--datasets', type=str, nargs='+', default=DEFAULT_DATASETS, help='List of target HDD dataset names')
    parser.add_argument('--models', type=str, nargs='+', default=DEFAULT_MODELS, help='List of model architectures')
    parser.add_argument('--imbalance', type=str, nargs='+', default=DEFAULT_IMBALANCE_STRATEGIES, help='List of imbalance strategies')
    parser.add_argument('--dry-run', action='store_true', help='Print planned tasks without executing')
    args = parser.parse_args()

    python_executable = sys.executable
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(exp_dir)
    run_exp_script = os.path.join(exp_dir, "run_experiment.py")
    # NOTE: run_single_observation_eval.py is NOT called here.
    # Single-obs evaluation is computed inside run_experiment.py using the same
    # val-set threshold (opt_threshold) as row-level and disk-level evaluations.
    # This guarantees ONE unified threshold per (model, dataset, seed) across all 3 methods.

    # 1. 전체 실행 계획 수립
    total_planned = 0
    skipped_count = 0
    tasks = []

    for seed in args.seeds:
        for dataset in args.datasets:
            if os.path.isabs(dataset) or os.path.exists(dataset):
                data_path = dataset
            else:
                data_path = os.path.join(project_dir, "data", "splitted", dataset)

            for model in args.models:
                for strategy in args.imbalance:
                    if is_compatible(model, strategy):
                        tasks.append({
                            'seed': seed,
                            'dataset': dataset,
                            'data_path': data_path,
                            'model': model,
                            'strategy': strategy
                        })
                    else:
                        skipped_count += 1

    print("=" * 80)
    print(" MULTI-SEED REPEATED EXPERIMENT RUNNER")
    print(f" Target Seeds ({len(args.seeds)})             : {args.seeds}")
    print(f" Target Datasets ({len(args.datasets)})          : {args.datasets}")
    print(f" Target Models ({len(args.models)})            : {args.models}")
    print(f" Imbalance Strategies ({len(args.imbalance)})   : {args.imbalance}")
    print(f" Total Planned Model Runs       : {len(tasks)}")
    print(f" Skipped Incompatible Tasks     : {skipped_count}")
    print(f" Threshold Policy               : ONE threshold per (model, dataset, seed) from val-set")
    print(f" Evaluations per run            : Row-level / Disk-level rolling / Single-obs (unified threshold)")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN MODE] Printing planned execution tasks:")
        for idx, t in enumerate(tasks, 1):
            print(f"  {idx:02d}. SEED={t['seed']} | DATA={t['dataset']} | MODEL={t['model'].upper()} | IMBALANCE={t['strategy'].upper()}")
        print("\nDry run completed. No code was executed.")
        return

    start_time = time.time()
    successful_runs = 0
    failed_runs = 0

    # 2. Seed별 실험 수행
    for seed in args.seeds:
        seed_tasks = [t for t in tasks if t['seed'] == seed]
        print("\n" + "#" * 80)
        print(f" [START SEED ITERATION: SEED = {seed}] ({len(seed_tasks)} experiments planned)")
        print("#" * 80)

        for idx, t in enumerate(seed_tasks, 1):
            ds = t['dataset']
            data_path = t['data_path']
            model = t['model']
            strat = t['strategy']

            print(f"\n>>> [Seed {seed} | Task {idx}/{len(seed_tasks)}] DATA={ds} | MODEL={model.upper()} | IMBALANCE={strat.upper()}")
            print("-" * 80)

            cmd = [
                python_executable, run_exp_script,
                '--data', data_path,
                '--model', model,
                '--imbalance', strat,
                '--seed', str(seed)
            ]

            try:
                subprocess.run(cmd, check=True)
                successful_runs += 1
                print(f"[Success] Seed={seed} | DATA={ds} | MODEL={model.upper()}")
            except subprocess.CalledProcessError as e:
                failed_runs += 1
                print(f"[Failed] (exit code {e.returncode}): Seed={seed} | DATA={ds} | MODEL={model.upper()}")
            except Exception as e:
                failed_runs += 1
                print(f"[Error] Unexpected Error: {e}")

        print(f"\n[Seed {seed}] All {len(seed_tasks)} experiments done. Results (all 3 evals) saved to master_experiment_results.csv")

    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\n" + "=" * 80)
    print(" ALL MULTI-SEED EXPERIMENTS COMPLETED!")
    print(f" Total Time Elapsed : {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f" Successful Runs    : {successful_runs} / {len(tasks)}")
    print(f" Failed Runs        : {failed_runs} / {len(tasks)}")
    print(" Check results in   : results/master_experiment_results.csv")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_seed_experiments()
