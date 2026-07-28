import os
import sys
import subprocess
import config

# ------------------------------------------------------------------------------
# 1. 순회 대상 설정
# ------------------------------------------------------------------------------
# 분류 모델 6종 전체
MODELS = ['lgbm', 'xgb', 'mlp', 'lstm', 'gru']

# 불균형 처리 방식 ('oversampling', 'adasyn' 제외)
IMBALANCE_STRATEGIES = [
    'none',
    'class_weight',
    'undersampling',
    'smote',
    'easyensemble',
    'focal_loss'
]

# PyTorch 전용 Focal Loss 비호환 체크 함수
def is_compatible(model: str, strategy: str) -> bool:
    # focal_loss는 딥러닝/신경망 모델(mlp, lstm, gru)만 지원
    if strategy == 'focal_loss' and model not in ['mlp', 'lstm', 'gru']:
        return False
    # smote는 3D 시퀀스 모델(lstm, gru)에서 메모리 문제로 지원 불가
    if strategy == 'smote' and model in ['lstm', 'gru']:
        return False
    return True

def run_grid_search():
    python_executable = sys.executable
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_experiment.py")

    total_runs = 0
    skipped_runs = 0
    
    # 실행 계획 계산
    tasks = []
    for model in MODELS:
        for strategy in IMBALANCE_STRATEGIES:
            if is_compatible(model, strategy):
                tasks.append((model, strategy))
            else:
                skipped_runs += 1

    print("=" * 70)
    print(f"🚀 AUTOMATED BATCH EXPERIMENT RUNNER")
    print(f" Total Planned Experiments : {len(tasks)}")
    print(f" Skipped Incompatible combinations : {skipped_runs}")
    print(f" Models ({len(MODELS)})                 : {MODELS}")
    print(f" Imbalance Strategies ({len(IMBALANCE_STRATEGIES)})   : {IMBALANCE_STRATEGIES}")
    print("=" * 70)

    for idx, (model, strategy) in enumerate(tasks, 1):
        print(f"\n[{idx}/{len(tasks)}] Starting Experiment: MODEL={model.upper()} | IMBALANCE={strategy.upper()}")
        print("-" * 70)

        # Command-line 인자로 실행하여 안전하게 독립 프로세스로 구동
        cmd = [
            python_executable, script_path,
            '--model', model,
            '--imbalance', strategy
        ]

        try:
            result = subprocess.run(cmd, check=True)
            print(f"✅ [{idx}/{len(tasks)}] Finished: MODEL={model.upper()} | IMBALANCE={strategy.upper()}")
        except subprocess.CalledProcessError as e:
            print(f"❌ [{idx}/{len(tasks)}] Failed with return code {e.returncode}: MODEL={model.upper()} | IMBALANCE={strategy.upper()}")
        except Exception as e:
            print(f"❌ [{idx}/{len(tasks)}] Unexpected error: {e}")

    print("\n" + "=" * 70)
    print("🎉 ALL BATCH EXPERIMENTS COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    run_grid_search()
