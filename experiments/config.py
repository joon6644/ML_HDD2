import os

# 1. 분할 데이터셋 경로
DATASET_DIR = r"C:\Workspace\projects\26_2_COIN\data\splitted\ST12000NM0007_seed1234"

# 2. 고정 분류 타깃 기간 (일 단위, 30일 타깃 고정)
TARGET_LEAD_TIME = 30

# 3. 타임시리즈 시퀀스 윈도우 크기 (LSTM / GRU 전용)
WINDOW_SIZE = 28

# 4. 피처 학습에서 제외할 메타데이터 및 타깃 컬럼
EXCLUDE_COLS = [
    'serial_number', 'date', 'segment', 'model', 'failure', 'RUL', 'censored'
]

# 5. 선택할 분류 모델 ('rf', 'lgbm', 'xgb', 'mlp', 'lstm', 'gru')
MODEL = 'xgb'

# 6. 기본 불균형 처리 방식:
#    ['none', 'class_weight', 'undersampling', 'oversampling', 'smote', 'adasyn', 'easyensemble', 'focal_loss']
#    - 'none': 클래스 가중치 미적용 (원본 데이터셋 그대로 학습)
#    - 'class_weight': 동적 클래스 가중치 (N_neg / N_pos) 적용
#    - 'focal_loss': PyTorch 모델('mlp', 'lstm', 'gru') 전용 Focal Loss
IMBALANCE_STRATEGY = 'oversampling'

# 7. 클래스 가중치(Class Weight) 명시적 활성화 여부
#    (True: IMBALANCE_STRATEGY와 상관없이 클래스 가중치 N_neg / N_pos 강제 적용, False: IMBALANCE_STRATEGY 설정에 따름)
USE_CLASS_WEIGHT = False

# 8. 추론 평가 방식 (항상 'both'로 고정: 행 단위 + 개체 단위 혼동행렬 동시 평가 및 이어붙여 저장)
INFERENCE_MODE = 'both'

# 9. 기본 난수 시드
SEED = 42

# 10. 롤링 추론 빠른 테스트용 시리얼 샘플 수 (None이면 전체 평가)
SAMPLE_SIZE = None

# 11. 학습 데이터에서 고장 당일(RUL == 0) 샘플 제거 여부 (True: 제거, False: 포함)
DROP_FAILURE_DAY_IN_TRAIN = False

# 12. 실험별 개별 상세 결과 폴더(혼동행렬 CSV, 리드타임 분포 CSV/Plot) 생성 및 저장 여부
#     - True : results/{model}_{imbalance}_.../ 상세 폴더 및 CSV/PNG 아티팩트 생성
#     - False: 상세 폴더를 생성하지 않고, 마스터 누적 CSV(master_experiment_results.csv)에만 기록 (폴더 난립 방지)
SAVE_EXPERIMENT_ARTIFACTS = False
