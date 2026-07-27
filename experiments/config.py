import os

# ==============================================================================
# ⚙️ EXPERIMENT GLOBAL CONFIGURATION
# 실험 관련 모든 설정(데이터 경로, 타깃 기간, 불균형 방식, 추론 방식 등)을 여기서 관리합니다.
# ==============================================================================

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
MODEL = 'lgbm'

# 6. 기본 불균형 처리 방식:
#    ['none', 'undersampling', 'oversampling', 'smote', 'adasyn', 'easyensemble', 'cost_sensitive', 'focal_loss']
#    (※ 'focal_loss'는 PyTorch 모델('mlp', 'lstm', 'gru') 전용입니다)
IMBALANCE_STRATEGY = 'none'

# 7. 추론 평가 방식 (항상 'both'로 고정: 행 단위 + 개체 단위 혼동행렬 동시 평가 및 이어붙여 저장)
INFERENCE_MODE = 'both'

# 8. 기본 난수 시드
SEED = 42

# 9. 롤링 추론 빠른 테스트용 시리얼 샘플 수 (None이면 전체 평가)
SAMPLE_SIZE = None

# 10. 학습 데이터에서 고장 당일(RUL == 0) 샘플 제거 여부 (True: 제거, False: 포함)
DROP_FAILURE_DAY_IN_TRAIN = False
