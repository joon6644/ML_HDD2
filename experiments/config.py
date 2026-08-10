import os

# 1. 분할 데이터셋 경로
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "splitted", "ST12000NM0007")

# 2. 고정 분류 타깃 기간 (일 단위, 30일 타깃 고정)
TARGET_LEAD_TIME = 30

# 3. 타임시리즈 시퀀스 윈도우 크기 (LSTM / GRU 전용)
WINDOW_SIZE = 14

# 4. 피처 학습에서 제외할 메타데이터 및 타깃 컬럼
EXCLUDE_COLS = [
    'serial_number', 'date', 'segment', 'model', 'failure', 'RUL', 'censored'
]

# 5. 선택할 분류 모델 ('lgbm', 'xgb', 'lstm', 'gru')
MODEL = 'lgbm'

# 6. 기본 불균형 처리 방식
IMBALANCE_STRATEGY = 'none'

# 7. 클래스 가중치(Class Weight) 명시적 활성화 여부
USE_CLASS_WEIGHT = False

# 8. 기본 난수 시드
SEED = 42

# 9. 롤링 추론 빠른 테스트용 시리얼 샘플 수 (None이면 전체 평가)
SAMPLE_SIZE = None

# 10. 학습 데이터에서 고장 당일(RUL == 0) 샘플 제거 여부 (True: 제거, False: 포함)
DROP_FAILURE_DAY_IN_TRAIN = False

# 11. 상세 평가 아티팩트 저장 여부
SAVE_EXPERIMENT_ARTIFACTS = False

# 12. GPU 가속 사용 여부 (XGBoost/LightGBM 공통 적용, GPU 미지원/실패 시 자동 CPU 폴백)
USE_GPU = True

# 13. 임곗값 탐색 제약 조건 (FAR 1% 제약 하에서 Recall 최대화)
MAX_FAR = 0.01

# 14. 모델 가중치(체크포인트) 저장 여부
SAVE_MODEL_WEIGHTS = True

# 15. 표준 벤치마크 실험 대상 상수
ALL_DATASETS = ['HGST_20HUH721212ALN604', 'ST12000NM0007', 'TOSHIBA_20MG07ACA14TA', ]
ALL_MODELS = ['lstm', 'gru', 'xgb', 'lgbm']
ALL_SEEDS = [42, 
            43, 44, 45, 46,
            47, 48, 49, 
            50, 51, 52, 53, 54, 
            # 55, 56, 57, 58, 59, 60, 61, 62
            ]


