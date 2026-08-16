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

# 12. GPU 가속 사용 여부 (XGBoost). 실패 시 조용한 CPU 폴백 없이 즉시 에러.
USE_GPU = True

# 12-1. LightGBM 전용 GPU 설정.
# LightGBM의 GPU 히스토그램(단정밀도)은 이 데이터에서 두 가지 문제를 일으켜 CPU로 고정한다.
#   (1) 재현 불가: 동일 seed/동일 디바이스로 3회 학습 시 서로 다른 모델이 나온다.
#   (2) 학습 중단: TOSHIBA seed 60에서 빈 자식 노드가 생성되며
#       "Check failed: (best_split_info.left_count) > (0)" 로 죽는다.
# CPU는 동일 seed 재실행 시 완전히 동일한 모델을 내고, 이 데이터에서는 GPU보다 빨랐다.
LGBM_USE_GPU = False

# 13. 임곗값 탐색 제약 조건
# 두 평가 층위가 동일한 개념의 제약을 쓰도록 맞춘다(METRIC_DESIGN.md 참조).
#   행 단위  : FAR = FP / N_negative_rows      <= MAX_FAR
#   디스크 단위: FAR = N_cens_early / N_censored <= MAX_DISK_FAR
# 이렇게 해야 "Row 최적화 vs 운영 최적화"의 차이가 제약 정의 차이가 아니라
# 평가 단위 차이에서 나온다.
MAX_FAR = 0.01
MAX_DISK_FAR = 0.01

# 14. 모델 가중치(체크포인트) 저장 여부
SAVE_MODEL_WEIGHTS = True

# 15. 표준 벤치마크 실험 대상 상수
ALL_DATASETS = ['HGST_20HUH721212ALN604', 'ST12000NM0007', 'TOSHIBA_20MG07ACA14TA', ]
ALL_MODELS = ['lstm', 'gru', 'xgb', 'lgbm']
ALL_SEEDS = [
            42, 
            43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 
            55, 56, 57, 58, 59, 60, 61, 62,
            63, 
            64, 65, 66, 67, 68, 69, 70, 71
            ]


