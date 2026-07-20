import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ─── 데이터셋 경로 설정 ───────────────────────────────────────────────────────
# 다른 데이터셋으로 교체할 때 이 경로만 수정하면 됩니다.
PARQUET_PATH = r"C:\Workspace\projects\26_2_COIN\data\ST12000NM0007_preprocessed.parquet"

# 학습:검증:평가 비율 (serial_number 기준 그룹 분할)
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
# TEST_RATIO  = 0.1  (나머지)

# 재현성 시드
RANDOM_SEED = 42

# 제외할 feature 컬럼 (분산이 극도로 낮거나 이상치 인코딩 문제가 있는 컬럼)
EXCLUDE_COLS = ['serial_number', 'date', 'segment', 'model', 'failure', 'RUL', 'censored']


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    공통 파생 변수(RUL, censored)를 생성합니다.
    입력 df는 이미 (serial_number, date) 기준으로 정렬된 상태를 가정합니다.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    # serial_number 기준 집계
    grouped = df.groupby('serial_number')
    df['max_date']  = grouped['date'].transform('max')
    df['has_failed'] = grouped['failure'].transform('max')

    # RUL: 해당 날짜로부터 마지막 관측일까지 남은 일수
    df['RUL'] = (df['max_date'] - df['date']).dt.days

    # censored: 1 = 관측 중단(고장 미발생), 0 = 정확한 RUL 관측(고장 발생)
    df['censored'] = 1 - df['has_failed']

    df = df.drop(columns=['max_date', 'has_failed'])
    return df


def split_by_serial(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int):
    """
    serial_number 단위로 8:1:1 층화 분할합니다.
    - 고장 발생 드라이브(has_failure=1)와 미발생(has_failure=0)을 각각 층화하여
      각 분할에 고장 비율이 유사하게 유지됩니다.
    """
    # serial별 고장 여부 집계
    serial_failure = (
        df.groupby('serial_number')['failure']
        .max()
        .reset_index()
        .rename(columns={'failure': 'has_failure'})
    )

    rng = np.random.default_rng(seed)

    train_serials, val_serials, test_serials = [], [], []

    for stratum in [0, 1]:
        pool = serial_failure.loc[
            serial_failure['has_failure'] == stratum, 'serial_number'
        ].values.copy()
        rng.shuffle(pool)

        n = len(pool)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train_serials.extend(pool[:n_train])
        val_serials.extend(pool[n_train:n_train + n_val])
        test_serials.extend(pool[n_train + n_val:])

    train_set = set(train_serials)
    val_set   = set(val_serials)
    test_set  = set(test_serials)

    train_df = df[df['serial_number'].isin(train_set)].copy()
    val_df   = df[df['serial_number'].isin(val_set)].copy()
    test_df  = df[df['serial_number'].isin(test_set)].copy()

    return train_df, val_df, test_df


def get_data(parquet_path: str = PARQUET_PATH):
    """
    전처리된 Parquet 파일을 읽고 8:1:1 serial_number 기준 층화 분할하여 반환합니다.

    Returns:
        train_df : 학습 데이터 (80%)
        val_df   : 검증 데이터 (10%) — 학습 중 하이퍼파라미터 모니터링용
        test_df  : 평가 데이터 (10%) — 최종 성능 보고용
        features : feature 컬럼 리스트 (StandardScaler 적용 완료)
    """
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    print("Computing RUL and censoring indicators...")
    df = prepare_dataset(df)

    print("Splitting by serial_number (8:1:1 stratified)...")
    train_df, val_df, test_df = split_by_serial(df, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)

    # feature 컬럼 결정 (exclude_cols 외 모든 컬럼)
    features = [c for c in df.columns if c not in EXCLUDE_COLS]

    # 결측치 보완 (전처리 후에도 남는 엣지 케이스 대비)
    for split in [train_df, val_df, test_df]:
        split[features] = split[features].fillna(0)

    # StandardScaler: train 기준으로 fit → 모든 split에 transform
    print("Standardizing features (fit on train)...")
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features]   = scaler.transform(val_df[features])
    test_df[features]  = scaler.transform(test_df[features])

    # 이상치 클리핑 [-10, 10]
    for split in [train_df, val_df, test_df]:
        split[features] = split[features].clip(-10.0, 10.0)

    # 분할 통계 출력
    def _stats(name, d):
        n_serial = d['serial_number'].nunique()
        n_fail   = d.groupby('serial_number')['failure'].max().sum()
        print(f"  {name:6s}: {len(d):>10,} rows | {n_serial:>6,} serials "
              f"| {int(n_fail):>5,} failed ({n_fail/n_serial*100:.1f}%)")

    print("Split summary:")
    _stats("Train", train_df)
    _stats("Val",   val_df)
    # test_df (10%) is reserved for final holdout evaluation, not used during training.

    return train_df, val_df, features
