import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ─── 정적 분할 데이터 경로 설정 ──────────────────────────────────────────────
# 다른 모델/데이터셋으로 교체 시 이 디렉토리 경로만 수정하면 됩니다.
SPLITTED_DIR = r"C:\Workspace\projects\26_2_COIN\data\splitted\ST12000NM0007"

# 제외할 feature 컬럼 (학습에 사용되지 않는 메타 데이터 및 파생변수)
EXCLUDE_COLS = ['serial_number', 'date', 'segment', 'model', 'failure', 'RUL', 'censored']


def get_data(splitted_dir: str = SPLITTED_DIR):
    """
    미리 분할되어 저장된 train.parquet, val.parquet 데이터를 로드하고 
    StandardScaler 정규화 및 Clamping을 거쳐 반환합니다.
    """
    # 디렉토리 이름에서 모델명 추출 (예: ST12000NM0007)
    model_name = os.path.basename(splitted_dir.rstrip(r"\/"))
    
    train_path = os.path.join(splitted_dir, f"{model_name}_train.parquet")
    val_path = os.path.join(splitted_dir, f"{model_name}_val.parquet")
    
    print(f"Loading pre-splitted train dataset: {train_path}...")
    train_df = pd.read_parquet(train_path)
    
    print(f"Loading pre-splitted val dataset: {val_path}...")
    val_df = pd.read_parquet(val_path)
    
    # feature 컬럼 자동 식별
    features = [c for c in train_df.columns if c not in EXCLUDE_COLS]
    
    # 메모리 절약을 위해 feature 컬럼들을 즉시 float32로 캐스팅
    print("Casting features to float32 to reduce memory footprint...")
    for col in features:
        train_df[col] = train_df[col].astype('float32')
        val_df[col] = val_df[col].astype('float32')
    
    # 만약을 대비한 결측치 0 채움
    for split in [train_df, val_df]:
        split[features] = split[features].fillna(0)
        
    # StandardScaler: Train 데이터 기준으로 fit하여 Val 데이터 정규화
    print("Standardizing features (fit on train only)...")
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features])
    
    # 이상치 클리핑 [-10.0, 10.0]
    for split in [train_df, val_df]:
        split[features] = split[features].clip(-10.0, 10.0)
        
    # 분할 통계 요약 출력
    def _print_summary(name, d):
        n_serial = d['serial_number'].nunique()
        n_fail = d.groupby('serial_number')['failure'].max().sum()
        print(f"  {name:6s}: {len(d):>10,} rows | {n_serial:>6,} serials | {int(n_fail):>5,} failed ({n_fail/n_serial*100:.1f}%)")
        
    print("Dataset summary:")
    _print_summary("Train", train_df)
    _print_summary("Val", val_df)
    
    return train_df, val_df, features
