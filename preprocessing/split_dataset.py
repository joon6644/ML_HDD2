import os
import argparse
import numpy as np
import pandas as pd

def split_and_save(input_path: str, output_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    """
    주어진 preprocessed parquet 데이터를 serial_number 단위로 8:1:1 층화 그룹 분할을 수행하여 
    모델명_train.parquet, 모델명_val.parquet, 모델명_test.parquet 파일로 저장합니다.
    """
    print(f"Loading preprocessed dataset: {input_path}...")
    df = pd.read_parquet(input_path)
    
    # 정렬 상태 보장
    df = df.sort_values(by=['serial_number', 'date'])
    
    print("Calculating RUL and censoring indicators...")
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    grouped = df.groupby('serial_number')
    
    df['max_date'] = grouped['date'].transform('max')
    df['has_failed'] = grouped['failure'].transform('max')
    
    # RUL (Target variable)
    df['RUL'] = (df['max_date'] - df['date']).dt.days
    # censored: 1 if right-censored, 0 if failure observed
    df['censored'] = 1 - df['has_failed']
    
    df = df.drop(columns=['max_date', 'has_failed'])
    
    # serial_number 단위 고장 여부 층화 준비
    serial_failure = (
        df.groupby('serial_number')['failure']
        .max()
        .reset_index()
        .rename(columns={'failure': 'has_failure'})
    )
    
    print("Performing stratified group split on serial_number...")
    rng = np.random.default_rng(seed)
    train_serials, val_serials, test_serials = [], [], []
    
    for stratum in [0, 1]:
        pool = serial_failure.loc[serial_failure['has_failure'] == stratum, 'serial_number'].values.copy().astype(str)
        rng.shuffle(pool)
        
        n = len(pool)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_serials.extend(pool[:n_train])
        val_serials.extend(pool[n_train:n_train + n_val])
        test_serials.extend(pool[n_train + n_val:])
        
    train_set = set(train_serials)
    val_set = set(val_serials)
    test_set = set(test_serials)
    
    train_df = df[df['serial_number'].isin(train_set)].copy()
    val_df = df[df['serial_number'].isin(val_set)].copy()
    test_df = df[df['serial_number'].isin(test_set)].copy()
    
    # 결과 폴더 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 입력 파일명에서 모델명 추출 (예: ST12000NM0007_preprocessed.parquet -> ST12000NM0007)
    base_name = os.path.basename(input_path)
    model_name = base_name.replace("_preprocessed.parquet", "").replace(".parquet", "")
    
    # parquet 파일로 저장
    print(f"Saving splits to {output_dir}...")
    train_path = os.path.join(output_dir, f"{model_name}_train.parquet")
    val_path = os.path.join(output_dir, f"{model_name}_val.parquet")
    test_path = os.path.join(output_dir, f"{model_name}_test.parquet")
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    # 통계 출력
    def _stats(name, d):
        n_serial = d['serial_number'].nunique()
        n_fail = d.groupby('serial_number')['failure'].max().sum()
        print(f"  {name:6s}: {len(d):>10,} rows | {n_serial:>6,} serials | {int(n_fail):>5,} failed ({n_fail/n_serial*100:.1f}%)")
        
    print("\nSplit statistics:")
    _stats("Train", train_df)
    _stats("Val", val_df)
    _stats("Test", test_df)
    print("\nSplitting and saving completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified group splitting tool by serial_number")
    parser.add_argument("--input", type=str, default=r"data/preprocessed/ST12000NM0007_preprocessed.parquet", help="Path to input preprocessed parquet file")
    parser.add_argument("--output_dir", type=str, default=r"data/splitted/ST12000NM0007", help="Directory to save splitted parquets")
    args = parser.parse_args()
    
    split_and_save(args.input, args.output_dir)
