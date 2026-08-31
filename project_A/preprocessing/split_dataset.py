import os
import sys
import argparse
import numpy as np
import pandas as pd

def split_and_save(input_path: str, output_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    """
    주어진 preprocessed parquet 데이터를 serial_number 단위로 8:1:1 층화 그룹 분할을 수행하여
    모델명_train.parquet, 모델명_val.parquet, 모델명_test.parquet 파일로 저장합니다.
    정규화(스케일링)는 여기서 수행하지 않으며, data_loader에서 딥러닝 모델 사용 시에만 적용됩니다.
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
    
    # feature 컬럼 식별
    exclude_cols = ['serial_number', 'date', 'segment', 'model', 'failure', 'RUL', 'censored']
    features = [c for c in df.columns if c not in exclude_cols]
    
    print("Casting features to float32 and targets to compact numeric types...")
    for col in features:
        df[col] = df[col].astype('float32')
    
    df['RUL'] = df['RUL'].astype('int32')
    df['censored'] = df['censored'].astype('int8')
    df['failure'] = df['failure'].astype('int8')

    train_df = df[df['serial_number'].isin(train_set)].copy()
    val_df = df[df['serial_number'].isin(val_set)].copy()
    test_df = df[df['serial_number'].isin(test_set)].copy()

    # 우측 검열(Censored, 정상 종료) HDD의 마지막 30일(lead_time) 제거 로직 적용
    # 학습(Train), 검증(Val), 테스트(Test) 세트 모두 동일하게 유지 적용
    def _trim_censored_tail(d: pd.DataFrame, name: str, lead_time: int = 30) -> pd.DataFrame:
        drop_mask = (d['censored'] == 1) & (d['RUL'] < lead_time)
        dropped = int(drop_mask.sum())
        if dropped > 0:
            print(f"  [{name.upper()}] 우측 검열 HDD 마지막 {lead_time}일 제거: {dropped:,}행 삭제됨")
            return d[~drop_mask].copy()
        return d

    print("Trimming last 30 days of right-censored (non-failed) HDD units across Train, Val, and Test sets...")
    train_df = _trim_censored_tail(train_df, "train")
    val_df = _trim_censored_tail(val_df, "val")
    test_df = _trim_censored_tail(test_df, "test")

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
    
    train_df.to_parquet(train_path, index=False, compression='zstd')
    val_df.to_parquet(val_path, index=False, compression='zstd')
    test_df.to_parquet(test_path, index=False, compression='zstd')
    
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
    MODELS = [
        "HGST_20HUH721212ALN604",
        "TOSHIBA_20MG07ACA14TA",
        "ST12000NM0007"
    ]
    
    parser = argparse.ArgumentParser(description="Stratified group splitting tool by serial_number")
    parser.add_argument("--model", type=str, choices=MODELS + ["ALL"], help="Model name to split")
    parser.add_argument("--input", type=str, help="Custom path to input preprocessed parquet file")
    parser.add_argument("--output_dir", type=str, help="Custom directory to save splitted parquets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for group stratified splitting")
    args, _ = parser.parse_known_args()
    
    target_model = args.model
    
    # CLI 인자가 주어지지 않고 직접 실행한 경우 대화형 메뉴 출력
    if not target_model and not args.input and sys.stdin.isatty():
        print("=" * 60)
        print("   데이터셋 층화 분할 (Train/Val/Test Split) - 모델 선택")
        print("=" * 60)
        for idx, m in enumerate(MODELS, 1):
            print(f"  {idx}. {m}")
        print(f"  4. 전체 모델 실행 (ALL)")
        print("=" * 60)
        
        try:
            choice = input("분할할 모델의 번호를 선택하세요 (1~4): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(MODELS):
                    target_model = MODELS[idx]
                elif idx == 3:
                    target_model = "ALL"
        except Exception:
            pass

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.input and args.output_dir:
        split_and_save(args.input, args.output_dir, seed=args.seed)
    else:
        selected_models = MODELS if target_model == "ALL" or not target_model else [target_model]
        for m in selected_models:
            input_path = os.path.join(project_dir, "data", "preprocessed", f"{m}_preprocessed.parquet")
            output_dir = os.path.join(project_dir, "data", "splitted", m)
            
            if not os.path.exists(input_path):
                print(f"\n[오류] 전처리된 입력 파일이 존재하지 않습니다: {input_path}")
                continue
                
            print(f"\n============================================================")
            print(f" [{m}] 층화 분할 작업 시작")
            print(f"============================================================")
            split_and_save(input_path, output_dir, seed=args.seed)
