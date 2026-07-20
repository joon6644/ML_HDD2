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


def save_epoch_results(model_name, history_list):
    """
    Saves a list of dictionaries (history_list) as a CSV file in the results/ folder.
    """
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{model_name}.csv")
    
    df = pd.DataFrame(history_list)
    # Reorder columns to put 'epoch' first if present
    if 'epoch' in df.columns:
        cols = ['epoch'] + [c for c in df.columns if c != 'epoch']
        df = df[cols]
    df.to_csv(csv_path, index=False)
    print(f"Saved epoch results to {csv_path}")


def log_epoch_to_csv(model_name, epoch_data_dict):
    """
    Saves a single epoch's metrics to results/{model_name}.csv in real-time.
    If epoch is 1, it overwrites/creates the file. Otherwise, it appends.
    """
    import os
    import csv
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{model_name}.csv")
    
    epoch = epoch_data_dict.get('epoch', 1)
    headers = ['epoch'] + sorted([k for k in epoch_data_dict.keys() if k != 'epoch'])
    
    mode = 'w' if epoch == 1 else 'a'
    write_header = (mode == 'w') or (not os.path.exists(csv_path))
    
    with open(csv_path, mode=mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow(epoch_data_dict)


def get_lgbm_callback(model_name):
    """
    Returns a custom LightGBM callback that logs evaluation results to CSV in real-time.
    """
    def callback(env):
        epoch = env.iteration + 1
        row = {'epoch': epoch}
        for dataset_name, metric_name, value, is_higher_better in env.evaluation_result_list:
            name_map = {'training': 'train', 'valid_1': 'val'}
            prefix = name_map.get(dataset_name, dataset_name)
            disp_name = 'loss' if metric_name == 'aft_loss' else metric_name
            row[f"{prefix}_{disp_name}"] = value
        log_epoch_to_csv(model_name, row)
    return callback


class LazySequenceTensor:
    def __init__(self, X_raw, valid_indices, window_size):
        self.X_raw = X_raw  # 2D tensor on CPU (len_df, n_features)
        self.valid_indices = valid_indices  # 1D long tensor
        self.window_size = window_size

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Handle slices, lists, or tensors
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            idx = torch.arange(start, stop, step, dtype=torch.long)
        elif isinstance(idx, int):
            idx = torch.tensor([idx], dtype=torch.long)
        elif not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx, dtype=torch.long)
        
        # Ensure index tensor is on CPU to match valid_indices
        if idx.device != torch.device('cpu'):
            idx = idx.cpu()

        global_idx = self.valid_indices[idx]
        offsets = torch.arange(-self.window_size + 1, 1, dtype=torch.long)
        idx_grid = global_idx.unsqueeze(1) + offsets.unsqueeze(0)  # (B, window_size)
        return self.X_raw[idx_grid]

    def size(self, dim=None):
        """Duck typing for size() method to match torch.Tensor's interface."""
        if dim == 0:
            return len(self)
        elif dim is None:
            return (len(self), self.window_size, self.X_raw.shape[1])
        else:
            raise IndexError("Dimension out of range")


def build_sequences(df, features, window_size):
    """
    Build sequence tensors on CPU using a memory-efficient LazySequenceTensor.
    Reduces memory usage from 16GB+ to under 3GB by avoiding full 3D tensor allocation.
    """
    import torch
    import numpy as np
    
    # Ensure DataFrame is sorted by serial_number and date to keep time-series order
    sort_cols = ['serial_number']
    if 'date' in df.columns:
        sort_cols.append('date')
    df_sorted = df.sort_values(sort_cols)
    
    serials = df_sorted['serial_number'].values
    x_data = df_sorted[features].values
    y_data = df_sorted['RUL'].values
    c_data = df_sorted['censored'].values
    
    n = len(df_sorted)
    if n < window_size:
        return torch.empty((0, window_size, len(features)), dtype=torch.float32), \
               torch.empty((0,), dtype=torch.float32), \
               torch.empty((0,), dtype=torch.float32)
    
    # Identify ending indices for valid windows that do not cross serial number boundaries
    end_indices = np.arange(window_size - 1, n)
    valid_mask = (serials[end_indices - window_size + 1] == serials[end_indices])
    valid_indices = torch.tensor(end_indices[valid_mask], dtype=torch.long)
    
    # Pre-convert raw data to CPU tensors
    X_raw = torch.tensor(x_data, dtype=torch.float32)
    y_raw = torch.tensor(y_data, dtype=torch.float32)
    c_raw = torch.tensor(c_data, dtype=torch.float32)
    
    # Target and censored values for the end of each window
    y_valid = torch.log1p(y_raw[valid_indices])
    c_valid = c_raw[valid_indices]
    
    # Wrap features in LazySequenceTensor
    X_lazy = LazySequenceTensor(X_raw, valid_indices, window_size)
    
    return X_lazy, y_valid, c_valid
