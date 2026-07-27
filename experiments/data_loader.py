import os
import math
import numpy as np
import pandas as pd
try:
    import torch
except ImportError:
    torch = None

import config


def load_dataset(splitted_dir: str = None, lead_time: int = None, drop_failure_day_in_train: bool = None):
    """
    Loads train, validation, and test datasets dynamically from parquet files in the target directory.
    Constructs the binary target label for 30-day failure classification.
    Optionally drops failure day (RUL == 0) samples from train_df if requested.
    """
    if splitted_dir is None:
        splitted_dir = config.DATASET_DIR
    if lead_time is None:
        lead_time = config.TARGET_LEAD_TIME
    if drop_failure_day_in_train is None:
        drop_failure_day_in_train = config.DROP_FAILURE_DAY_IN_TRAIN

    if not os.path.exists(splitted_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: '{splitted_dir}'")

    # Discover train, val, test parquet files
    files = os.listdir(splitted_dir)
    train_file = next((f for f in files if f.endswith('_train.parquet') or f == 'train.parquet'), None)
    val_file = next((f for f in files if f.endswith('_val.parquet') or f == 'val.parquet'), None)
    test_file = next((f for f in files if f.endswith('_test.parquet') or f == 'test.parquet'), None)

    if not train_file or not val_file:
        raise FileNotFoundError(
            f"Could not find valid *_train.parquet and *_val.parquet files in '{splitted_dir}'. "
            f"Directory contains: {files}"
        )

    train_path = os.path.join(splitted_dir, train_file)
    val_path = os.path.join(splitted_dir, val_file)
    test_path = os.path.join(splitted_dir, test_file) if test_file else val_path

    print(f"Loading datasets from: {splitted_dir}")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    # Option to drop failure day (RUL == 0) samples from training set only
    if drop_failure_day_in_train:
        before_count = len(train_df)
        train_df = train_df[train_df['RUL'] > 0].copy()
        dropped_count = before_count - len(train_df)
        print(f"[Data Loader] Filtered out failure-day (RUL == 0) samples from TRAIN set: {dropped_count:,} rows dropped.")

    exclude_cols = config.EXCLUDE_COLS
    features = [c for c in train_df.columns if c not in exclude_cols]

    for df in [train_df, val_df, test_df]:
        for col in features:
            if df[col].dtype != 'float32':
                df[col] = df[col].astype('float32')
        df[features] = df[features].fillna(0)

    print(f"Loaded: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,} | Features={len(features)}")
    return train_df, val_df, test_df, features


def create_binary_target(df: pd.DataFrame, lead_time: int = None) -> np.ndarray:
    """
    Computes binary failure classification label based on TARGET_LEAD_TIME.
    1: Disk will fail within `lead_time` days (censored == 0)
    0: Healthy disk or outside lead_time
    """
    if lead_time is None:
        lead_time = config.TARGET_LEAD_TIME
    return ((df['RUL'] <= lead_time) & (df['censored'] == 0)).astype(np.float32).values


class LazySequenceTensor:
    """
    Memory-efficient sequence window generator for PyTorch sequence models.
    """
    def __init__(self, X_raw, valid_indices, window_size):
        self.X_raw = X_raw
        self.valid_indices = valid_indices
        self.window_size = window_size

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            idx = torch.arange(start, stop, step, dtype=torch.long)
        elif isinstance(idx, int):
            idx = torch.tensor([idx], dtype=torch.long)
        elif not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx, dtype=torch.long)
        
        if idx.device != torch.device('cpu'):
            idx = idx.cpu()

        global_idx = self.valid_indices[idx]
        offsets = torch.arange(-self.window_size + 1, 1, dtype=torch.long)
        idx_grid = global_idx.unsqueeze(1) + offsets.unsqueeze(0)
        return self.X_raw[idx_grid]

    def size(self, dim=None):
        if dim == 0:
            return len(self)
        elif dim is None:
            return (len(self), self.window_size, self.X_raw.shape[1])
        else:
            raise IndexError("Dimension out of range")


def build_sequences(df: pd.DataFrame, features: list, window_size: int = None, lead_time: int = None):
    """
    Build sequence dataset for LSTM/GRU time-series models.
    """
    if torch is None:
        raise ImportError("PyTorch is required for build_sequences.")
    if window_size is None:
        window_size = config.WINDOW_SIZE
    if lead_time is None:
        lead_time = config.TARGET_LEAD_TIME
        
    sort_cols = ['serial_number']
    if 'date' in df.columns:
        sort_cols.append('date')
    df_sorted = df.sort_values(sort_cols)
    
    serials = df_sorted['serial_number'].values
    x_data = df_sorted[features].values
    y_rul = df_sorted['RUL'].values
    c_data = df_sorted['censored'].values
    
    n = len(df_sorted)
    if n < window_size:
        return (torch.empty((0, window_size, len(features)), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32))
    
    end_indices = np.arange(window_size - 1, n)
    valid_mask = (serials[end_indices - window_size + 1] == serials[end_indices])
    valid_indices = torch.tensor(end_indices[valid_mask], dtype=torch.long)
    
    X_raw = torch.tensor(x_data, dtype=torch.float32)
    y_raw = torch.tensor(y_rul, dtype=torch.float32)
    c_raw = torch.tensor(c_data, dtype=torch.float32)
    
    y_window = y_raw[valid_indices]
    c_window = c_raw[valid_indices]
    
    y_binary = ((y_window <= lead_time) & (c_window == 0)).float()
    X_lazy = LazySequenceTensor(X_raw, valid_indices, window_size)
    
    return X_lazy, y_binary


def log_epoch_to_csv(model_name: str, epoch_data_dict: dict, results_dir: str = None):
    """Logs epoch metrics to CSV in real time."""
    import csv
    if results_dir is None:
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
