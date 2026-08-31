import os
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
try:
    import torch
except ImportError:
    torch = None

import config

DL_MODELS = {'mlp', 'lstm', 'gru'}
_DATASET_CACHE = {}


def load_dataset(splitted_dir: str = None, lead_time: int = None, drop_failure_day_in_train: bool = None, model: str = None, use_cache: bool = True):
    """
    Loads train, validation, and test datasets dynamically from parquet files in the target directory.
    Constructs the binary target label for 30-day failure classification.
    Optionally drops failure day (RUL == 0) samples from train_df if requested.
    Standardizes features (fit on train only, clipped to [-10, 10]) when `model` is a
    deep learning architecture (mlp/lstm/gru); tree-based models receive raw features.
    Caches raw datasets in memory across consecutive calls for faster experiment runs.
    """
    if splitted_dir is None:
        splitted_dir = config.DATASET_DIR
    if lead_time is None:
        lead_time = config.TARGET_LEAD_TIME
    if drop_failure_day_in_train is None:
        drop_failure_day_in_train = config.DROP_FAILURE_DAY_IN_TRAIN
    if model is None:
        model = config.MODEL

    if not os.path.exists(splitted_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: '{splitted_dir}'")

    cache_key = (os.path.abspath(splitted_dir), lead_time, drop_failure_day_in_train)
    if use_cache and cache_key in _DATASET_CACHE:
        print(f"[Data Loader Cache HIT] Reusing in-memory preloaded dataset for: {splitted_dir}")
        raw_train, raw_val, raw_test, features = _DATASET_CACHE[cache_key]
        train_df, val_df, test_df = raw_train.copy(), raw_val.copy(), raw_test.copy()
    else:
        # Discover train, val, test parquet files (sorted for deterministic selection
        # if a directory ever ends up with more than one file matching a pattern)
        files = sorted(os.listdir(splitted_dir))
        train_candidates = [f for f in files if f.endswith('_train.parquet') or f == 'train.parquet']
        val_candidates = [f for f in files if f.endswith('_val.parquet') or f == 'val.parquet']
        test_candidates = [f for f in files if f.endswith('_test.parquet') or f == 'test.parquet']

        if not train_candidates or not val_candidates or not test_candidates:
            raise FileNotFoundError(
                f"Could not find valid *_train.parquet, *_val.parquet, and *_test.parquet files in '{splitted_dir}'. "
                f"A distinct test set is required to avoid evaluating on the same data used for threshold tuning. "
                f"Directory contains: {files}"
            )
        # Ambiguity here silently decides which split a whole experiment runs on,
        # so it is an error rather than a warning.
        for name, candidates in [("train", train_candidates), ("val", val_candidates), ("test", test_candidates)]:
            if len(candidates) > 1:
                raise ValueError(
                    f"[Data Loader] Multiple candidate {name} files found in '{splitted_dir}': {candidates}. "
                    f"Refusing to guess which split to use -- leave exactly one *_{name}.parquet in the directory."
                )

        train_file, val_file, test_file = train_candidates[0], val_candidates[0], test_candidates[0]
        train_path = os.path.join(splitted_dir, train_file)
        val_path = os.path.join(splitted_dir, val_file)
        test_path = os.path.join(splitted_dir, test_file)

        print(f"Loading datasets from: {splitted_dir}")
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)

        # Trim the trailing `lead_time` days of HDD units whose observation ends
        # normally (censored == 1: covers both mid-life dropout and simply
        # reaching the dataset's last observed date). RUL counts down to 0 at
        # that cutoff, so those final `lead_time` days carry an unverifiable
        # "no failure" label -- the disk could have failed shortly after
        # observation stopped, we just never got to see it.
        def _trim_censored_tail(df: pd.DataFrame, name: str) -> pd.DataFrame:
            drop_mask = (df['censored'] == 1) & (df['RUL'] < lead_time)
            dropped = int(drop_mask.sum())
            if dropped == 0:
                return df
            print(f"[Data Loader] Trimmed last {lead_time} days of censored (non-failed) HDD units "
                  f"from {name.upper()} set: {dropped:,} rows dropped.")
            return df[~drop_mask].copy()

        train_df = _trim_censored_tail(train_df, "train")
        val_df = _trim_censored_tail(val_df, "val")
        test_df = _trim_censored_tail(test_df, "test")

        # Option to drop failure day (RUL == 0) samples from training set only
        if drop_failure_day_in_train:
            before_count = len(train_df)
            train_df = train_df[train_df['RUL'] > 0].copy()
            dropped_count = before_count - len(train_df)
            print(f"[Data Loader] Filtered out failure-day (RUL == 0) samples from TRAIN set: {dropped_count:,} rows dropped.")

        exclude_cols = config.EXCLUDE_COLS
        features = [c for c in train_df.columns if c not in exclude_cols]

        # Preprocessing already removes remaining missing values (paper 4.1.2 step 5),
        # so a NaN here means the preprocessing contract was broken. Imputing it with
        # 0 would feed a fabricated SMART reading into training and evaluation.
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            for col in features:
                if df[col].dtype != 'float32':
                    df[col] = df[col].astype('float32')
            na_counts = df[features].isna().sum()
            if na_counts.any():
                offending = na_counts[na_counts > 0].to_dict()
                raise ValueError(
                    f"[Data Loader] Missing values found in feature columns of the {name.upper()} split "
                    f"of '{splitted_dir}': {offending}. Preprocessing is expected to leave no NaNs; "
                    f"re-run preprocessing rather than imputing at load time."
                )

        if use_cache:
            _DATASET_CACHE[cache_key] = (train_df.copy(), val_df.copy(), test_df.copy(), features)

    # Standardize features for deep learning models only (tree-based models
    # don't need it). Fit strictly on train to avoid leakage, then apply the
    # same transform to val/test and clip outliers to [-10, 10].
    if model in DL_MODELS:
        print(f"[Data Loader] Standardizing features for '{model}' (fit on train only, clip to [-10, 10])...")
        scaler = StandardScaler()
        train_df[features] = scaler.fit_transform(train_df[features]).astype('float32')
        val_df[features] = scaler.transform(val_df[features]).astype('float32')
        test_df[features] = scaler.transform(test_df[features]).astype('float32')
        for df in [train_df, val_df, test_df]:
            df[features] = df[features].clip(-10.0, 10.0)

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
        # Use torch.index_select or direct view for zero-copy efficiency where possible
        return torch.index_select(self.X_raw, 0, idx_grid.view(-1)).view(len(idx), self.window_size, -1)

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

    [Solution A – Training/Inference consistency]
    For every (serial_number, segment) group, window_size-1 copies of the first row
    are prepended BEFORE the actual observations.  This lets the model learn from
    the same first-row-replicated padding windows that predict_disk emits at
    segment starts during rolling inference, eliminating the train/infer mismatch.

    Cross-segment windows (i.e. windows that would span a 4-day+ gap) are still
    rejected by the group_id boundary check.
    """
    if torch is None:
        raise ImportError("PyTorch is required for build_sequences.")
    if window_size is None:
        window_size = config.WINDOW_SIZE
    if lead_time is None:
        lead_time = config.TARGET_LEAD_TIME

    has_segment = 'segment' in df.columns

    # Sort by (serial_number, segment, date) so each segment is a contiguous block
    sort_cols = ['serial_number']
    if has_segment:
        sort_cols.append('segment')
    if 'date' in df.columns:
        sort_cols.append('date')
    df_sorted = df.sort_values(sort_cols).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Solution A: prepend (window_size - 1) copies of each segment's
    # first row so the model sees the same padded windows as inference.
    # ------------------------------------------------------------------
    if window_size > 1:
        group_cols = ['serial_number', 'segment'] if has_segment else ['serial_number']
        pad_frames = []
        for _, grp in df_sorted.groupby(group_cols, sort=False):
            first_row = grp.iloc[0:1]
            pad_frames.append(pd.concat([first_row] * (window_size - 1), ignore_index=True))
        padding_df = pd.concat(pad_frames, ignore_index=True)
        df_aug = pd.concat([padding_df, df_sorted], ignore_index=True)
        # Re-sort so each segment block is: [pad … pad | actual rows]
        df_aug = df_aug.sort_values(sort_cols).reset_index(drop=True)
        n_padded = len(padding_df)
        print(f"[build_sequences] Prepended {n_padded:,} padding rows "
              f"({len(df_sorted.groupby(group_cols))} segment(s) × {window_size-1} rows each).")
    else:
        df_aug = df_sorted
        n_padded = 0

    serials = df_aug['serial_number'].values
    x_data  = df_aug[features].values
    y_rul   = df_aug['RUL'].values
    c_data  = df_aug['censored'].values
    n       = len(df_aug)

    if n < window_size:
        return (torch.empty((0, window_size, len(features)), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32))

    end_indices = np.arange(window_size - 1, n)

    if has_segment:
        seg_arr  = df_aug['segment'].values
        boundary = np.zeros(n, dtype=bool)
        boundary[0]  = True
        boundary[1:] = (serials[1:] != serials[:-1]) | (seg_arr[1:] != seg_arr[:-1])
        group_id = np.cumsum(boundary) - 1

        valid_mask = (group_id[end_indices - window_size + 1] == group_id[end_indices])
        n_excluded = int((~valid_mask).sum())
        if n_excluded > 0:
            print(f"[build_sequences] Excluded {n_excluded:,} cross-segment windows "
                  f"({n_excluded / len(end_indices):.2%} of {len(end_indices):,} candidates).")
    else:
        valid_mask = (serials[end_indices - window_size + 1] == serials[end_indices])

    valid_indices = torch.tensor(end_indices[valid_mask], dtype=torch.long)

    X_raw = torch.tensor(x_data, dtype=torch.float32)
    y_raw = torch.tensor(y_rul,  dtype=torch.float32)
    c_raw = torch.tensor(c_data, dtype=torch.float32)

    y_window = y_raw[valid_indices]
    c_window = c_raw[valid_indices]

    y_binary = ((y_window <= lead_time) & (c_window == 0)).float()
    X_lazy   = LazySequenceTensor(X_raw, valid_indices, window_size)

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
