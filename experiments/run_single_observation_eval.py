import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import torch
    # Patch torch.load to default weights_only=False for PyTorch 2.6+ compatibility with saved sklearn/LGBM boosters
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except ImportError:
    torch = None

# Ensure experiments directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from checkpoint_utils import load_checkpoint
from evaluator import RollingEvaluator

DL_MODELS = {'mlp', 'lstm', 'gru'}

# Default fallback thresholds from operational evaluation (val-set tuned)
DEFAULT_THRESHOLDS = {
    ("ST12000NM0007", "LGBM"): 0.08,
    ("ST12000NM0007", "XGB"): 0.11,
    ("ST12000NM0007", "LSTM"): 0.30,
    ("ST12000NM0007", "GRU"): 0.32,
    ("HGST_20HUH721212ALN604", "LGBM"): 0.99,
    ("HGST_20HUH721212ALN604", "XGB"): 0.46,
    ("HGST_20HUH721212ALN604", "LSTM"): 0.11,
    ("HGST_20HUH721212ALN604", "GRU"): 0.16,
    ("TOSHIBA_20MG07ACA14TA", "LGBM"): 0.13,
    ("TOSHIBA_20MG07ACA14TA", "XGB"): 0.13,
    ("TOSHIBA_20MG07ACA14TA", "LSTM"): 0.18,
    ("TOSHIBA_20MG07ACA14TA", "GRU"): 0.13,
}


def load_threshold_map(seed: int = None) -> dict:
    """
    Loads operational evaluation thresholds from master_experiment_results.csv.
    When seed is provided, prefers thresholds from that exact seed's run.
    Falls back to DEFAULT_THRESHOLDS if file is missing or entry is not found.
    """
    threshold_map = DEFAULT_THRESHOLDS.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    master_csv = os.path.join(project_root, "results", "master_experiment_results.csv")

    if os.path.exists(master_csv):
        try:
            df = pd.read_csv(master_csv, encoding='utf-8-sig')
            if 'Threshold' not in df.columns:
                raise KeyError("'Threshold' column missing from master CSV")

            # Normalize Seed column for comparison
            def _norm(v):
                if pd.isna(v): return ""
                try:
                    f = float(v)
                    return str(int(f)) if f.is_integer() else str(f)
                except (ValueError, TypeError):
                    return str(v).strip()

            # First pass: load all thresholds (seed-agnostic baseline)
            for _, row in df.iterrows():
                hdd = str(row['데이터']).strip()
                model_name = str(row['Model']).upper()
                thresh = float(row['Threshold'])
                threshold_map[(hdd, model_name)] = thresh

            # Second pass: if seed specified, override with exact-seed thresholds
            if seed is not None:
                seed_str = _norm(seed)
                seed_df = df[df['Seed'].apply(_norm) == seed_str]
                for _, row in seed_df.iterrows():
                    hdd = str(row['데이터']).strip()
                    model_name = str(row['Model']).upper()
                    thresh = float(row['Threshold'])
                    threshold_map[(hdd, model_name)] = thresh
                if len(seed_df) > 0:
                    print(f"[Threshold Loader] Seed={seed}: loaded {len(seed_df)} seed-specific thresholds from: {master_csv}")
                else:
                    print(f"[Threshold Loader] Seed={seed}: no seed-specific rows found; using latest available thresholds.")
            else:
                print(f"[Threshold Loader] Loaded operational thresholds (all seeds) from: {master_csv}")
        except Exception as e:
            print(f"[Threshold Loader] Warning: Could not read master CSV ({e}). Using default map.")
    return threshold_map


def load_dataset_no_censoring_trim(splitted_dir: str, model_name: str):
    """
    Loads train and test datasets for evaluation.
    - Standardizes features for DL models (fit on train_df, applied to test_df).
    - Crucially: DOES NOT trim trailing 30 days of healthy (censored == 1) HDDs in test set.
    """
    if not os.path.exists(splitted_dir):
        raise FileNotFoundError(f"Dataset directory not found: '{splitted_dir}'")

    files = sorted(os.listdir(splitted_dir))
    train_candidates = [f for f in files if f.endswith('_train.parquet') or f == 'train.parquet']
    test_candidates = [f for f in files if f.endswith('_test.parquet') or f == 'test.parquet']

    if not train_candidates or not test_candidates:
        raise FileNotFoundError(f"Train/Test parquet files missing in '{splitted_dir}'")

    train_path = os.path.join(splitted_dir, train_candidates[0])
    test_path = os.path.join(splitted_dir, test_candidates[0])

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Note: Right censoring 30-day trim is intentionally EXCLUDED for healthy HDDs in test_df

    exclude_cols = config.EXCLUDE_COLS
    features = [c for c in train_df.columns if c not in exclude_cols]

    for df in [train_df, test_df]:
        for col in features:
            if df[col].dtype != 'float32':
                df[col] = df[col].astype('float32')
        df[features] = df[features].fillna(0)

    if model_name.lower() in DL_MODELS:
        scaler = StandardScaler()
        train_df[features] = scaler.fit_transform(train_df[features]).astype('float32')
        test_df[features] = scaler.transform(test_df[features]).astype('float32')
        test_df[features] = test_df[features].clip(-10.0, 10.0)

    return train_df, test_df, features


def evaluate_single_observation(model_name: str, hdd_name: str, threshold: float, seed: int = 42):
    """
    Performs single observation time point evaluation for a given (HDD, Model) pair.
    - Selects the final observation point (last date) for each HDD in test set.
    - Evaluates model prediction at that exact row against the threshold.
    - Computes TP, FP, FN, TN, Disk Precision, Disk Recall, and FAR.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    splitted_dir = os.path.join(project_root, "data", "splitted", hdd_name)

    is_sequence_model = (model_name.lower() in ['lstm', 'gru'])
    ckpt_window_size = config.WINDOW_SIZE if is_sequence_model else None

    # Use config.PIPELINE_VERSION as-is (do NOT override to 'v2'; checkpoints trained by
    # run_experiment.py are saved under the current PIPELINE_VERSION in config.py)
    train_df, test_df, features = load_dataset_no_censoring_trim(splitted_dir, model_name)

    # ckpt_tag must mirror run_experiment.py logic: cw0_focal0 when none/no class weight
    is_cost_sensitive = config.USE_CLASS_WEIGHT or (config.IMBALANCE_STRATEGY == 'class_weight')
    use_focal_loss = (config.IMBALANCE_STRATEGY == 'focal_loss')
    ckpt_tag = f"cw{int(is_cost_sensitive)}_focal{int(use_focal_loss)}"

    model = load_checkpoint(
        model_name.lower(), "none", seed, config.TARGET_LEAD_TIME, splitted_dir,
        input_dim=len(features), extra_tag=ckpt_tag, features=features, window_size=ckpt_window_size
    )

    if model is None:
        raise FileNotFoundError(f"Checkpoint for model '{model_name}' on '{hdd_name}' not found!")

    model_type = 'pytorch_class' if is_sequence_model or model_name.lower() == 'mlp' else model_name.lower()

    evaluator = RollingEvaluator(
        model=model,
        features=features,
        window_size=config.WINDOW_SIZE if is_sequence_model else 1,
        device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
        model_type=model_type,
        seed=seed
    )

    # Perform rolling inference to obtain prediction scores for all dates of each test HDD
    raw_preds = evaluator.get_raw_predictions(test_df, lead_time=config.TARGET_LEAD_TIME)

    # Filter to final observation point (last date) for each HDD
    tp, fp, fn, tn = 0, 0, 0, 0
    for disk in raw_preds:
        has_failed = disk['has_failed']  # True if actual failed HDD, False if healthy
        preds = disk['preds']
        if len(preds) == 0:
            continue
        
        last_pred_score = float(preds[-1])
        is_positive_pred = (last_pred_score >= threshold)

        if has_failed:
            if is_positive_pred:
                tp += 1
            else:
                fn += 1
        else:
            if is_positive_pred:
                fp += 1
            else:
                tn += 1

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        'hdd': hdd_name,
        'model': model_name.upper(),
        'threshold': threshold,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'disk_precision': precision,
        'disk_recall': recall,
        'f1': f1,
        'far': far
    }


def _update_master_csv_single_obs(results: list, seed: int):
    """
    Updates the Single-Obs Precision/Recall/F1/FAR columns in master_experiment_results.csv
    for the given seed using results computed via the uncensored (no-trim) method.
    Preserves all other rows and columns.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    master_csv = os.path.join(project_root, "results", "master_experiment_results.csv")

    if not os.path.exists(master_csv):
        print(f"[Master CSV Update] master_experiment_results.csv not found, skipping.")
        return

    def _norm(v):
        if pd.isna(v): return ""
        try:
            f = float(v)
            return str(int(f)) if f.is_integer() else str(f)
        except (ValueError, TypeError):
            return str(v).strip().lower()

    try:
        df = pd.read_csv(master_csv, encoding='utf-8-sig')

        # Ensure Single-Obs columns exist with object dtype to accept float values safely
        for col in ['Single-Obs Precision', 'Single-Obs Recall', 'Single-Obs F1', 'Single-Obs FAR (%)']:
            if col not in df.columns:
                df[col] = None
            df[col] = df[col].astype(object)

        seed_str = _norm(seed)
        updated = 0
        for res in results:
            hdd = str(res['hdd']).strip()
            model = str(res['model']).upper().strip()
            mask = (
                df['Seed'].apply(_norm) == seed_str
            ) & (
                df['데이터'].apply(lambda v: _norm(v)) == _norm(hdd)
            ) & (
                df['Model'].apply(lambda v: _norm(v)) == _norm(model)
            )
            if mask.sum() == 0:
                print(f"  [Skip] No master CSV row for Seed={seed} | HDD={hdd} | Model={model}")
                continue
            df.loc[mask, 'Single-Obs Precision'] = round(res['disk_precision'], 4)
            df.loc[mask, 'Single-Obs Recall']    = round(res['disk_recall'], 4)
            df.loc[mask, 'Single-Obs F1']         = round(res['f1'], 4)
            df.loc[mask, 'Single-Obs FAR (%)']    = round(res['far'] * 100, 2)
            updated += mask.sum()

        df.to_csv(master_csv, index=False, encoding='utf-8-sig')
        print(f"[Master CSV Update] Updated {updated} row(s) with corrected Single-Obs metrics (Seed={seed}) -> {master_csv}")
    except PermissionError:
        print(f"[Master CSV Update] master_experiment_results.csv is locked (Excel open?). Skipping master update.")
    except Exception as e:
        print(f"[Master CSV Update] Failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Single Observation Time Point Disk Performance Evaluation")
    parser.add_argument('--hdd', type=str, default=None, help='Filter evaluation to specific HDD')
    parser.add_argument('--model', type=str, default=None, help='Filter evaluation to specific model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    hdd_list = ['ST12000NM0007', 'HGST_20HUH721212ALN604', 'TOSHIBA_20MG07ACA14TA']
    model_list = ['LGBM', 'XGB', 'LSTM', 'GRU']

    if args.hdd:
        hdd_list = [args.hdd]
    if args.model:
        model_list = [args.model.upper()]

    threshold_map = load_threshold_map(seed=args.seed)
    results = []

    print("\n" + "=" * 80)
    print("      SINGLE OBSERVATION TIME-POINT DISK PERFORMANCE EVALUATION      ")
    print("=" * 80)

    for hdd in hdd_list:
        for model_name in model_list:
            thresh = threshold_map.get((hdd, model_name), DEFAULT_THRESHOLDS.get((hdd, model_name), 0.5))
            print(f"\n[Evaluating] HDD: {hdd} | Model: {model_name} | Threshold: {thresh:.4f}")
            res = evaluate_single_observation(model_name, hdd, thresh, seed=args.seed)
            results.append(res)

    df_res = pd.DataFrame(results)

    # Format result table as requested:
    # HDD | Model | Disk Precision | Disk Recall | FAR
    print("\n" + "=" * 70)
    print("                       FINAL EVALUATION SUMMARY TABLE                      ")
    print("=" * 70)
    header = f"{'HDD':<25} | {'Model':<6} | {'Disk Precision':<14} | {'Disk Recall':<11} | {'FAR':<8}"
    print(header)
    print("-" * 70)

    for _, row in df_res.iterrows():
        hdd_str = row['hdd']
        m_str = row['model']
        prec_str = f"{row['disk_precision']:.4f}"
        rec_str = f"{row['disk_recall']:.4f}"
        far_str = f"{row['far']:.4f}"
        print(f"{hdd_str:<25} | {m_str:<6} | {prec_str:<14} | {rec_str:<11} | {far_str:<8}")

    print("=" * 70 + "\n")

    # Save summary table to results directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_csv = os.path.join(project_root, "results", "single_observation_eval_results.csv")
    
    export_df = df_res[['hdd', 'model', 'disk_precision', 'disk_recall', 'f1', 'far']].copy()
    export_df.columns = ['HDD', 'Model', 'Disk Precision', 'Disk Recall', 'F1', 'FAR']
    export_df['Seed'] = args.seed

    if os.path.exists(out_csv):
        try:
            existing_df = pd.read_csv(out_csv, encoding='utf-8-sig')
            if 'Seed' not in existing_df.columns:
                existing_df['Seed'] = 42

            def _norm(val):
                if pd.isna(val):
                    return ""
                try:
                    f = float(val)
                    if f.is_integer():
                        return str(int(f))
                    return str(f)
                except (ValueError, TypeError):
                    return str(val).strip().lower()

            mask = pd.Series(False, index=existing_df.index)
            for _, r in export_df.iterrows():
                hdd_match = existing_df['HDD'].apply(_norm) == _norm(r['HDD'])
                model_match = existing_df['Model'].apply(_norm) == _norm(r['Model'])
                seed_match = existing_df['Seed'].apply(_norm) == _norm(r['Seed'])
                mask |= (hdd_match & model_match & seed_match)
            existing_df = existing_df[~mask]
            final_df = pd.concat([existing_df, export_df], ignore_index=True)
        except Exception:
            final_df = export_df
    else:
        final_df = export_df

    final_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"Saved evaluation summary table (Seed={args.seed}) to: {out_csv}\n")

    # Also update master_experiment_results.csv Single-Obs columns with the corrected (uncensored) metrics
    _update_master_csv_single_obs(results, seed=args.seed)


if __name__ == "__main__":
    main()
