import os
import sys
import argparse
import numpy as np
import pandas as pd
try:
    import torch
except ImportError:
    torch = None

import config
from data_loader import load_dataset, create_binary_target, build_sequences
from imbalance import apply_imbalance_treatment, validate_config_compatibility, SUPPORTED_STRATEGIES
from evaluator import calculate_row_level_metrics, RollingEvaluator, save_concatenated_confusion_matrices
from models import (
    train_rf_model,
    train_lgbm_model,
    train_xgb_model,
    train_mlp_model,
    train_lstm_model,
    train_gru_model
)

SUPPORTED_MODELS = ['rf', 'lgbm', 'xgb', 'mlp', 'lstm', 'gru']


def main():
    parser = argparse.ArgumentParser(description="Modular 30-Day Disk Failure Classification Framework")
    parser.add_argument('--model', type=str, default=config.MODEL, choices=SUPPORTED_MODELS, help='Model architecture to evaluate')
    parser.add_argument('--data', type=str, default=config.DATASET_DIR, help='Path to pre-splitted parquet data directory')
    parser.add_argument('--imbalance', type=str, default=config.IMBALANCE_STRATEGY, choices=SUPPORTED_STRATEGIES, help='Imbalance handling strategy')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed for reproducibility')
    parser.add_argument('--lead-time', type=int, default=config.TARGET_LEAD_TIME, help='Fixed target lead time in days')
    parser.add_argument('--sample-size', type=int, default=config.SAMPLE_SIZE, help='Sample serials for fast rolling evaluation')
    parser.add_argument('--drop-failure-day', action='store_true', default=config.DROP_FAILURE_DAY_IN_TRAIN, help='Drop failure day (RUL==0) samples from training set')
    args = parser.parse_args()

    # 0. Validate Configuration & Model-Strategy Compatibility
    validate_config_compatibility(args.model, args.imbalance)

    print("\n============================================================")
    print(f" EXPERIMENT CONFIGURATION (Loaded from config.py)")
    print(f" Model Architecture         : {args.model.upper()}")
    print(f" Dataset Path               : {args.data}")
    print(f" Imbalance Strategy         : {args.imbalance.upper()}")
    print(f" Inference Mode             : ALWAYS BOTH (Row-Level + Disk-Level)")
    print(f" Fixed Target Horizon       : {args.lead_time} days")
    print(f" Seed                       : {args.seed}")
    print(f" Drop Failure Day in Train  : {args.drop_failure_day}")
    print("============================================================\n")

    # 1. Load Data
    train_df, val_df, test_df, features = load_dataset(
        splitted_dir=args.data,
        lead_time=args.lead_time,
        drop_failure_day_in_train=args.drop_failure_day
    )

    # 2. Train Model & Evaluate based on Model Type (Tabular vs Sequence)
    is_sequence_model = args.model in ['lstm', 'gru']
    is_cost_sensitive = (args.imbalance == 'cost_sensitive')
    use_focal_loss = (args.imbalance == 'focal_loss')

    if is_sequence_model:
        window_size = config.WINDOW_SIZE
        print(f"\n[Step 1] Building 3D sequences (window_size={window_size})...")
        X_train_seq, y_train_seq = build_sequences(train_df, features, window_size=window_size, lead_time=args.lead_time)
        X_val_seq, y_val_seq = build_sequences(val_df, features, window_size=window_size, lead_time=args.lead_time)
        X_test_seq, y_test_seq = build_sequences(test_df, features, window_size=window_size, lead_time=args.lead_time)
        
        # Apply Imbalance Handling
        X_train_proc, y_train_proc = apply_imbalance_treatment(X_train_seq, y_train_seq, strategy=args.imbalance, seed=args.seed)
        
        print(f"\n[Step 2] Training {args.model.upper()} sequence model...")
        if args.model == 'lstm':
            model = train_lstm_model(X_train_proc, y_train_proc, X_val_seq, y_val_seq, seed=args.seed, use_focal_loss=use_focal_loss)
        else:
            model = train_gru_model(X_train_proc, y_train_proc, X_val_seq, y_val_seq, seed=args.seed, use_focal_loss=use_focal_loss)
            
        model_type = 'pytorch_class'
    else:
        # Tabular 2D Models (rf, lgbm, xgb, mlp)
        print("\n[Step 1] Preparing 2D tabular features & target...")
        y_train = create_binary_target(train_df, lead_time=args.lead_time)
        y_val = create_binary_target(val_df, lead_time=args.lead_time)
        y_test = create_binary_target(test_df, lead_time=args.lead_time)
        
        X_train_2d = train_df[features].values
        X_val_2d = val_df[features].values
        X_test_2d = test_df[features].values
        
        # Apply Imbalance Strategy
        X_train_proc, y_train_proc = apply_imbalance_treatment(X_train_2d, y_train, strategy=args.imbalance, seed=args.seed)
        
        print(f"\n[Step 2] Training {args.model.upper()} tabular model...")
        if args.model == 'rf':
            model = train_rf_model(X_train_proc, y_train_proc, seed=args.seed, is_cost_sensitive=is_cost_sensitive)
            model_type = 'rf'
        elif args.model == 'lgbm':
            model = train_lgbm_model(X_train_proc, y_train_proc, X_val_2d, y_val, seed=args.seed, is_cost_sensitive=is_cost_sensitive)
            model_type = 'lightgbm'
        elif args.model == 'xgb':
            model = train_xgb_model(X_train_proc, y_train_proc, X_val_2d, y_val, seed=args.seed, is_cost_sensitive=is_cost_sensitive)
            model_type = 'xgb'
        elif args.model == 'mlp':
            model = train_mlp_model(X_train_proc, y_train_proc, X_val_2d, y_val, seed=args.seed, is_cost_sensitive=is_cost_sensitive, use_focal_loss=use_focal_loss)
            model_type = 'pytorch_class'

    # 3. Perform Inference & Evaluation (ALWAYS BOTH)
    print("\n[Step 3] Running Full Inference & Evaluation (Row-Level + Disk-Level)...")

    # A. Row-Level Evaluation
    print("\n--- 1. [ROW-LEVEL (SAMPLE-WISE) EVALUATION] ---")
    if is_sequence_model:
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            test_logits = []
            batch_size = 16384
            for i in range(0, len(X_test_seq), batch_size):
                bx = X_test_seq[i:i+batch_size].to(device)
                test_logits.append(model(bx).cpu())
            test_probs = torch.sigmoid(torch.cat(test_logits)).numpy().flatten()
        row_metrics = calculate_row_level_metrics(y_test_seq.numpy(), test_probs)
    else:
        if hasattr(model, 'predict_proba'):
            test_probs = model.predict_proba(X_test_2d)[:, 1]
        elif hasattr(model, 'predict'):
            test_probs = model.predict(X_test_2d)
            if test_probs.ndim > 1:
                test_probs = test_probs[:, 1]
        elif isinstance(model, torch.nn.Module):
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                X_test_t = torch.tensor(X_test_2d, dtype=torch.float32).to(device)
                test_probs = torch.sigmoid(model(X_test_t)).cpu().numpy().flatten()
        row_metrics = calculate_row_level_metrics(y_test, test_probs)

    print(f"  Confusion Matrix : TN={row_metrics['tn']:,}, FP={row_metrics['fp']:,}, FN={row_metrics['fn']:,}, TP={row_metrics['tp']:,}")
    print(f"  Precision / Rec  : {row_metrics['precision']:.4f} / {row_metrics['recall']:.4f} (F1: {row_metrics['f1']:.4f})")
    print(f"  PR-AUC / AUROC   : {row_metrics['pr_auc']:.4f} / {row_metrics['auroc']:.4f}")
    print(f"  FAR (%)          : {row_metrics['far'] * 100:.2f}%")

    # B. Disk-Level (Entity-wise) Rolling Evaluation
    print("\n--- 2. [DISK-LEVEL (ENTITY-WISE) ROLLING EVALUATION] ---")
    evaluator = RollingEvaluator(
        model=model,
        features=features,
        window_size=config.WINDOW_SIZE if is_sequence_model else 1,
        device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
        model_type=model_type
    )
    disk_metrics = evaluator.evaluate_alarms(test_df, threshold='auto', lead_time=args.lead_time, sample_size=args.sample_size)

    # 4. Save Concatenated Confusion Matrices & Combined Report
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", f"{args.model}_{args.imbalance}_seed{args.seed}")
    save_concatenated_confusion_matrices(
        row_metrics=row_metrics,
        disk_metrics=disk_metrics,
        save_dir=results_dir,
        filename_prefix=f"{args.model}_{args.imbalance}"
    )

    print("Experiment completed successfully.")

if __name__ == "__main__":
    main()
