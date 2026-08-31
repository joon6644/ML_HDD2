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
from imbalance import apply_imbalance_treatment, validate_config_compatibility, EasyEnsembleHandler, EasyEnsembleModelWrapper, SUPPORTED_STRATEGIES
from evaluator import (
    calculate_row_level_metrics,
    find_best_threshold_row_level,
    RollingEvaluator,
    save_concatenated_confusion_matrices,
    save_lead_time_distribution,
    append_master_experiment_result
)
from checkpoint_utils import load_checkpoint, save_checkpoint
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
    parser.add_argument('--use-class-weight', action='store_true', default=config.USE_CLASS_WEIGHT, help='Force enable class weighting (N_neg / N_pos)')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed for reproducibility')
    parser.add_argument('--lead-time', type=int, default=config.TARGET_LEAD_TIME, help='Fixed target lead time in days')
    parser.add_argument('--drop-failure-day', action='store_true', default=config.DROP_FAILURE_DAY_IN_TRAIN, help='Drop failure-day (RUL == 0) samples from the training set')
    parser.add_argument('--save-artifacts', action='store_true', default=config.SAVE_EXPERIMENT_ARTIFACTS, help='Save detailed subfolders/plots')
    parser.add_argument('--save-model-weights', action='store_true', default=getattr(config, 'SAVE_MODEL_WEIGHTS', True), help='Save model checkpoint weights to checkpoints/ directory')
    parser.add_argument('--sample-size', type=int, default=config.SAMPLE_SIZE, help='Limit val/test disks for rapid testing')
    parser.add_argument('--use-gpu', action='store_true', default=config.USE_GPU, help='Attempt GPU training for XGBoost/LightGBM (auto CPU fallback on failure)')
    args = parser.parse_args()

    # Validate compatibility
    validate_config_compatibility(args.model, args.imbalance)
    is_cost_sensitive = args.use_class_weight or (args.imbalance == 'class_weight')
    use_focal_loss = (args.imbalance == 'focal_loss')
    is_sequence_model = args.model in ['lstm', 'gru']
    ckpt_tag = f"cw{int(is_cost_sensitive)}_focal{int(use_focal_loss)}"

    print("=" * 60)
    print(" EXPERIMENT CONFIGURATION (Loaded from config.py)")
    print(f" Model Architecture         : {args.model.upper()}")
    print(f" Dataset Path               : {args.data}")
    print(f" Imbalance Strategy         : {args.imbalance.upper()}")
    print(f" Class Weight Enabled       : {is_cost_sensitive}")
    print(f" Early Stopping             : ENABLED for NN/Boosting (Val Loss); RF reports OOB + Val AUROC")
    print(f" Threshold Optimization     : Recall-Maximization under FAR <= 1% Constraint on Validation Set")
    print(f" Inference Mode             : ALWAYS BOTH (Row-Level + Disk-Level)")
    print(f" Save Detail Subfolders     : {args.save_artifacts}")
    print(f" Save Model Weights         : {args.save_model_weights}")
    print(f" Fixed Target Horizon       : {args.lead_time} days")
    print(f" Seed                       : {args.seed}")
    print(f" Drop Failure Day in Train  : {args.drop_failure_day}")
    print(f" GPU Acceleration (XGB/LGBM): {'ENABLED (auto CPU fallback)' if args.use_gpu else 'DISABLED (CPU only)'}")

    if args.sample_size:
        print(f" [FAST TEST MODE]          : Sample size limited to {args.sample_size} disks")
    print("=" * 60)

    # 1. Load Data
    train_df, val_df, test_df, features = load_dataset(args.data, drop_failure_day_in_train=args.drop_failure_day, model=args.model)
    
    # 2. Check Checkpoint First
    ckpt_window_size = config.WINDOW_SIZE if is_sequence_model else None
    cached_model = load_checkpoint(args.model, args.imbalance, args.seed, args.lead_time, args.data, input_dim=len(features), extra_tag=ckpt_tag, features=features, window_size=ckpt_window_size)

    # 3. Build Features (Only build validation and test features for inference if checkpoint exists)
    if is_sequence_model:
        window_size = config.WINDOW_SIZE
        print(f"\n[Step 1] Building Val/Test 3D sequences (window_size={window_size})...")
        X_val_seq, y_val_seq = build_sequences(val_df, features, window_size=window_size, lead_time=args.lead_time)
        X_test_seq, y_test_seq = build_sequences(test_df, features, window_size=window_size, lead_time=args.lead_time)
        if cached_model is None:
            print(f"[Step 1] Building Train 3D sequences for model training...")
            X_train_seq, y_train_seq = build_sequences(train_df, features, window_size=window_size, lead_time=args.lead_time)
    else:
        print("\n[Step 1] Preparing Val/Test 2D tabular features & target...")
        y_val = create_binary_target(val_df, lead_time=args.lead_time)
        y_test = create_binary_target(test_df, lead_time=args.lead_time)
        X_val_2d = val_df[features].values
        X_test_2d = test_df[features].values
        if cached_model is None:
            print("[Step 1] Preparing Train 2D tabular features & target for model training...")
            y_train = create_binary_target(train_df, lead_time=args.lead_time)
            X_train_2d = train_df[features].values

    # 4. Train Model (if not cached)
    if cached_model is not None:
        model = cached_model
        model_type = 'ensemble' if args.imbalance == 'easyensemble' else ('pytorch_class' if is_sequence_model or args.model == 'mlp' else args.model)
    else:
        if is_sequence_model:
            print(f"\n[Step 2] Training {args.model.upper()} sequence model with Early Stopping on Val set...")
            if args.imbalance == 'easyensemble':
                print(f"[EasyEnsemble Protocol] Generating K=10 balanced sub-datasets & training 10 bagged sequence estimators...")
                ee = EasyEnsembleHandler(n_estimators=10, seed=args.seed)
                subsets = ee.generate_subsets(X_train_seq, y_train_seq)
                models = []
                for k, (X_sub, y_sub) in enumerate(subsets):
                    sub_seed = args.seed
                    print(f" -> Training EasyEnsemble Sequence Estimator {k+1}/10 (seed={sub_seed})...")
                    if args.model == 'lstm':
                        m = train_lstm_model(X_sub, y_sub, X_val_seq, y_val_seq, seed=sub_seed, is_cost_sensitive=is_cost_sensitive, use_focal_loss=use_focal_loss)
                    else:
                        m = train_gru_model(X_sub, y_sub, X_val_seq, y_val_seq, seed=sub_seed, is_cost_sensitive=is_cost_sensitive, use_focal_loss=use_focal_loss)
                    models.append(m)
                model = EasyEnsembleModelWrapper(models, is_pytorch=True)
            else:
                X_train_proc, y_train_proc = apply_imbalance_treatment(X_train_seq, y_train_seq, strategy=args.imbalance, seed=args.seed, dataset_path=args.data, lead_time=args.lead_time, drop_failure_day=args.drop_failure_day)
                if args.model == 'lstm':
                    model = train_lstm_model(X_train_proc, y_train_proc, X_val_seq, y_val_seq, seed=args.seed, is_cost_sensitive=is_cost_sensitive, use_focal_loss=use_focal_loss)
                else:
                    model = train_gru_model(X_train_proc, y_train_proc, X_val_seq, y_val_seq, seed=args.seed, is_cost_sensitive=is_cost_sensitive, use_focal_loss=use_focal_loss)

            model_type = 'ensemble' if args.imbalance == 'easyensemble' else 'pytorch_class'
        else:
            print(f"\n[Step 2] Training {args.model.upper()} tabular model with Early Stopping on Val set...")
            if args.imbalance == 'easyensemble':
                print(f"[EasyEnsemble Protocol] Generating K=10 balanced sub-datasets & training 10 bagged tabular estimators...")
                ee = EasyEnsembleHandler(n_estimators=10, seed=args.seed)
                subsets = ee.generate_subsets(X_train_2d, y_train)
                models = []
                for k, (X_sub, y_sub) in enumerate(subsets):
                    sub_seed = args.seed
                    print(f" -> Training EasyEnsemble Tabular Estimator {k+1}/10 (seed={sub_seed})...")
                    if args.model == 'rf':
                        m = train_rf_model(X_sub, y_sub, X_val_2d, y_val, seed=sub_seed, is_cost_sensitive=is_cost_sensitive)
                    elif args.model == 'lgbm':
                        m = train_lgbm_model(X_sub, y_sub, X_val_2d, y_val, seed=sub_seed, is_cost_sensitive=is_cost_sensitive, use_gpu=args.use_gpu)
                    elif args.model == 'xgb':
                        m = train_xgb_model(X_sub, y_sub, X_val_2d, y_val, seed=sub_seed, is_cost_sensitive=is_cost_sensitive, use_gpu=args.use_gpu)
                    elif args.model == 'mlp':
                        m = train_mlp_model(X_sub, y_sub, X_val_2d, y_val, seed=sub_seed, is_cost_sensitive=is_cost_sensitive, use_focal_loss=use_focal_loss)
                    models.append(m)
                model = EasyEnsembleModelWrapper(models, is_pytorch=(args.model == 'mlp'))
                model_type = 'ensemble'
            else:
                X_train_proc, y_train_proc = apply_imbalance_treatment(X_train_2d, y_train, strategy=args.imbalance, seed=args.seed, dataset_path=args.data, lead_time=args.lead_time, drop_failure_day=args.drop_failure_day)
                if args.model == 'rf':
                    model = train_rf_model(X_train_proc, y_train_proc, X_val_2d, y_val, seed=args.seed, is_cost_sensitive=is_cost_sensitive)
                    model_type = 'rf'
                elif args.model == 'lgbm':
                    model = train_lgbm_model(X_train_proc, y_train_proc, X_val_2d, y_val, seed=args.seed, is_cost_sensitive=is_cost_sensitive, use_gpu=args.use_gpu)
                    model_type = 'lgbm'
                elif args.model == 'xgb':
                    model = train_xgb_model(X_train_proc, y_train_proc, X_val_2d, y_val, seed=args.seed, is_cost_sensitive=is_cost_sensitive, use_gpu=args.use_gpu)
                    model_type = 'xgb'
                elif args.model == 'mlp':
                    model = train_mlp_model(X_train_proc, y_train_proc, X_val_2d, y_val, seed=args.seed, is_cost_sensitive=is_cost_sensitive, use_focal_loss=use_focal_loss)
                    model_type = 'pytorch_class'

        # Save checkpoint immediately after training if enabled in config/cli
        if args.save_model_weights:
            save_checkpoint(model, args.model, args.imbalance, args.seed, args.lead_time, args.data, extra_tag=ckpt_tag, features=features, window_size=ckpt_window_size)
        else:
            print("\n[Checkpoint Manager] Skipping checkpoint save (SAVE_MODEL_WEIGHTS = False).")

    # 3. Perform Inference & Evaluation (ALWAYS BOTH)
    print("\n[Step 3] Running Full Inference & Evaluation (Row-Level + Disk-Level)...")
    evaluator = RollingEvaluator(
        model=model,
        features=features,
        window_size=config.WINDOW_SIZE if is_sequence_model else 1,
        device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
        model_type=model_type,
        seed=args.seed
    )

    # Helper: compute row-level predicted probabilities for a given feature matrix,
    # shared between Validation (threshold search) and Test (final evaluation).
    def _predict_row_probs(X_seq_or_2d):
        if is_sequence_model:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X_seq_or_2d)[:, 1]
            model.eval()
            with torch.no_grad():
                logits = []
                batch_size = 16384
                for i in range(0, len(X_seq_or_2d), batch_size):
                    bx = X_seq_or_2d[i:i+batch_size].to(device)
                    logits.append(model(bx).cpu())
                return torch.sigmoid(torch.cat(logits)).numpy().flatten()
        else:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_seq_or_2d)[:, 1]
            elif hasattr(model, 'predict'):
                probs = model.predict(X_seq_or_2d)
                if probs.ndim > 1:
                    probs = probs[:, 1]
            elif isinstance(model, torch.nn.Module):
                model.eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                with torch.no_grad():
                    X_t = torch.tensor(X_seq_or_2d, dtype=torch.float32).to(device)
                    probs = torch.sigmoid(model(X_t)).cpu().numpy().flatten()
            return probs

    # 1) Search optimal decision threshold on Validation Set, Row-Level (Sample-wise),
    # Recall-Maximization under FAR <= 1% Constraint. No data leakage (val set only).
    print("\n[Search] Finding optimal decision threshold on Validation Set, Row-Level (Recall Maximization under FAR <= 1%)...")
    val_probs = _predict_row_probs(X_val_seq if is_sequence_model else X_val_2d)
    y_val_np = y_val_seq.numpy() if is_sequence_model else y_val
    opt_threshold, max_val_recall = find_best_threshold_row_level(y_val_np, val_probs, max_far=getattr(config, 'MAX_FAR', 0.01))
    print(f"[Optimal Threshold] Found on Val Set (Row-Level): {opt_threshold:.4f} (Max Val Recall @ FAR <= 1%: {max_val_recall:.4%})")

    # A. Row-Level Evaluation (Using Validation-Optimal Threshold)
    print("\n--- 1. [ROW-LEVEL (SAMPLE-WISE) EVALUATION] ---")
    test_probs = _predict_row_probs(X_test_seq if is_sequence_model else X_test_2d)
    row_metrics = calculate_row_level_metrics(y_test_seq.numpy() if is_sequence_model else y_test, test_probs, threshold=opt_threshold)

    print(f"  Threshold Used   : {opt_threshold:.4f}")
    print(f"  Confusion Matrix : TN={row_metrics['tn']:,}, FP={row_metrics['fp']:,}, FN={row_metrics['fn']:,}, TP={row_metrics['tp']:,}")
    print(f"  Precision / Rec  : {row_metrics['precision']:.4f} / {row_metrics['recall']:.4f} (F1: {row_metrics['f1']:.4f})")
    print(f"  PR-AUC / AUROC   : {row_metrics['pr_auc']:.4f} / {row_metrics['auroc']:.4f}")
    print(f"  FAR (%)          : {row_metrics['far'] * 100:.2f}%")

    # B. Disk-Level (Entity-wise) Rolling Evaluation
    print("\n--- 2. [DISK-LEVEL (ENTITY-WISE) ROLLING EVALUATION] ---")

    # 2) Evaluate Test Set using fixed optimal threshold found from Val Set
    print("\n[Evaluation] Evaluating Test Set using fixed Validation-Optimal Threshold...")
    disk_metrics, report_df = evaluator.evaluate_alarms(test_df, threshold=opt_threshold, sample_size=args.sample_size)

    # 4. Save Experiment Artifacts (Controlled by --save-artifacts / SAVE_EXPERIMENT_ARTIFACTS)
    if args.save_artifacts:
        drop_day_str = "dropday_true" if args.drop_failure_day else "dropday_false"
        folder_name = f"{args.model}_{args.imbalance}_{drop_day_str}_seed{args.seed}"
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(project_root, "results", folder_name)

        print(f"\n[Step 4] Saving paired evaluation detail artifacts to: {results_dir}")

        # A. Concatenated Confusion Matrix CSV
        save_concatenated_confusion_matrices(
            row_metrics=row_metrics,
            disk_metrics=disk_metrics,
            save_dir=results_dir,
            filename_prefix=f"{args.model}_{args.imbalance}"
        )

        # B. Lead Time Distribution CSV & Plot PNG
        save_lead_time_distribution(
            report_df=report_df,
            save_dir=results_dir,
            filename_prefix=f"{args.model}_{args.imbalance}"
        )
    else:
        print("\n[Step 4] Skipping detailed subfolder creation (SAVE_EXPERIMENT_ARTIFACTS = False).")

    # C. Permanent Master Experiment Cumulative Log CSV
    # DO NOT log fast test runs with sample reduction (sample_size is not None) to prevent master CSV metrics distortion
    if args.sample_size is None:
        try:
            append_master_experiment_result(
                model_name=args.model,
                dataset_name=args.data,
                imbalance_strategy=args.imbalance,
                drop_failure_day=args.drop_failure_day,
                row_metrics=row_metrics,
                disk_metrics=disk_metrics,
                seed=args.seed,
                lead_time=args.lead_time
            )
        except PermissionError:
            print("\n[Warning] Could not write to master_experiment_results.csv because the file is currently open in Excel or another program.")
            print("  Please close the file in Excel if you wish to record full benchmark results.\n")
    else:
        print(f"\n[Notice] Fast test run with sample_size={args.sample_size}. SKIPPED appending to master CSV to prevent distorted benchmark records.")

    print("Experiment completed successfully.")

if __name__ == "__main__":
    main()
