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
from data_loader import load_dataset, build_sequences, create_binary_target
from checkpoint_utils import load_checkpoint, get_checkpoint_path
from evaluator import RollingEvaluator
from shap_analyzer import SHAPAnalyzer
from alarm_visualizer import AlarmLifetimeVisualizer


def run_post_analysis(checkpoint_path=None, model_name=None, imbalance_strategy=None, dataset_path=None, seed=None, sample_size=None):
    """
    Loads model checkpoint and performs modularized post-analysis:
    1. Rolling Inference & Optimal Threshold Search
    2. Lead Time Distribution Visualization
    3. Disk Lifetime Total Alarm Count Visualization (Failed vs Healthy)
    4. SHAP Feature Importance Summary & Bar Plots
    """
    if model_name is None:
        model_name = config.MODEL
        if isinstance(model_name, (tuple, list)):
            model_name = model_name[0]
            
    if imbalance_strategy is None:
        imbalance_strategy = config.IMBALANCE_STRATEGY
    if dataset_path is None:
        dataset_path = config.DATASET_DIR
    if seed is None:
        seed = config.SEED

    is_cost_sensitive = (config.USE_CLASS_WEIGHT or (imbalance_strategy == 'class_weight'))
    use_focal_loss = (imbalance_strategy == 'focal_loss')
    is_sequence_model = (model_name in ['lstm', 'gru'])
    ckpt_tag = f"cw{int(is_cost_sensitive)}_focal{int(use_focal_loss)}"
    lead_time = config.TARGET_LEAD_TIME

    print("=" * 70)
    print("      MODULAR POST-ANALYSIS & VISUALIZATION PIPELINE      ")
    print(f" Model Architecture   : {model_name.upper()}")
    print(f" Imbalance Strategy   : {imbalance_strategy.upper()}")
    print(f" Dataset              : {dataset_path}")
    print(f" Lead Time Horizon    : {lead_time} days")
    print(f" Pipeline Version     : {config.PIPELINE_VERSION}")
    print("=" * 70)

    # 1. Load Data
    print("\n[Step 1] Loading preprocessed dataset...")
    train_df, val_df, test_df, features = load_dataset(
        dataset_path,
        drop_failure_day_in_train=config.DROP_FAILURE_DAY_IN_TRAIN,
        model=model_name
    )

    # 2. Load Checkpoint
    ckpt_window_size = config.WINDOW_SIZE if is_sequence_model else None
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"\n[Step 2] Loading explicit checkpoint from: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location='cpu') if torch is not None else None
        model = load_checkpoint(
            model_name, imbalance_strategy, seed, lead_time, dataset_path,
            input_dim=len(features), extra_tag=ckpt_tag, features=features, window_size=ckpt_window_size
        )
    else:
        print(f"\n[Step 2] Loading default checkpoint for model '{model_name.upper()}'...")
        model = load_checkpoint(
            model_name, imbalance_strategy, seed, lead_time, dataset_path,
            input_dim=len(features), extra_tag=ckpt_tag, features=features, window_size=ckpt_window_size
        )

    if model is None:
        raise FileNotFoundError(
            f"Checkpoint for model '{model_name}' ({imbalance_strategy}) not found! "
            f"Please run model training first to generate a .ckpt file."
        )

    model_type = 'ensemble' if imbalance_strategy == 'easyensemble' else ('pytorch_class' if is_sequence_model or model_name == 'mlp' else model_name)

    # 3. Setup Results Output Directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_ds = os.path.basename(dataset_path.rstrip('/\\'))
    save_dir = os.path.join(project_root, "results", f"{model_name}_{imbalance_strategy}_{clean_ds}_post_analysis")
    os.makedirs(save_dir, exist_ok=True)
    prefix = f"{model_name}_{imbalance_strategy}"

    # 4. Rolling Evaluator & Optimal Threshold Search
    print("\n[Step 3] Running Rolling Evaluator on Validation Set for Threshold Search...")
    evaluator = RollingEvaluator(
        model=model,
        features=features,
        window_size=config.WINDOW_SIZE if is_sequence_model else 1,
        device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
        model_type=model_type,
        seed=seed
    )

    val_raw_preds = evaluator.get_raw_predictions(val_df, sample_size=sample_size, lead_time=lead_time)
    opt_threshold, max_val_recall = evaluator.find_best_threshold(val_raw_preds, max_far=config.MAX_FAR, lead_time=lead_time)
    print(f"[Optimal Threshold] Found: {opt_threshold:.4f} (Max Val Recall @ FAR <= {config.MAX_FAR:.2%}: {max_val_recall:.4%})")

    # 5. Evaluate Test Set & Lifetime Alarm Analysis
    print("\n[Step 4] Running Rolling Evaluation on Test Set & Generating Lifetime Alarm Visualizations...")
    test_raw_preds = evaluator.get_raw_predictions(test_df, sample_size=sample_size, lead_time=lead_time)

    # Initialize Visualizer
    alarm_vis = AlarmLifetimeVisualizer(test_raw_preds, threshold=opt_threshold, lead_time=lead_time)

    # A. Save Lead Time Distribution Plot
    alarm_vis.plot_lead_time_distribution(save_dir=save_dir, filename_prefix=prefix)

    # B. Save Lifetime Alarm Count Distribution Plot (Failed vs Healthy)
    alarm_vis.plot_lifetime_alarm_counts(save_dir=save_dir, filename_prefix=prefix)

    # 6. SHAP Analysis
    print("\n[Step 5] Running SHAP Feature Importance Analysis...")
    shap_analyzer = SHAPAnalyzer(
        model=model,
        features=features,
        model_type=model_type,
        bg_sample_size=100,
        test_sample_size=200,
        seed=seed
    )

    if is_sequence_model:
        X_test_data, _ = build_sequences(test_df, features, window_size=config.WINDOW_SIZE, lead_time=lead_time)
        X_test_np = X_test_data.numpy() if hasattr(X_test_data, 'numpy') else X_test_data
    else:
        X_test_np = test_df[features].values

    shap_analyzer.generate_and_save_plots(X_test_np, save_dir=save_dir, filename_prefix=prefix)

    print("\n" + "=" * 70)
    print(f"🎉 POST-ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f" All Output Visualizations & Reports saved to: {save_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Modular Post-Analysis (SHAP, Lead Time, Lifetime Alarm Counts) from Checkpoint")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to explicit .ckpt model checkpoint file')
    parser.add_argument('--model', type=str, default=None, help='Model architecture (rf, lgbm, xgb, mlp, lstm, gru)')
    parser.add_argument('--imbalance', type=str, default=None, help='Imbalance strategy')
    parser.add_argument('--data', type=str, default=None, help='Dataset path')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed')
    parser.add_argument('--sample-size', type=int, default=None, help='Limit sample size for rapid testing')
    args = parser.parse_args()

    run_post_analysis(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        imbalance_strategy=args.imbalance,
        dataset_path=args.data,
        seed=args.seed,
        sample_size=args.sample_size
    )


if __name__ == "__main__":
    main()
