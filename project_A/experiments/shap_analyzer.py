import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import torch
except ImportError:
    torch = None

try:
    import shap
except ImportError:
    shap = None

import config


class SHAPAnalyzer:
    """
    SHAP (SHapley Additive exPlanations) Analyzer for Machine Learning & Deep Learning Disk Models.
    Supports Tree-based (RF, LGBM, XGB) and PyTorch (MLP, LSTM, GRU) models.
    """

    def __init__(self, model, features, model_type='sklearn', bg_sample_size=100, test_sample_size=200, seed=42):
        self.model = model
        self.features = features
        self.model_type = model_type
        self.bg_sample_size = bg_sample_size
        self.test_sample_size = test_sample_size
        self.seed = seed

    def compute_shap_values(self, X_data):
        """
        Computes SHAP values based on model architecture.
        X_data: numpy array of shape (N, num_features) for tabular or (N, seq_len, num_features) for sequence.
        Returns:
            shap_values: numpy array of SHAP values matching feature dimension
            X_eval: background-adjusted evaluation array (N_eval, num_features)
        """
        if shap is None:
            print("[SHAP Warning] 'shap' package is not installed. Run 'pip install shap' to enable SHAP analysis.")
            return None, None

        np.random.seed(self.seed)
        n_samples = len(X_data)
        if n_samples == 0:
            return None, None

        # Sample data for background and evaluation
        eval_size = min(self.test_sample_size, n_samples)
        eval_indices = np.random.choice(n_samples, size=eval_size, replace=False)

        bg_size = min(self.bg_sample_size, n_samples)
        bg_indices = np.random.choice(n_samples, size=bg_size, replace=False)

        # ---------------------------------------------------------
        # Case A: Sequence Model (3D: N, W, F)
        # ---------------------------------------------------------
        if X_data.ndim == 3:
            # Check if PyTorch NN module for DeepExplainer / GradientExplainer
            if torch is not None and isinstance(self.model, torch.nn.Module):
                try:
                    self.model.eval()
                    device = next(self.model.parameters()).device
                    t_bg = torch.tensor(X_data[bg_indices], dtype=torch.float32).to(device)
                    t_eval = torch.tensor(X_data[eval_indices], dtype=torch.float32).to(device)
                    
                    # Try PyTorch DeepExplainer or GradientExplainer for deep sequence models
                    try:
                        explainer = shap.DeepExplainer(self.model, t_bg)
                        raw_shap = explainer.shap_values(t_eval)
                    except Exception:
                        explainer = shap.GradientExplainer(self.model, t_bg)
                        raw_shap = explainer.shap_values(t_eval)

                    if isinstance(raw_shap, list):
                        raw_shap = raw_shap[0]
                    # Convert tensor to numpy and average over sequence length (W) -> (N, F)
                    shap_vals = np.array(raw_shap).mean(axis=1)
                    X_2d_eval = X_data[eval_indices].mean(axis=1)
                    return shap_vals, X_2d_eval
                except Exception as e:
                    print(f"[SHAP Fallback] PyTorch DeepExplainer failed for 3D sequence ({e}), falling back to Kernel Explainer...")

            # Fallback for sequence models: Average over time dimension
            X_2d_bg = X_data[bg_indices].mean(axis=1)
            X_2d_eval = X_data[eval_indices].mean(axis=1)

            def _predict_seq_wrapper(x_2d):
                W = X_data.shape[1]
                x_3d = np.repeat(x_2d[:, np.newaxis, :], W, axis=1)

                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(x_3d)[:, 1]
                elif hasattr(self.model, 'predict'):
                    preds = self.model.predict(x_3d)
                    return preds[:, 1] if preds.ndim > 1 else preds
                elif isinstance(self.model, torch.nn.Module):
                    self.model.eval()
                    device = next(self.model.parameters()).device
                    with torch.no_grad():
                        t_x = torch.tensor(x_3d, dtype=torch.float32).to(device)
                        probs = torch.sigmoid(self.model(t_x)).cpu().numpy().flatten()
                    return probs
                else:
                    raise ValueError(f"Unsupported model type for sequence SHAP: {type(self.model)}")

            explainer = shap.Explainer(_predict_seq_wrapper, X_2d_bg)
            shap_vals = explainer(X_2d_eval)
            return shap_vals.values, X_2d_eval

        # ---------------------------------------------------------
        # Case B: Tabular Model (2D: N, F)
        # ---------------------------------------------------------
        X_bg = X_data[bg_indices]
        X_eval = X_data[eval_indices]

        # 1. Tree-based Models (Random Forest, LightGBM, XGBoost -> TreeExplainer)
        if self.model_type in ['rf', 'lgbm', 'xgb', 'tree'] or hasattr(self.model, 'tree_explanation'):
            try:
                explainer = shap.TreeExplainer(self.model)
                shap_vals = explainer.shap_values(X_eval)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]  # Binary classification positive class
                return shap_vals, X_eval
            except Exception as e:
                print(f"[SHAP Fallback] TreeExplainer failed ({e}), falling back to generic Explainer...")

        # 2. PyTorch Deep Learning Model (MLP -> DeepExplainer / GradientExplainer)
        if torch is not None and (isinstance(self.model, torch.nn.Module) or self.model_type == 'pytorch_class'):
            try:
                self.model.eval()
                device = next(self.model.parameters()).device
                t_bg = torch.tensor(X_bg, dtype=torch.float32).to(device)
                t_eval = torch.tensor(X_eval, dtype=torch.float32).to(device)

                try:
                    explainer = shap.DeepExplainer(self.model, t_bg)
                    raw_shap = explainer.shap_values(t_eval)
                except Exception:
                    explainer = shap.GradientExplainer(self.model, t_bg)
                    raw_shap = explainer.shap_values(t_eval)

                if isinstance(raw_shap, list):
                    raw_shap = raw_shap[0]
                return np.array(raw_shap), X_eval
            except Exception as e:
                print(f"[SHAP Fallback] PyTorch DeepExplainer failed for 2D MLP ({e}), falling back to Kernel Explainer...")

        # 3. EasyEnsemble or generic model wrapper
        if hasattr(self.model, 'predict_proba'):
            def _predict_proba_wrapper(x_2d):
                return self.model.predict_proba(x_2d)[:, 1]

            explainer = shap.Explainer(_predict_proba_wrapper, X_bg)
            shap_vals = explainer(X_eval)
            return shap_vals.values, X_eval

        # Default fallback
        explainer = shap.Explainer(self.model.predict, X_bg)
        shap_vals = explainer(X_eval)
        return shap_vals.values, X_eval


    def generate_and_save_plots(self, X_data, save_dir, filename_prefix="shap"):
        """
        Computes SHAP values and saves Summary Plot & Bar Plot.
        """
        os.makedirs(save_dir, exist_ok=True)
        print("\n[SHAP Analysis] Calculating SHAP feature importance...")

        shap_values, X_eval = self.compute_shap_values(X_data)
        if shap_values is None:
            return None, None

        # 1. Summary Plot (Beeswarm)
        summary_plot_path = os.path.join(save_dir, f"{filename_prefix}_summary_plot.png")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_eval, feature_names=self.features, show=False)
        plt.title(f"SHAP Feature Importance Summary ({filename_prefix})", fontsize=13, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Mean Absolute SHAP Bar Plot
        bar_plot_path = os.path.join(save_dir, f"{filename_prefix}_bar_plot.png")
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            'feature': self.features,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)

        # Save SHAP Values CSV
        csv_path = os.path.join(save_dir, f"{filename_prefix}_importance.csv")
        shap_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        plt.figure(figsize=(10, 6))
        sns.barplot(data=shap_df.head(20), x='mean_abs_shap', y='feature', palette='Blues_r')
        plt.title(f"Top 20 Features by Mean |SHAP| Value ({filename_prefix})", fontsize=13, fontweight='bold')
        plt.xlabel("Mean |SHAP Value| (Impact on Model Output)", fontsize=11)
        plt.ylabel("Feature", fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(bar_plot_path, dpi=300)
        plt.close()

        print(f"[SHAP Success] Saved Summary Plot : {summary_plot_path}")
        print(f"[SHAP Success] Saved Bar Plot     : {bar_plot_path}")
        print(f"[SHAP Success] Saved Importance CSV: {csv_path}")

        return summary_plot_path, bar_plot_path
