import os
import torch
import joblib
import config
from models.lstm import LSTMClass
from models.gru import GRUClass
from models.mlp import MLPClass
from imbalance import EasyEnsembleModelWrapper

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")

def get_checkpoint_path(model_name: str, imbalance: str, seed: int, lead_time: int, dataset_name: str, extra_tag: str = "") -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    clean_ds = os.path.basename(dataset_name.rstrip('/\\'))
    tag_part = f"_{extra_tag}" if extra_tag else ""
    pv = getattr(config, 'PIPELINE_VERSION', 'v1')
    filename = f"{model_name.lower()}_{imbalance.lower()}{tag_part}_lead{lead_time}_seed{seed}_{clean_ds}_p{pv}.ckpt"
    return os.path.join(CHECKPOINT_DIR, filename)

ARCH_KWARG_NAMES = {
    "GRUClass": ["hidden_dim", "num_layers", "dropout"],
    "LSTMClass": ["hidden_dim", "num_layers", "dropout"],
    "MLPClass": ["hidden_dim1", "hidden_dim2", "dropout"],
}


def _extract_arch_kwargs(m):
    """Recover the architecture kwargs a PyTorch model was constructed with, so
    checkpoint reload reconstructs the exact same architecture instead of relying
    on each class's (possibly different, if custom_params is ever used) defaults."""
    names = ARCH_KWARG_NAMES.get(m.__class__.__name__, [])
    return {name: getattr(m, name) for name in names if hasattr(m, name)}


def save_checkpoint(model, model_name: str, imbalance: str, seed: int, lead_time: int, dataset_name: str,
                    extra_meta: dict = None, extra_tag: str = "", features: list = None, window_size: int = None):
    ckpt_path = get_checkpoint_path(model_name, imbalance, seed, lead_time, dataset_name, extra_tag=extra_tag)
    print(f"\n[Checkpoint Manager] Saving model checkpoint to: {ckpt_path}")

    payload = {
        "model_name": model_name,
        "imbalance": imbalance,
        "seed": seed,
        "lead_time": lead_time,
        "dataset_name": dataset_name,
        "features": list(features) if features is not None else None,
        "window_size": window_size,
        "pipeline_ver": getattr(config, 'PIPELINE_VERSION', 'v1'),
        "extra_meta": extra_meta or {}
    }

    if isinstance(model, EasyEnsembleModelWrapper):
        payload["type"] = "easyensemble"
        payload["is_pytorch"] = model.is_pytorch
        sub_weights = []
        for m in model.models:
            if isinstance(m, torch.nn.Module):
                sub_weights.append({
                    "class_name": m.__class__.__name__,
                    "state_dict": m.state_dict(),
                    "input_dim": getattr(m, 'input_dim', None),
                    "arch_kwargs": _extract_arch_kwargs(m)
                })
            else:
                sub_weights.append(m)
        payload["models"] = sub_weights
    elif isinstance(model, torch.nn.Module):
        payload["type"] = "pytorch"
        payload["class_name"] = model.__class__.__name__
        payload["state_dict"] = model.state_dict()
        payload["arch_kwargs"] = _extract_arch_kwargs(model)
    else:
        payload["type"] = "sklearn_or_tree"
        payload["model_obj"] = model

    torch.save(payload, ckpt_path)
    print(f"[Checkpoint Manager] Model successfully saved! ({os.path.getsize(ckpt_path) / (1024*1024):.2f} MB)")

def _validate_checkpoint_compatibility(payload, ckpt_path, features=None, window_size=None):
    """Guard against silently reusing a checkpoint trained under a different feature
    set, sequence window size, or pipeline version."""
    # Pipeline version check (catches segment/padding logic changes, etc.)
    current_pv = getattr(config, 'PIPELINE_VERSION', 'v1')
    saved_pv   = payload.get('pipeline_ver', 'v1')
    if saved_pv != current_pv:
        raise ValueError(
            f"[Checkpoint Mismatch] '{ckpt_path}' was saved under pipeline version '{saved_pv}', "
            f"but the current pipeline is '{current_pv}'. "
            f"Delete or rename the checkpoint to force retraining under the new pipeline."
        )

    saved_features = payload.get("features")
    if features is not None and saved_features is not None and list(features) != saved_features:
        raise ValueError(
            f"[Checkpoint Mismatch] '{ckpt_path}' was trained on a different feature set than the one "
            f"currently requested (saved={len(saved_features)} features, current={len(features)} features). "
            f"This checkpoint is stale relative to the current config.EXCLUDE_COLS / dataset schema. "
            f"Delete the checkpoint file to retrain, or restore the original feature configuration."
        )

    saved_window_size = payload.get("window_size")
    if window_size is not None and saved_window_size is not None and saved_window_size != window_size:
        raise ValueError(
            f"[Checkpoint Mismatch] '{ckpt_path}' was trained with window_size={saved_window_size}, "
            f"but the current run requests window_size={window_size} (config.WINDOW_SIZE changed?). "
            f"Delete the checkpoint file to retrain with the new window size."
        )


def load_checkpoint(model_name: str, imbalance: str, seed: int, lead_time: int, dataset_name: str,
                    input_dim: int = None, extra_tag: str = "", features: list = None, window_size: int = None):
    ckpt_path = get_checkpoint_path(model_name, imbalance, seed, lead_time, dataset_name, extra_tag=extra_tag)
    if not os.path.exists(ckpt_path):
        return None

    print(f"\n[Checkpoint Manager] Found existing checkpoint: {ckpt_path}")
    print(f"[Checkpoint Manager] Loading pre-trained model weights directly (Skipping training)...")
    payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    _validate_checkpoint_compatibility(payload, ckpt_path, features=features, window_size=window_size)

    ckpt_type = payload.get("type")

    if ckpt_type == "easyensemble":
        is_pytorch = payload.get("is_pytorch", False)
        sub_models = []
        for item in payload["models"]:
            if isinstance(item, dict) and "state_dict" in item:
                cls_name = item["class_name"]
                st_dict = item["state_dict"]
                
                # Infer input_dim if not saved explicitly
                in_dim = item.get("input_dim")
                if in_dim is None and input_dim is not None:
                    in_dim = input_dim
                arch_kwargs = {k: v for k, v in item.get("arch_kwargs", {}).items() if v is not None}

                if cls_name == "LSTMClass":
                    m = LSTMClass(input_dim=in_dim, **arch_kwargs)
                elif cls_name == "GRUClass":
                    m = GRUClass(input_dim=in_dim, **arch_kwargs)
                elif cls_name == "MLPClass":
                    m = MLPClass(input_dim=in_dim, **arch_kwargs)
                else:
                    raise ValueError(f"Unknown PyTorch model class in checkpoint: {cls_name}")
                
                m.load_state_dict(st_dict)
                if torch.cuda.is_available():
                    m = m.cuda()
                sub_models.append(m)
            else:
                sub_models.append(item)
        model = EasyEnsembleModelWrapper(sub_models, is_pytorch=is_pytorch)
        if torch.cuda.is_available() and is_pytorch:
            model.to("cuda")
        return model

    elif ckpt_type == "pytorch":
        cls_name = payload["class_name"]
        st_dict = payload["state_dict"]
        arch_kwargs = {k: v for k, v in payload.get("arch_kwargs", {}).items() if v is not None}
        if cls_name == "LSTMClass":
            model = LSTMClass(input_dim=input_dim, **arch_kwargs)
        elif cls_name == "GRUClass":
            model = GRUClass(input_dim=input_dim, **arch_kwargs)
        elif cls_name == "MLPClass":
            model = MLPClass(input_dim=input_dim, **arch_kwargs)
        else:
            raise ValueError(f"Unknown PyTorch model class in checkpoint: {cls_name}")
        
        model.load_state_dict(st_dict)
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    elif ckpt_type == "sklearn_or_tree":
        return payload["model_obj"]

    return None
