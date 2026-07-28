import os
import torch
import joblib
from models.lstm import LSTMClass
from models.gru import GRUClass
from models.mlp import MLPClass
from imbalance import EasyEnsembleModelWrapper

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")

def get_checkpoint_path(model_name: str, imbalance: str, seed: int, lead_time: int, dataset_name: str, extra_tag: str = "") -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    clean_ds = os.path.basename(dataset_name.rstrip('/\\'))
    tag_part = f"_{extra_tag}" if extra_tag else ""
    filename = f"{model_name.lower()}_{imbalance.lower()}{tag_part}_lead{lead_time}_seed{seed}_{clean_ds}.ckpt"
    return os.path.join(CHECKPOINT_DIR, filename)

def save_checkpoint(model, model_name: str, imbalance: str, seed: int, lead_time: int, dataset_name: str, extra_meta: dict = None, extra_tag: str = ""):
    ckpt_path = get_checkpoint_path(model_name, imbalance, seed, lead_time, dataset_name, extra_tag=extra_tag)
    print(f"\n[Checkpoint Manager] Saving model checkpoint to: {ckpt_path}")
    
    payload = {
        "model_name": model_name,
        "imbalance": imbalance,
        "seed": seed,
        "lead_time": lead_time,
        "dataset_name": dataset_name,
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
                    "input_dim": getattr(m, 'input_dim', None)
                })
            else:
                sub_weights.append(m)
        payload["models"] = sub_weights
    elif isinstance(model, torch.nn.Module):
        payload["type"] = "pytorch"
        payload["class_name"] = model.__class__.__name__
        payload["state_dict"] = model.state_dict()
    else:
        payload["type"] = "sklearn_or_tree"
        payload["model_obj"] = model

    torch.save(payload, ckpt_path)
    print(f"[Checkpoint Manager] Model successfully saved! ({os.path.getsize(ckpt_path) / (1024*1024):.2f} MB)")

def load_checkpoint(model_name: str, imbalance: str, seed: int, lead_time: int, dataset_name: str, input_dim: int = None, extra_tag: str = ""):
    ckpt_path = get_checkpoint_path(model_name, imbalance, seed, lead_time, dataset_name, extra_tag=extra_tag)
    if not os.path.exists(ckpt_path):
        return None

    print(f"\n[Checkpoint Manager] Found existing checkpoint: {ckpt_path}")
    print(f"[Checkpoint Manager] Loading pre-trained model weights directly (Skipping training)...")
    payload = torch.load(ckpt_path, map_location='cpu')

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
                
                if cls_name == "LSTMClass":
                    m = LSTMClass(input_dim=in_dim)
                elif cls_name == "GRUClass":
                    m = GRUClass(input_dim=in_dim)
                elif cls_name == "MLPClass":
                    m = MLPClass(input_dim=in_dim)
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
        if cls_name == "LSTMClass":
            model = LSTMClass(input_dim=input_dim)
        elif cls_name == "GRUClass":
            model = GRUClass(input_dim=input_dim)
        elif cls_name == "MLPClass":
            model = MLPClass(input_dim=input_dim)
        else:
            raise ValueError(f"Unknown PyTorch model class in checkpoint: {cls_name}")
        
        model.load_state_dict(st_dict)
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    elif ckpt_type == "sklearn_or_tree":
        return payload["model_obj"]

    return None
