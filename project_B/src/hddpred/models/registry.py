"""모델 설정 -> 모델 인스턴스.

configs/models/*.yaml 의 class 필드를 그대로 import 한다. 새 모델을 추가할 때
이 파일을 고칠 필요가 없다. yaml 하나와 BaseModel 구현 하나면 된다.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from .. import config as cfg_mod
from .base import BaseModel


def load_model_config(path: str | Path) -> dict:
    cfg = cfg_mod.load_yaml(path)
    for key in ("name", "family", "class"):
        if key not in cfg:
            raise KeyError(f"{path}: 모델 설정에 {key!r} 가 없습니다.")
    if cfg["family"] not in {"tabular", "sequence"}:
        raise ValueError(f"{path}: family 는 tabular 또는 sequence 여야 합니다.")
    return cfg


def resolve_class(dotted: str) -> type[BaseModel]:
    module_name, _, class_name = dotted.rpartition(".")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not issubclass(cls, BaseModel):
        raise TypeError(f"{dotted} 는 BaseModel 을 상속하지 않습니다.")
    return cls


def create(model_cfg: dict, seed: int) -> BaseModel:
    cls = resolve_class(model_cfg["class"])
    model = cls(model_cfg.get("params", {}), model_cfg.get("training", {}), seed)
    model.name = model_cfg["name"]
    model.family = model_cfg["family"]
    return model
