"""모든 모델이 지키는 인터페이스.

    fit(train, val) -> dict
    predict_proba(data) -> np.ndarray
    save(path)
    load(path)

평가 모듈은 이 인터페이스만 안다. 모델이 XGBoost인지 GRU인지 알 필요가 없다.
그래서 새 모델을 추가할 때 evaluation/ 을 건드리지 않는다.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

META_FILE = "model_meta.json"


class BaseModel(ABC):
    #: "tabular" 이면 FoldMatrix, "sequence" 이면 SequenceFold 를 받는다.
    family: str = "tabular"
    name: str = "base"

    def __init__(self, params: dict, training: dict, seed: int = 42):
        self.params = dict(params or {})
        self.training = dict(training or {})
        self.seed = int(seed)
        self.fit_info: dict[str, Any] = {}

    @abstractmethod
    def fit(self, train, val) -> dict:
        """학습한다. val 은 early stopping 과 threshold 선정에만 쓴다."""

    def warm_start(self, previous: "BaseModel | None") -> bool:
        """직전 fold 의 모델을 초기값으로 삼는다. 지원하면 True.

        월 단위 파인튜닝 프로토콜에서 쓴다. 지원하지 않는 모델은 매 fold
        새로 학습하며, 그 사실이 fit_info 에 기록된다.
        """
        return False

    @abstractmethod
    def predict_proba(self, data) -> np.ndarray:
        """양성 확률을 표본 순서대로 돌려준다."""

    @abstractmethod
    def _save_weights(self, directory: Path) -> None: ...

    @abstractmethod
    def _load_weights(self, directory: Path) -> None: ...

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._save_weights(directory)
        meta = {
            "name": self.name,
            "family": self.family,
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "params": self.params,
            "training": self.training,
            "seed": self.seed,
            "fit_info": self.fit_info,
        }
        with (directory / META_FILE).open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)
        return directory

    @classmethod
    def load(cls, directory: Path) -> "BaseModel":
        directory = Path(directory)
        with (directory / META_FILE).open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        model = cls(meta["params"], meta["training"], meta["seed"])
        model.fit_info = meta.get("fit_info", {})
        model._load_weights(directory)
        return model

    @staticmethod
    def positive_weight(y: np.ndarray) -> float:
        """음성/양성 비. 극단적 불균형에서 손실 가중에 쓴다."""
        positives = float((y == 1).sum())
        negatives = float((y == 0).sum())
        return negatives / positives if positives else 1.0

    def complexity(self) -> int:
        """모델 선정의 마지막 tiebreaker 로 쓰는 대략적인 크기."""
        return 0
