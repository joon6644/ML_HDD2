"""sklearn 추정기 범용 래퍼.

새 모델을 추가할 때 파이썬 코드를 쓰지 않아도 되게 한다. configs/models/*.yaml
에 estimator 경로만 적으면 된다.

    params:
      estimator: sklearn.ensemble.ExtraTreesClassifier
      kwargs: {n_estimators: 300}

세 가지 축을 설정으로 다룬다.

  scale_inputs   트리 계열은 스케일에 불변이라 필요 없지만, 선형/거리 기반
                 모델은 표준화가 없으면 SMART 원시값의 자릿수 차이(전원인가
                 시간 수만 시간 vs 재할당 섹터 수십)에 눌린다. 결측도 트리는
                 분기로 처리하지만 이쪽은 못 하므로 함께 채운다.

  unsupervised   IsolationForest 처럼 라벨을 안 쓰는 모델. AutoEncoder 와 같이
                 정상 개체의 행만 학습하고, 점수는 이상도(높을수록 이상)다.
                 FoldMatrix.healthy_mask 가 "창 끝까지 고장이 확인되지 않은"
                 디스크를 고르므로 미래를 보지 않는다.

  score_attr     확률이 없는 모델(IsolationForest)은 score_samples 를 쓴다.
                 값이 낮을수록 이상이라 부호를 뒤집는다.
"""

from __future__ import annotations

import importlib
import json
import warnings
from pathlib import Path

import numpy as np

from ..features.fold import Scaler
from .base import BaseModel


def _resolve(dotted: str):
    module, _, attr = dotted.rpartition(".")
    return getattr(importlib.import_module(module), attr)


class SklearnModel(BaseModel):
    family = "tabular"
    name = "sklearn"

    def __init__(self, params: dict, training: dict, seed: int = 42):
        super().__init__(params, training, seed)
        self._model = None
        self.scaler: Scaler | None = None

    # -- 내부 ---------------------------------------------------------------
    @property
    def _unsupervised(self) -> bool:
        return bool(self.training.get("unsupervised", False))

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return X
        return self.scaler.transform(X, fill_nan=True)

    # -- 인터페이스 ----------------------------------------------------------
    def fit(self, train, val) -> dict:
        estimator = _resolve(self.params["estimator"])
        kwargs = dict(self.params.get("kwargs", {}))
        # 추정기마다 시드 인자 이름이 다르다. 받는 것만 넘긴다.
        try:
            accepted = estimator().get_params()
        except Exception:  # noqa: BLE001
            accepted = {}
        if "random_state" in accepted:
            kwargs["random_state"] = self.seed

        if self.training.get("scale_inputs", False):
            scaling = self.training.get("scaling", {})
            self.scaler = Scaler(
                scaling.get("method", "standard"),
                scaling.get("clip_quantile", [0.001, 0.999]),
            ).fit(train.X)

        X = self._prepare(train.X)
        self._model = estimator(**kwargs)

        info = {
            "estimator": self.params["estimator"],
            "n_train": len(train),
            "n_val": len(val),
            "train_positive_rate": train.positive_rate,
            "scaled": self.scaler is not None,
            "unsupervised": self._unsupervised,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self._unsupervised:
                healthy = train.healthy_mask()
                n_healthy = int(healthy.sum())
                if n_healthy == 0:
                    raise ValueError("정상 개체의 행이 없어 학습할 수 없습니다.")
                print(
                    f"      [정상만 학습] {len(train):,}행 중 정상 {n_healthy:,}행 "
                    f"({n_healthy / len(train):.2%})"
                )
                info["n_train_healthy"] = n_healthy
                self._model.fit(X[healthy])
            else:
                self._model.fit(X, train.y)

        self.fit_info = info
        return info

    def predict_proba(self, data) -> np.ndarray:
        X = self._prepare(data.X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self._unsupervised:
                # score_samples 는 낮을수록 이상이다. 부호를 뒤집어 이상도로 쓴다.
                # 확률이 아니므로 Brier score 는 이 모델에서 해석하지 않는다.
                return (-self._model.score_samples(X)).astype(np.float64)
            return self._model.predict_proba(X)[:, 1].astype(np.float64)

    def complexity(self) -> int:
        estimators = getattr(self._model, "estimators_", None)
        if estimators is not None:
            total = 0
            for tree in estimators:
                node = getattr(getattr(tree, "tree_", None), "node_count", None)
                total += int(node) if node else 1
            return total
        coef = getattr(self._model, "coef_", None)
        return int(np.asarray(coef).size) if coef is not None else 0

    def _save_weights(self, directory: Path) -> None:
        import joblib

        joblib.dump(self._model, directory / "model.joblib", compress=3)
        state = self.scaler.state_dict() if self.scaler is not None else None
        with (directory / "scaler.json").open("w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False)

    def _load_weights(self, directory: Path) -> None:
        import joblib

        self._model = joblib.load(directory / "model.joblib")
        with (directory / "scaler.json").open("r", encoding="utf-8") as fh:
            state = json.load(fh)
        self.scaler = Scaler.from_state_dict(state) if state else None
