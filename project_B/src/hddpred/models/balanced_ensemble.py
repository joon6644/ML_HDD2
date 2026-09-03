"""균형 배깅 — 아무 hddpred 모델이나 기반으로 쓸 수 있는 앙상블.

imblearn 의 BalancedBaggingClassifier 와 같은 알고리즘이다. 다른 점은 기반
추정기가 sklearn 규약을 지킬 필요가 없다는 것 하나다. 우리 MLP 는 torch 로
쓴 모델이라 imblearn 에 그대로 못 꽂는데, sklearn.neural_network.MLPClassifier
로 갈아끼우면 배치정규화·드롭아웃·PR-AUC 조기종료가 사라져 "표집 방식의
효과"를 재려던 실험에 구조 차이가 섞인다. 그걸 막으려고 직접 쓴다.

알고리즘 (Wallace et al. 2011, "Class Imbalance, Redux" 의 균형 배깅):

    for i in 1..K
        S_i = [양성 전량] + [음성 중 무작위 r x |양성| 개]
        f_i = 기반모델.fit(S_i)
    점수 = mean_i f_i(x)

부분표본마다 음성이 다르므로 K 개의 f 가 서로 다른 음성 영역을 본다. 단일
언더샘플링(rus_1)이 버리는 음성 정보를 앙상블로 되찾는 것이 요지다.

  - 양성은 매 부분표본에 전량 들어간다. 버리지 않는다.
  - 음성 표집은 시드로 결정되므로 같은 시드면 같은 부분표본이 나온다.
  - val 은 기반 모델의 조기종료에 그대로 넘긴다. 재표집하지 않는다.
    평가 구간의 유병률을 건드리면 임곗값 선정이 망가진다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import BaseModel


class BalancedEnsembleModel(BaseModel):
    """params:
    base            기반 모델 config 경로 (configs/models/*.yaml)
    n_estimators    부분표본 개수 K
    negative_ratio  부분표본의 음성/양성 비 r (1 이면 1:1)
    """

    family = "tabular"
    name = "balanced_ensemble"

    def __init__(self, params: dict, training: dict, seed: int = 42):
        super().__init__(params, training, seed)
        self._members: list[BaseModel] = []

    # -- 내부 ---------------------------------------------------------------
    @property
    def _k(self) -> int:
        return int(self.params.get("n_estimators", 10))

    @property
    def _ratio(self) -> float:
        return float(self.params.get("negative_ratio", 1))

    def _base_config(self) -> dict:
        from . import registry

        return registry.load_model_config(self.params["base"])

    def _subsample(self, train, member: int):
        """부분표본 하나. 양성 전량 + 음성 r x |양성| 개."""
        from ..features.fold import FoldMatrix

        positives = np.flatnonzero(train.y == 1)
        negatives = np.flatnonzero(train.y == 0)
        take = min(len(negatives), int(round(len(positives) * self._ratio)))
        rng = np.random.default_rng(self.seed * 1000 + member)
        picked = rng.choice(negatives, size=take, replace=False)
        index = np.sort(np.concatenate([positives, picked]))
        return FoldMatrix(
            X=train.X[index],
            y=train.y[index],
            record_date=train.record_date[index],
            serial=train.serial[index] if train.serial is not None else None,
            failure_date=(
                train.failure_date[index] if train.failure_date is not None else None
            ),
            columns=list(train.columns),
            sampling=dict(train.sampling),
            window=train.window,
        )

    # -- 인터페이스 ----------------------------------------------------------
    def fit(self, train, val) -> dict:
        from . import registry

        base_cfg = self._base_config()
        self._members = []
        sizes = []
        for member in range(self._k):
            part = self._subsample(train, member)
            sizes.append({"n": len(part), "positive_rate": part.positive_rate})
            print(
                f"      [균형배깅 {member + 1}/{self._k}] {len(part):,}행 "
                f"(양성 {part.positive_rate:.2%})"
            )
            model = registry.create(base_cfg, self.seed * 1000 + member)
            model.fit(part, val)
            self._members.append(model)

        self.fit_info = {
            "base": self.params["base"],
            "base_name": base_cfg["name"],
            "n_estimators": self._k,
            "negative_ratio": self._ratio,
            "n_train": len(train),
            "train_positive_rate": train.positive_rate,
            "member_rows": sizes[0]["n"] if sizes else 0,
            "member_positive_rate": sizes[0]["positive_rate"] if sizes else 0.0,
        }
        return self.fit_info

    def predict_proba(self, data) -> np.ndarray:
        if not self._members:
            raise RuntimeError("학습되지 않은 앙상블이다.")
        total = None
        for model in self._members:
            score = model.predict_proba(data)
            total = score if total is None else total + score
        return (total / len(self._members)).astype(np.float64)

    def complexity(self) -> int:
        total = 0
        for model in self._members:
            value = getattr(model, "complexity", None)
            if callable(value):
                try:
                    total += int(value())
                except Exception:  # noqa: BLE001
                    pass
        return total

    def _save_weights(self, directory: Path) -> None:
        manifest = []
        for member, model in enumerate(self._members):
            sub = directory / f"member{member:02d}"
            model.save(sub)
            manifest.append(sub.name)
        with (directory / "members.json").open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False)

    def _load_weights(self, directory: Path) -> None:
        from . import registry

        with (directory / "members.json").open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        klass = registry.resolve_class(self._base_config()["class"])
        self._members = [klass.load(directory / name) for name in manifest]
