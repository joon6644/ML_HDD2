"""균형 배깅 — 시퀀스 계열용.

balanced_ensemble.BalancedEnsembleModel 과 같은 알고리즘이고, 부분표본을
만드는 방식만 다르다.

    for i in 1..K
        S_i = [양성 전량] + [음성 중 무작위 r x |양성| 개]
        f_i = 기반모델.fit(S_i)
    점수 = mean_i f_i(x)

━━ 시퀀스에서 무엇이 다른가 ━━

SequenceFold 는 (행렬, 표본 끝 인덱스) 로 되어 있다. 윈도우를 미리 복제하지
않고, 표본 i 는 matrix 의 end_index[i] 에서 lookback 만큼 거슬러 읽는다.

그래서 부분표집은 **end_index 쪽에만** 건다. matrix 는 창을 채우는 재료이므로
손대지 않는다. 행렬에서 행을 빼면 남은 표본의 lookback 창에 구멍이 뚫려,
"음성을 덜 봤다"가 아니라 "입력이 망가졌다"가 된다.

fold.py 의 stride 표집이 같은 이유로 같은 일을 한다:
    "표본 추출은 채점 대상 인덱스에만 적용한다. 행렬은 lookback 재료이므로
     손대지 않는다."

━━ val 은 재표집하지 않는다 ━━

기반 모델의 조기 종료에 그대로 넘긴다. 평가 구간의 유병률을 건드리면 임곗값
선정이 망가진다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import BaseModel


class BalancedSequenceEnsembleModel(BaseModel):
    """params:
    base            기반 모델 config 경로 (configs/models/*.yaml, family=sequence)
    n_estimators    부분표본 개수 K
    negative_ratio  부분표본의 음성/양성 비 r (1 이면 1:1)
    """

    family = "sequence"
    name = "balanced_sequence_ensemble"

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
        """부분표본 하나. 양성 전량 + 음성 r x |양성| 개.

        end_index 와 그에 맞물린 배열만 고른다. matrix 는 통째로 넘긴다.
        """
        from ..features.fold import SequenceFold

        positives = np.flatnonzero(train.y == 1)
        negatives = np.flatnonzero(train.y == 0)
        take = min(len(negatives), int(round(len(positives) * self._ratio)))
        rng = np.random.default_rng(self.seed * 1000 + member)
        picked = rng.choice(negatives, size=take, replace=False)
        index = np.sort(np.concatenate([positives, picked]))

        def pick(array):
            return None if array is None else array[index]

        return SequenceFold(
            matrix=train.matrix,  # 창 재료. 자르지 않는다.
            end_index=train.end_index[index],
            lookback=train.lookback,
            y=train.y[index],
            record_date=train.record_date[index],
            serial=pick(train.serial),
            failure_date=pick(train.failure_date),
            columns=list(train.columns),
            valid_len=pick(train.valid_len),
            sampling=dict(train.sampling),
            window=train.window,
        )

    # -- 인터페이스 ----------------------------------------------------------
    def fit(self, train, val) -> dict:
        from . import registry

        base_cfg = self._base_config()
        if base_cfg.get("family") != "sequence":
            raise ValueError(
                f"기반 모델이 시퀀스가 아니다: {base_cfg.get('name')} "
                f"(family={base_cfg.get('family')})"
            )
        self._members = []
        sizes = []
        for member in range(self._k):
            part = self._subsample(train, member)
            sizes.append({"n": len(part), "positive_rate": part.positive_rate})
            print(
                f"      [균형배깅 {member + 1}/{self._k}] 표본 {len(part):,}개 "
                f"(양성 {part.positive_rate:.2%})",
                flush=True,
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
            json.dump({"members": manifest, "base": self.params["base"]}, fh,
                      ensure_ascii=False)

    def _load_weights(self, directory: Path) -> None:
        from . import registry

        with (directory / "members.json").open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        base_cfg = registry.load_model_config(manifest["base"])
        cls = registry.resolve_class(base_cfg["class"])
        self._members = [cls.load(directory / name) for name in manifest["members"]]
