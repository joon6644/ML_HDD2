"""트리 모델 (XGBoost, LightGBM).

입력은 FoldMatrix. 두 라이브러리 모두 결측을 그대로 처리하므로 segment 시작부
diff/rolling 의 NaN 을 채우지 않고 넘긴다. 임의의 0으로 채우면 "변화가 없었다"
는 잘못된 신호가 된다.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from .base import BaseModel


class XGBoostModel(BaseModel):
    family = "tabular"
    name = "xgboost"

    def __init__(self, params: dict, training: dict, seed: int = 42):
        super().__init__(params, training, seed)
        self._model = None

    def fit(self, train, val) -> dict:
        from xgboost import XGBClassifier

        params = dict(self.params)
        if self.training.get("auto_scale_pos_weight", False):
            params["scale_pos_weight"] = self.positive_weight(train.y)

        self._model = XGBClassifier(
            **params,
            random_state=self.seed,
            early_stopping_rounds=self.training.get("early_stopping_rounds"),
        )
        self._model.fit(
            train.X,
            train.y,
            eval_set=[(val.X, val.y)],
            verbose=False,
        )
        self.fit_info = {
            "feature_names": list(train.columns),
            "best_iteration": int(getattr(self._model, "best_iteration", -1)),
            "best_score": float(getattr(self._model, "best_score", float("nan"))),
            "scale_pos_weight": float(params.get("scale_pos_weight", 1.0)),
            "n_train": len(train),
            "n_val": len(val),
            "train_positive_rate": train.positive_rate,
        }
        return self.fit_info

    def predict_proba(self, data) -> np.ndarray:
        return self._model.predict_proba(data.X)[:, 1].astype(np.float64)

    def complexity(self) -> int:
        best = self.fit_info.get("best_iteration", 0)
        return int(best if best and best > 0 else self.params.get("n_estimators", 0))

    def _save_weights(self, directory: Path) -> None:
        self._model.save_model(str(directory / "model.ubj"))

    def _load_weights(self, directory: Path) -> None:
        from xgboost import XGBClassifier

        self._model = XGBClassifier()
        self._model.load_model(str(directory / "model.ubj"))


class LightGBMModel(BaseModel):
    family = "tabular"
    name = "lightgbm"

    def __init__(self, params: dict, training: dict, seed: int = 42):
        super().__init__(params, training, seed)
        self._model = None

    def fit(self, train, val) -> dict:
        import lightgbm as lgb

        params = dict(self.params)
        if self.training.get("auto_scale_pos_weight", False):
            params["scale_pos_weight"] = self.positive_weight(train.y)

        callbacks = []
        rounds = self.training.get("early_stopping_rounds")
        if rounds:
            callbacks.append(lgb.early_stopping(int(rounds), verbose=False))

        # feature_name 을 넘기지 않는다. 입력이 numpy 배열이라 predict 때마다
        # "X does not have valid feature names" 경고가 난다. 이름은 fit_info 에
        # 남겨 두고 중요도 해석에 쓴다.
        self._model = lgb.LGBMClassifier(**params, random_state=self.seed)
        self._model.fit(
            train.X,
            train.y,
            eval_set=[(val.X, val.y)],
            eval_metric=params.get("metric", "average_precision"),
            callbacks=callbacks,
        )
        self.fit_info = {
            "feature_names": list(train.columns),
            "best_iteration": int(self._model.best_iteration_ or 0),
            "scale_pos_weight": float(params.get("scale_pos_weight", 1.0)),
            "n_train": len(train),
            "n_val": len(val),
            "train_positive_rate": train.positive_rate,
        }
        return self.fit_info

    def predict_proba(self, data) -> np.ndarray:
        # LightGBM 은 numpy 로 학습해도 feature_names_in_ 을 Column_N 으로
        # 채워두기 때문에, numpy 로 예측하면 sklearn 이 매번 경고를 낸다.
        # 라이브러리 쪽 사정이고 결과에는 영향이 없어 여기서만 막는다.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="X does not have valid feature names"
            )
            return self._model.predict_proba(data.X)[:, 1].astype(np.float64)

    def complexity(self) -> int:
        best = self.fit_info.get("best_iteration", 0)
        return int(best or self.params.get("n_estimators", 0))

    def _save_weights(self, directory: Path) -> None:
        self._model.booster_.save_model(str(directory / "model.txt"))
        with (directory / "feature_names.json").open("w", encoding="utf-8") as fh:
            json.dump(self.fit_info.get("feature_names", []), fh, ensure_ascii=False)

    def _load_weights(self, directory: Path) -> None:
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(directory / "model.txt"))
        self._model = _BoosterAdapter(booster)


class _BoosterAdapter:
    """저장된 Booster 를 sklearn 인터페이스처럼 쓰기 위한 얇은 wrapper."""

    def __init__(self, booster):
        self.booster = booster

    def predict_proba(self, X):
        positive = self.booster.predict(X)
        return np.column_stack([1.0 - positive, positive])


class RandomForestModel(BaseModel):
    """sklearn RandomForest. 부스팅 계열과 달리 배깅이라 과적합 경향이 다르다.

    sklearn 1.4+ 는 결측을 그대로 처리하므로 트리 계열과 같은 입력을 쓴다.
    조기 종료가 없어 val 은 임곗값 선정에만 쓰인다.
    """

    family = "tabular"
    name = "randomforest"

    def __init__(self, params: dict, training: dict, seed: int = 42):
        super().__init__(params, training, seed)
        self._model = None

    def fit(self, train, val) -> dict:
        from sklearn.ensemble import RandomForestClassifier

        params = dict(self.params)
        if self.training.get("auto_class_weight", False):
            params["class_weight"] = "balanced_subsample"

        self._model = RandomForestClassifier(**params, random_state=self.seed)
        self._model.fit(train.X, train.y)
        self.fit_info = {
            "feature_names": list(train.columns),
            "n_estimators": int(self._model.n_estimators),
            "class_weight": str(params.get("class_weight")),
            "n_train": len(train),
            "n_val": len(val),
            "train_positive_rate": train.positive_rate,
        }
        return self.fit_info

    def predict_proba(self, data) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            return self._model.predict_proba(data.X)[:, 1].astype(np.float64)

    def complexity(self) -> int:
        if self._model is None:
            return 0
        return int(sum(tree.tree_.node_count for tree in self._model.estimators_))

    def _save_weights(self, directory: Path) -> None:
        import joblib

        joblib.dump(self._model, directory / "model.joblib", compress=3)

    def _load_weights(self, directory: Path) -> None:
        import joblib

        self._model = joblib.load(directory / "model.joblib")


class CatBoostModel(BaseModel):
    """CatBoost. 대칭 트리(oblivious tree)라 부스팅 계열 안에서도 구조가 다르다."""

    family = "tabular"
    name = "catboost"

    def __init__(self, params: dict, training: dict, seed: int = 42):
        super().__init__(params, training, seed)
        self._model = None

    def fit(self, train, val) -> dict:
        from catboost import CatBoostClassifier

        params = dict(self.params)
        # scale_pos_weight 는 쓰지 않는다. 확률이 포화되어 임곗값 선택이
        # 무의미해지는 것을 lightgbm 에서 실측했다 (configs/models/*.yaml 참고).
        self._model = CatBoostClassifier(
            **params,
            random_seed=self.seed,
            early_stopping_rounds=self.training.get("early_stopping_rounds"),
        )
        self._model.fit(train.X, train.y, eval_set=(val.X, val.y), verbose=False)
        self.fit_info = {
            "feature_names": list(train.columns),
            "best_iteration": int(self._model.get_best_iteration() or 0),
            "n_train": len(train),
            "n_val": len(val),
            "train_positive_rate": train.positive_rate,
        }
        return self.fit_info

    def predict_proba(self, data) -> np.ndarray:
        return self._model.predict_proba(data.X)[:, 1].astype(np.float64)

    def complexity(self) -> int:
        best = self.fit_info.get("best_iteration", 0)
        return int(best or self.params.get("iterations", 0))

    def _save_weights(self, directory: Path) -> None:
        self._model.save_model(str(directory / "model.cbm"))

    def _load_weights(self, directory: Path) -> None:
        from catboost import CatBoostClassifier

        self._model = CatBoostClassifier()
        self._model.load_model(str(directory / "model.cbm"))
