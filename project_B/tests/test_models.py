"""모델 인터페이스 검증.

평가 모듈은 모델이 무엇인지 모른 채 fit / predict_proba / save / load 만
호출한다. 그 계약이 실제로 지켜지는지 작은 합성 데이터로 확인한다.

새 모델을 추가하면 MODEL_CONFIGS 에 한 줄 넣으면 된다.
"""

from __future__ import annotations

import numpy as np
import pytest

from hddpred.features.fold import FoldMatrix, SequenceFold
from hddpred.models import registry

N_FEATURES = 6
LOOKBACK = 5

TABULAR_CONFIGS = [
    {
        "name": "xgboost",
        "family": "tabular",
        "class": "hddpred.models.trees.XGBoostModel",
        "params": {"n_estimators": 20, "max_depth": 3, "eval_metric": "aucpr"},
        "training": {"early_stopping_rounds": 5, "auto_scale_pos_weight": True},
    },
    {
        "name": "lightgbm",
        "family": "tabular",
        "class": "hddpred.models.trees.LightGBMModel",
        "params": {"n_estimators": 20, "num_leaves": 7, "verbose": -1},
        "training": {"early_stopping_rounds": 5, "auto_scale_pos_weight": True},
    },
]

SEQUENCE_CONFIGS = [
    {
        "name": "gru",
        "family": "sequence",
        "class": "hddpred.models.sequence.RNNModel",
        "params": {"cell": "gru", "hidden_size": 8, "num_layers": 1},
        "training": {"epochs": 1, "batch_size": 64, "amp": False, "auto_pos_weight": True},
    },
    {
        "name": "tcn",
        "family": "sequence",
        "class": "hddpred.models.sequence.TCNModel",
        "params": {"channels": [8, 8], "kernel_size": 2},
        "training": {"epochs": 1, "batch_size": 64, "amp": False},
    },
    {
        "name": "transformer",
        "family": "sequence",
        "class": "hddpred.models.sequence.TransformerModel",
        "params": {"d_model": 8, "nhead": 2, "num_layers": 1, "dim_feedforward": 16},
        "training": {"epochs": 1, "batch_size": 64, "amp": False},
    },
]


def _tabular(n: int, seed: int) -> FoldMatrix:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
    # 첫 두 피처가 양성을 만든다. 모델이 학습할 신호가 있어야 한다.
    y = ((X[:, 0] + X[:, 1]) > 1.5).astype(np.int8)
    return FoldMatrix(
        X=X,
        y=y,
        record_date=np.arange(n).astype("datetime64[D]"),
        columns=[f"f{i}" for i in range(N_FEATURES)],
    )


def _sequence(n: int, seed: int) -> SequenceFold:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
    end_index = np.arange(LOOKBACK - 1, n)
    y = (matrix[end_index, 0] > 0.8).astype(np.int8)
    return SequenceFold(
        matrix=matrix,
        end_index=end_index,
        lookback=LOOKBACK,
        y=y,
        record_date=end_index.astype("datetime64[D]"),
        columns=[f"f{i}" for i in range(N_FEATURES)],
    )


@pytest.mark.parametrize("model_cfg", TABULAR_CONFIGS, ids=lambda c: c["name"])
def test_tabular_model_round_trip(model_cfg, tmp_path):
    train, val = _tabular(2000, 0), _tabular(600, 1)
    model = registry.create(model_cfg, seed=42)
    info = model.fit(train, val)
    assert info["n_train"] == len(train)

    score = model.predict_proba(val)
    assert score.shape == (len(val),)
    assert np.isfinite(score).all()
    assert (score >= 0).all() and (score <= 1).all()

    model.save(tmp_path / model_cfg["name"])
    restored = type(model).load(tmp_path / model_cfg["name"])
    np.testing.assert_allclose(restored.predict_proba(val), score, rtol=1e-6)


@pytest.mark.parametrize("model_cfg", SEQUENCE_CONFIGS, ids=lambda c: c["name"])
def test_sequence_model_round_trip(model_cfg, tmp_path):
    train, val = _sequence(800, 0), _sequence(300, 1)
    model = registry.create(model_cfg, seed=42)
    info = model.fit(train, val)
    assert info["n_features"] == N_FEATURES
    assert info["lookback"] == LOOKBACK

    score = model.predict_proba(val)
    assert score.shape == (len(val),)
    assert np.isfinite(score).all()
    assert (score >= 0).all() and (score <= 1).all()

    model.save(tmp_path / model_cfg["name"])
    restored = type(model).load(tmp_path / model_cfg["name"])
    np.testing.assert_allclose(restored.predict_proba(val), score, rtol=1e-5, atol=1e-6)


def test_tabular_models_handle_missing_values():
    """segment 시작부의 diff/rolling 은 NaN 이다. 트리 모델은 그대로 받아야 한다."""
    train, val = _tabular(2000, 0), _tabular(600, 1)
    train.X[:50, 2] = np.nan
    val.X[:10, 2] = np.nan
    for model_cfg in TABULAR_CONFIGS:
        model = registry.create(model_cfg, seed=42)
        model.fit(train, val)
        assert np.isfinite(model.predict_proba(val)).all()


def test_positive_weight_matches_the_class_ratio():
    from hddpred.models.base import BaseModel

    y = np.array([0] * 90 + [1] * 10, dtype=np.int8)
    assert BaseModel.positive_weight(y) == pytest.approx(9.0)
