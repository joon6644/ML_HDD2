"""오토인코더 이상탐지 검증.

  정상 개체만 학습한다 (그리고 그 선별이 미래를 보지 않는다)
  점수는 재구성 오차다
  결측 칸은 손실과 점수에서 빠진다
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from hddpred import config as cfg_mod, paths
from hddpred.features.fold import FoldMatrix
from hddpred.models import registry

WINDOW = (date(2020, 1, 1), date(2020, 1, 31))
AE_CFG = {
    "name": "autoencoder",
    "family": "tabular",
    "class": "hddpred.models.autoencoder.AutoEncoderModel",
    "params": {"hidden_dims": [8], "latent_dim": 3, "batch_norm": False},
    "training": {"epochs": 2, "batch_size": 64, "early_stopping_patience": 2},
}


N_FEATURES = 6
# 정상 다양체를 정의하는 기저. seed 와 무관하게 고정해야 train / val / probe 가
# 같은 분포에서 나온다. seed 마다 기저가 바뀌면 val 이 다른 분포가 되어버려서
# 오토인코더가 학습에 실패한 것처럼 보인다.
BASIS = np.random.default_rng(12345).normal(size=(2, N_FEATURES))


def _matrix(n=400, n_features=N_FEATURES, *, failures=None, seed=0, window=WINDOW):
    """정상 행은 저차원 구조를 갖게 만든다. 오토인코더가 배울 것이 있어야 한다."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 2))
    basis = BASIS[:, :n_features]
    X = (latent @ basis + rng.normal(scale=0.01, size=(n, n_features))).astype(np.float32)

    failure_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[D]")
    y = np.zeros(n, np.int8)
    for index, when in (failures or {}).items():
        failure_date[index] = np.datetime64(when)
        if np.datetime64(when) <= np.datetime64(window[1]):
            y[index] = 1
    return FoldMatrix(
        X=X,
        y=y,
        record_date=np.full(n, np.datetime64(window[0]), dtype="datetime64[D]"),
        serial=np.array([f"D{i:04d}" for i in range(n)]),
        failure_date=failure_date,
        columns=[f"f{i}" for i in range(n_features)],
        window=window,
    )


# --------------------------------------------------------------------------
# 정상 개체 선별
# --------------------------------------------------------------------------
def test_healthy_mask_excludes_failures_confirmed_inside_the_window():
    matrix = _matrix(failures={3: "2020-01-20", 7: "2020-01-05"})
    healthy = matrix.healthy_mask()
    assert not healthy[3] and not healthy[7]
    assert healthy.sum() == len(matrix) - 2


def test_healthy_mask_does_not_look_past_the_window_end():
    """창이 1월에 끝나는데 3월 고장을 미리 빼면 미래를 본 것이다."""
    matrix = _matrix(failures={5: "2020-03-15"})
    assert matrix.healthy_mask()[5]


def test_healthy_mask_needs_failure_date():
    matrix = _matrix()
    matrix.failure_date = None
    with pytest.raises(ValueError, match="failure_date"):
        matrix.healthy_mask()


def test_fit_uses_only_healthy_rows():
    train = _matrix(failures={i: "2020-01-15" for i in range(10)})
    val = _matrix(n=200, seed=1, failures={0: "2020-01-15"})
    model = registry.create(AE_CFG, seed=42)
    info = model.fit(train, val)
    assert info["n_train"] == len(train)
    assert info["n_train_healthy"] == len(train) - 10


def test_fit_refuses_a_window_with_no_healthy_rows():
    train = _matrix(n=20, failures={i: "2020-01-15" for i in range(20)})
    val = _matrix(n=50, seed=1, failures={0: "2020-01-15"})
    model = registry.create(AE_CFG, seed=42)
    with pytest.raises(ValueError, match="정상 개체"):
        model.fit(train, val)


# --------------------------------------------------------------------------
# 점수
# --------------------------------------------------------------------------
def test_score_is_higher_for_rows_off_the_normal_manifold():
    train = _matrix(n=2000)
    val = _matrix(n=400, seed=1, failures={0: "2020-01-15"})
    model = registry.create(dict(AE_CFG, training=dict(AE_CFG["training"], epochs=40)), 42)
    model.fit(train, val)

    probe = _matrix(n=200, seed=2)
    normal_score = model.predict_proba(probe)
    # 정상 다양체에서 벗어난 행을 섞는다. 2차원 구조를 깨는 잡음이다.
    rng = np.random.default_rng(99)
    probe.X[:50] += rng.normal(scale=3.0, size=(50, probe.X.shape[1])).astype(np.float32)
    mixed = model.predict_proba(probe)
    assert mixed[:50].mean() > normal_score[:50].mean() * 5
    # 건드리지 않은 행의 점수는 그대로여야 한다.
    assert np.allclose(mixed[50:], normal_score[50:])


def test_missing_cells_do_not_create_anomalies():
    """관측 공백으로 결측이 생긴 행이 그 이유만으로 이상 판정되면 안 된다."""
    train = _matrix(n=1000)
    val = _matrix(n=300, seed=1, failures={0: "2020-01-15"})
    model = registry.create(AE_CFG, seed=42)
    model.fit(train, val)

    probe = _matrix(n=100, seed=3)
    baseline = model.predict_proba(probe)
    probe.X[:20, 0] = np.nan
    with_missing = model.predict_proba(probe)
    assert np.isfinite(with_missing).all()
    # 결측 칸이 손실에서 빠지므로 남은 칸의 오차와 같은 규모여야 한다.
    assert with_missing[:20].mean() < baseline.mean() * 5


# --------------------------------------------------------------------------
# 저장/복원
# --------------------------------------------------------------------------
def test_save_load_roundtrip(tmp_path):
    train = _matrix(n=500)
    val = _matrix(n=200, seed=1, failures={0: "2020-01-15"})
    model = registry.create(AE_CFG, seed=42)
    model.fit(train, val)

    probe = _matrix(n=100, seed=4)
    before = model.predict_proba(probe)
    model.save(tmp_path / "model")

    restored = registry.resolve_class(AE_CFG["class"]).load(tmp_path / "model")
    assert np.allclose(before, restored.predict_proba(probe), atol=1e-6)


def test_complexity_counts_parameters():
    train = _matrix(n=300)
    val = _matrix(n=100, seed=1, failures={0: "2020-01-15"})
    model = registry.create(AE_CFG, seed=42)
    assert model.complexity() == 0
    model.fit(train, val)
    assert model.complexity() > 0


# --------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------
def test_experiment_config_matches_the_supervised_protocol():
    """fold 집합이 같아야 monthly_expanding 과 나란히 놓을 수 있다."""
    ae = cfg_mod.load_yaml(paths.CONFIG_DIR / "experiments" / "monthly_autoencoder.yaml")
    supervised = cfg_mod.load_yaml(
        paths.CONFIG_DIR / "experiments" / "monthly_expanding.yaml"
    )
    split_keys = [k for k in supervised["overrides"] if k.startswith("split.")]
    label_keys = [k for k in supervised["overrides"] if k.startswith("labeling.")]
    for key in split_keys + label_keys:
        assert ae["overrides"][key] == supervised["overrides"][key], key
    assert ae["warm_start"] is False
    # 양성 비율을 맞추는 표집은 이 모델에 의미가 없다.
    assert ae["overrides"]["features.train_sampling.strategy"] == "stride"


def test_model_config_bottleneck_is_narrower_than_the_input():
    cfg = cfg_mod.load_yaml(paths.CONFIG_DIR / "models" / "autoencoder.yaml")
    assert cfg["params"]["latent_dim"] < min(cfg["params"]["hidden_dims"])
    assert cfg["training"]["early_stopping_metric"] in ("val_loss", "val_pr_auc")
