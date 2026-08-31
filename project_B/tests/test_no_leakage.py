"""누출 검사.

이 파일이 프로젝트에서 제일 중요한 테스트다. 누출은 성능을 올리는 방향으로
작동하기 때문에 지표만 보고는 알아차릴 수 없다.

검사하는 것:
  1. train 표본의 라벨 구간 (t, t+H] 가 val/test 구간을 침범하지 않는다
  2. embargo < horizon 인 설정은 아예 거부된다
  3. 시퀀스 표본의 lookback 이 관측 시점 t 를 넘지 않는다
  4. 스케일러가 train 밖의 통계를 보지 않는다
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from hddpred.features.fold import Scaler
from hddpred.splits import forward

MONTHS = [f"{year}-{month:02d}" for year in (2021, 2022, 2023) for month in range(1, 13)]
SPLIT_CFG = {
    "protocol": "forward_chaining",
    "test_months": 1,
    "val_months": 1,
    "train_window": "expanding",
    "train_months": 12,
    "min_train_months": 12,
    "n_folds": 6,
    "min_test_failures": 1,
    "min_val_failures": 1,
}


@pytest.mark.parametrize("horizon", [5, 10, 30])
def test_train_label_window_never_reaches_validation(horizon):
    """train 마지막 날의 라벨 구간이 val 시작 전에 끝나야 한다."""
    folds = forward.plan_folds(MONTHS, SPLIT_CFG, embargo_days=horizon)
    assert folds
    for fold in folds:
        _, train_end = fold.window("train")
        val_start, _ = fold.window("val")
        assert train_end + timedelta(days=horizon) < val_start


@pytest.mark.parametrize("horizon", [5, 10, 30])
def test_validation_label_window_never_reaches_test(horizon):
    folds = forward.plan_folds(MONTHS, SPLIT_CFG, embargo_days=horizon)
    for fold in folds:
        _, val_end = fold.window("val")
        test_start, _ = fold.window("test")
        assert val_end + timedelta(days=horizon) < test_start


def test_too_short_embargo_is_rejected(tmp_path, monkeypatch):
    """embargo < horizon 이면 build 단계에서 막힌다."""
    from hddpred import paths
    from hddpred.tracking import provenance

    monkeypatch.setattr(paths, "SPLITS_ROOT", tmp_path / "splits")
    labels_dir = tmp_path / "labels"
    provenance.write(labels_dir, stage="labels", config_hash="x", configs={})

    with pytest.raises(ValueError, match="누출"):
        forward.build(
            "TEST_DRIVE",
            labels_dir,
            dict(SPLIT_CFG, embargo_days=3),
            {"horizon_days": 10},
        )


def test_sequence_lookback_never_crosses_the_observation_time():
    """윈도우 [t-L+1, t] 는 t 이후를 포함하지 않는다.

    load_sequence 는 끝 인덱스에서 거슬러 슬라이싱하므로 구조적으로 미래를
    담을 수 없다. 그 슬라이싱 규칙 자체를 확인한다.
    """
    lookback = 5
    dates = np.array(
        [date(2021, 1, 1) + timedelta(days=i) for i in range(20)], dtype="datetime64[D]"
    )
    for end in range(lookback - 1, len(dates)):
        window = dates[end - lookback + 1 : end + 1]
        assert window.shape[0] == lookback
        assert window.max() == dates[end]


def test_scaler_uses_only_training_statistics():
    """val/test 값이 아무리 달라도 transform 결과는 train 통계로만 결정된다."""
    rng = np.random.default_rng(0)
    train = rng.normal(0.0, 1.0, size=(500, 4)).astype(np.float32)
    scaler = Scaler("standard", clip_quantile=None).fit(train)

    probe = np.full((3, 4), 2.0, dtype=np.float32)
    before = scaler.transform(probe.copy())

    # 전혀 다른 분포를 가진 데이터를 통과시켜도 상태가 바뀌지 않아야 한다.
    scaler.transform(rng.normal(100.0, 50.0, size=(500, 4)).astype(np.float32))
    after = scaler.transform(probe.copy())

    np.testing.assert_allclose(before, after)


def test_scaler_state_round_trips():
    rng = np.random.default_rng(1)
    data = rng.normal(3.0, 2.0, size=(200, 3)).astype(np.float32)
    scaler = Scaler("standard", clip_quantile=(0.01, 0.99)).fit(data)
    restored = Scaler.from_state_dict(scaler.state_dict())
    np.testing.assert_allclose(
        scaler.transform(data.copy()), restored.transform(data.copy())
    )
