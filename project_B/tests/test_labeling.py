"""라벨 정의 검증.

이 파이프라인에서 가장 조용히 틀리기 쉬운 부분이다. 관측 종료 직전 구간을
음성으로 넣으면 성능이 좋아 보이고, 그 사실이 지표에 드러나지 않는다.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from conftest import make_canonical, read_labels

from hddpred.labeling import horizon_labels

HORIZON = 10
BASE_CFG = {
    "horizon_days": HORIZON,
    "exclude_post_failure_rows": True,
    "require_horizon_observability": True,
    "min_history_days": 0,
}


@pytest.fixture
def labels(derived_root, preprocessing_cfg):
    canonical = make_canonical(
        derived_root,
        {
            # 고장 디스크: 2020-03-31 에 고장
            "A": {
                "start": date(2020, 1, 1),
                "end": date(2020, 3, 31),
                "failure_date": date(2020, 3, 31),
            },
            # 미고장 디스크: 끝까지 생존
            "B": {"start": date(2020, 1, 1), "end": date(2020, 3, 31)},
        },
    )
    path = horizon_labels.build("TEST_DRIVE", canonical, BASE_CFG, preprocessing_cfg)
    return read_labels(path)


def test_post_failure_rows_are_excluded(labels):
    """고장 당일과 그 이후는 표본이 아니다."""
    a = labels[labels["serial_number"] == "A"]
    assert a["record_date"].max() == pd.Timestamp("2020-03-30")


def test_positive_window_is_exactly_horizon_days(labels):
    """(t, t + 10] 안에 고장이 있는 행만 양성이다."""
    a = labels[labels["serial_number"] == "A"]
    positives = a.loc[a["y"] == 1, "record_date"]
    assert len(positives) == HORIZON
    assert positives.min() == pd.Timestamp("2020-03-21")
    assert positives.max() == pd.Timestamp("2020-03-30")


def test_censored_tail_is_dropped(labels):
    """미고장 디스크의 마지막 H일은 '음성'이 아니라 '확인 불가'다."""
    b = labels[labels["serial_number"] == "B"]
    assert b["y"].sum() == 0
    # last_date(2020-03-31) - 10일 = 2020-03-21 까지만 생존을 확인할 수 있다.
    assert b["record_date"].max() == pd.Timestamp("2020-03-21")


def test_disabling_observability_creates_fake_negatives(derived_root, preprocessing_cfg):
    """반례. 검열 처리를 끄면 확인 불가 구간이 음성으로 섞인다."""
    canonical = make_canonical(
        derived_root,
        {"B": {"start": date(2020, 1, 1), "end": date(2020, 3, 31)}},
    )
    cfg = dict(BASE_CFG, require_horizon_observability=False)
    frame = read_labels(
        horizon_labels.build("TEST_DRIVE", canonical, cfg, preprocessing_cfg)
    )
    assert frame["record_date"].max() == pd.Timestamp("2020-03-31")
    assert (frame["y"] == 0).all()


def test_min_history_days_trims_the_start(derived_root, preprocessing_cfg):
    canonical = make_canonical(
        derived_root,
        {
            "A": {
                "start": date(2020, 1, 1),
                "end": date(2020, 3, 31),
                "failure_date": date(2020, 3, 31),
            }
        },
    )
    cfg = dict(BASE_CFG, min_history_days=30)
    frame = read_labels(
        horizon_labels.build("TEST_DRIVE", canonical, cfg, preprocessing_cfg)
    )
    assert frame["record_date"].min() == pd.Timestamp("2020-01-31")


def test_labels_never_look_past_the_horizon(labels):
    """양성 라벨의 days_to_failure 는 항상 (0, H] 안에 있다."""
    positives = labels[labels["y"] == 1]
    assert positives["days_to_failure"].min() > 0
    assert positives["days_to_failure"].max() <= HORIZON
