"""임곗값 선정 검증.

임곗값은 validation 에서만 고른다. 특히 fixed_disk_far 는 RODMAN 방식이라
분모가 미고장 '디스크'다. 행 단위 fixed_fpr 과 혼동하면 안 된다.
"""

from __future__ import annotations

import numpy as np
import pytest

from hddpred.inference import threshold as threshold_mod


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], [900, 100])
    score = np.concatenate(
        [rng.uniform(0.0, 0.6, 900), rng.uniform(0.4, 1.0, 100)]
    )
    return y, score


def test_max_f1_picks_a_threshold_inside_the_score_range(scores):
    y, score = scores
    record = threshold_mod.select(y, score, {"policy": "max_f1"})
    assert score.min() <= record["threshold"] <= score.max()
    assert record["selected_on"] == "validation"


def test_precision_floor_is_respected_when_reachable(scores):
    y, score = scores
    record = threshold_mod.select(
        y, score, {"policy": "precision_floor", "precision_floor": 0.5}
    )
    assert record["rationale"]["floor_met"] is True
    assert record["rationale"]["val_precision"] >= 0.5


def test_unreachable_precision_floor_is_reported_not_silently_met():
    """점수와 라벨이 무관하면 precision 이 양성 비율 근처를 못 벗어난다."""
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], [950, 50])
    score = rng.uniform(0.0, 1.0, 1000)  # 라벨과 독립
    record = threshold_mod.select(
        y, score, {"policy": "precision_floor", "precision_floor": 0.9}
    )
    assert record["rationale"]["floor_met"] is False


def test_fixed_fpr_stays_under_the_target(scores):
    y, score = scores
    record = threshold_mod.select(
        y, score, {"policy": "fixed_fpr", "fixed_fpr": 0.01}
    )
    predicted = score >= record["threshold"]
    observed = float((predicted & (y == 0)).sum() / (y == 0).sum())
    assert observed <= 0.01 + 1e-9


def test_single_class_validation_is_rejected():
    with pytest.raises(ValueError, match="한쪽 클래스"):
        threshold_mod.select(np.zeros(10, int), np.linspace(0, 1, 10), {})


# --------------------------------------------------------------------------
# RODMAN 방식
# --------------------------------------------------------------------------
def _linear_far(candidate: float) -> float:
    """임곗값이 오르면 단조 감소하는 가짜 FAR."""
    return float(max(0.0, 1.0 - candidate))


def test_fixed_disk_far_finds_the_lowest_threshold_meeting_the_target(scores):
    y, score = scores
    record = threshold_mod.select(
        y,
        score,
        {"policy": "fixed_disk_far", "fixed_disk_far": 0.04, "n_grid": 200},
        disk_far_fn=_linear_far,
    )
    assert record["rationale"]["achieved_disk_far"] <= 0.04
    # 목표를 만족하는 가장 낮은 임곗값이어야 recall 이 최대가 된다.
    assert record["threshold"] >= 0.96
    assert record["threshold"] < 0.99


def test_fixed_disk_far_needs_the_callback(scores):
    y, score = scores
    with pytest.raises(ValueError, match="disk_far_fn"):
        threshold_mod.select(y, score, {"policy": "fixed_disk_far"})


def test_fixed_disk_far_falls_back_when_the_target_is_unreachable(scores):
    y, score = scores
    record = threshold_mod.select(
        y,
        score,
        {"policy": "fixed_disk_far", "fixed_disk_far": 0.0, "n_grid": 50},
        # 어떤 임곗값에서도 목표를 못 맞추는 경우
        disk_far_fn=lambda candidate: 0.5,
    )
    assert record["rationale"]["achieved_disk_far"] == 0.5
    assert record["threshold"] == pytest.approx(score.max(), rel=1e-6)


def test_search_is_monotone_safe():
    """FAR 이 단조 감소한다는 가정 위에서 이분 탐색이 경계를 정확히 찾는다."""
    score = np.linspace(0.0, 1.0, 1001)
    step = lambda candidate: 0.10 if candidate < 0.7 else 0.01  # noqa: E731
    chosen, achieved = threshold_mod._search_disk_far(score, step, 0.05, 500)
    assert achieved == 0.01
    assert 0.69 <= chosen <= 0.71
