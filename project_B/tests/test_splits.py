"""forward chaining split 의 날짜 경계 검증."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from hddpred.splits import forward

MONTHS = [f"{year}-{month:02d}" for year in (2021, 2022, 2023) for month in range(1, 13)]
BASE_CFG = {
    "protocol": "forward_chaining",
    "test_months": 1,
    "val_months": 1,
    "train_window": "expanding",
    "train_months": 12,
    "min_train_months": 12,
    "n_folds": 3,
    "min_test_failures": 1,
    "min_val_failures": 1,
}


def test_folds_are_ordered_and_use_the_last_months():
    folds = forward.plan_folds(MONTHS, BASE_CFG, embargo_days=10)
    assert [f.test_month for f in folds] == ["2023-10", "2023-11", "2023-12"]
    assert [f.fold for f in folds] == [0, 1, 2]


def test_windows_do_not_overlap_and_move_forward():
    folds = forward.plan_folds(MONTHS, BASE_CFG, embargo_days=10)
    for fold in folds:
        train_start, train_end = fold.window("train")
        val_start, val_end = fold.window("val")
        test_start, test_end = fold.window("test")
        assert train_start < train_end < val_start < val_end < test_start < test_end


def test_test_window_is_exactly_one_calendar_month():
    folds = forward.plan_folds(MONTHS, BASE_CFG, embargo_days=10)
    for fold in folds:
        start, end = fold.window("test")
        assert start.day == 1
        assert (end + timedelta(days=1)).day == 1


def test_expanding_window_keeps_the_same_start():
    folds = forward.plan_folds(MONTHS, BASE_CFG, embargo_days=10)
    starts = {fold.train_start for fold in folds}
    assert starts == {date(2021, 1, 1).isoformat()}


def test_rolling_window_moves_the_start():
    cfg = dict(BASE_CFG, train_window="rolling", train_months=12)
    folds = forward.plan_folds(MONTHS, cfg, embargo_days=10)
    starts = [fold.train_start for fold in folds]
    assert len(set(starts)) == len(starts)
    assert starts == sorted(starts)


def test_short_history_folds_are_skipped():
    cfg = dict(BASE_CFG, n_folds=None, min_train_months=30)
    folds = forward.plan_folds(MONTHS, cfg, embargo_days=10)
    assert all(f.test_month >= "2023-08" for f in folds)


def test_rolling_window_of_exactly_min_train_months_survives():
    """RODMAN 재현(rolling 3개월, min 3개월)이 걸러지지 않아야 한다.

    개월 수를 일수/30.44 로 재면 3개월 창이 2.99 로 나와 전부 사라진다.
    """
    cfg = dict(BASE_CFG, train_window="rolling", train_months=3, min_train_months=3)
    folds = forward.plan_folds(MONTHS, cfg, embargo_days=30)
    assert len(folds) == BASE_CFG["n_folds"]
    for fold in folds:
        train_start, train_end = fold.window("train")
        assert (train_end - train_start).days >= 88


def test_incomplete_tail_month_is_excluded():
    """관측이 2023-12-31 에 끝나면 12월은 라벨이 잘려 test 로 쓸 수 없다."""
    folds = forward.plan_folds(
        MONTHS,
        BASE_CFG,
        embargo_days=10,
        data_end=date(2023, 12, 31),
        horizon_days=10,
    )
    assert [f.test_month for f in folds] == ["2023-09", "2023-10", "2023-11"]


def test_tail_exclusion_can_be_turned_off():
    cfg = dict(BASE_CFG, exclude_incomplete_tail=False)
    folds = forward.plan_folds(
        MONTHS, cfg, embargo_days=10, data_end=date(2023, 12, 31), horizon_days=10
    )
    assert folds[-1].test_month == "2023-12"


def test_multi_month_test_window_is_rejected():
    with pytest.raises(NotImplementedError):
        forward.plan_folds(MONTHS, dict(BASE_CFG, test_months=3), embargo_days=10)
