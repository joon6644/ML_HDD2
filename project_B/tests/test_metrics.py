"""평가 지표 검증.

디스크 단위 판정이 이 프로젝트의 핵심 정의라 네 칸을 하나씩 다 확인한다.

기본 규칙은 in_horizon 이다 (정답 구간 안에서 울리면 TP, 구간 밖 알람은
판정을 바꾸지 않음). on_time 은 구간 밖 알람을 FP 로 강등하는 더 엄격한
변형이고, or 는 시점을 아예 보지 않는 선행연구 표준 집계다.

    out        = horizon 밖(y=0 행) 알람 >= 1
    in         = horizon 안(y=1 행) 알람 >= 1
    has_window = 창 안에 y=1 행이 있는가

    out                       -> FP
    ~out &  in                -> TP
    ~out & ~in &  has_window  -> FN
    ~out & ~in & ~has_window  -> TN
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from hddpred.evaluation import metrics
from hddpred.evaluation.metrics import DiskLevelSpec

APRIL = (date(2023, 4, 1), date(2023, 4, 30))
DAY = pd.Timestamp("2023-04-01")


def _frame(rows):
    """rows: (serial, day_offset, y, score, failure_date)"""
    return pd.DataFrame(
        [
            (serial, DAY + pd.Timedelta(days=offset), y, score, failure)
            for serial, offset, y, score, failure in rows
        ],
        columns=metrics.PREDICTION_COLUMNS,
    )


def _on_time(**kwargs) -> DiskLevelSpec:
    return DiskLevelSpec(rule="on_time", period=APRIL, **kwargs)


def _or(**kwargs) -> DiskLevelSpec:
    return DiskLevelSpec(rule="or", period=APRIL, **kwargs)


def _outcome(frame, threshold=0.5, spec=None):
    spec = spec or _on_time()
    units = metrics.collapse_to_disks(frame, threshold, spec)
    codes = metrics.outcome_codes(units, spec)
    return dict(zip(units.index.get_level_values(0), codes))


# --------------------------------------------------------------------------
# 순위 점수 모델 (이상탐지)
# --------------------------------------------------------------------------
def test_brier_is_undefined_for_scores_outside_zero_one():
    """재구성 오차처럼 확률이 아닌 점수에는 Brier score 가 성립하지 않는다.

    억지로 정규화해 숫자를 만들면 모델 비교표에서 확률 예측과 나란히 놓여
    오독된다. NaN 으로 두고, 순위 기반 지표는 그대로 낸다.
    """
    frame = _frame(
        [
            ("A", 0, 1, 14.2, pd.Timestamp("2023-04-08")),
            ("B", 0, 0, 0.3, pd.NaT),
            ("C", 0, 0, 2.5, pd.NaT),
        ]
    )
    row = metrics.evaluate(
        frame, 1.0, {"disk_level": {"rule": "on_time"}}, horizon_days=10, period=APRIL
    )["row_level"]
    assert np.isnan(row["brier"])
    # 순위만 쓰는 지표는 정상적으로 나와야 한다.
    assert np.isfinite(row["pr_auc"]) and np.isfinite(row["roc_auc"])


def test_brier_still_reported_for_probability_models():
    frame = _frame(
        [
            ("A", 0, 1, 0.9, pd.Timestamp("2023-04-08")),
            ("B", 0, 0, 0.1, pd.NaT),
        ]
    )
    row = metrics.evaluate(
        frame, 0.5, {"disk_level": {"rule": "on_time"}}, horizon_days=10, period=APRIL
    )["row_level"]
    assert row["brier"] == pytest.approx(0.01)


# --------------------------------------------------------------------------
# rule: in_horizon — 선언한 horizon 이 판정을 지배한다
# --------------------------------------------------------------------------
def _in_horizon(**kwargs) -> DiskLevelSpec:
    return DiskLevelSpec(rule="in_horizon", period=APRIL, **kwargs)


def _codes(frame, threshold=0.5, spec=None):
    spec = spec or _in_horizon()
    units = metrics.collapse_to_disks(frame, threshold, spec)
    return dict(zip(units.index.get_level_values(0), metrics.outcome_codes(units, spec)))


def test_in_horizon_alarm_inside_the_window_is_a_true_positive():
    frame = _frame(
        [("A", 0, 0, 0.10, pd.Timestamp("2023-04-08")),
         ("A", 1, 1, 0.90, pd.Timestamp("2023-04-08"))]
    )
    assert _codes(frame)["A"] == metrics.TP


def test_in_horizon_ignores_an_alarm_outside_the_window():
    """구간 밖에서만 울리면 놓친 것이다. on_time 과 달리 FP 로 강등하지 않는다."""
    frame = _frame(
        [("A", 0, 0, 0.90, pd.Timestamp("2023-04-20")),   # horizon 밖 알람
         ("A", 1, 1, 0.10, pd.Timestamp("2023-04-20"))]   # horizon 안, 조용
    )
    assert _codes(frame)["A"] == metrics.FN
    # 같은 상황을 on_time 으로 보면 FP_EARLY 다.
    spec = DiskLevelSpec(rule="on_time", period=APRIL)
    assert _codes(frame, spec=spec)["A"] == metrics.FP_EARLY


def test_in_horizon_early_alarm_does_not_cancel_a_correct_one():
    """구간 밖에서도 울리고 안에서도 울렸으면 TP 다 (on_time 은 FP)."""
    frame = _frame(
        [("A", 0, 0, 0.90, pd.Timestamp("2023-04-20")),
         ("A", 1, 1, 0.90, pd.Timestamp("2023-04-20"))]
    )
    assert _codes(frame)["A"] == metrics.TP
    spec = DiskLevelSpec(rule="on_time", period=APRIL)
    assert _codes(frame, spec=spec)["A"] == metrics.FP_EARLY


def test_in_horizon_healthy_disk_that_alarms_is_a_false_positive():
    frame = _frame([("A", 0, 0, 0.90, pd.NaT), ("A", 1, 0, 0.10, pd.NaT)])
    assert _codes(frame)["A"] == metrics.FP_HEALTHY


def test_in_horizon_quiet_healthy_disk_is_a_true_negative():
    frame = _frame([("A", 0, 0, 0.10, pd.NaT), ("A", 1, 0, 0.20, pd.NaT)])
    assert _codes(frame)["A"] == metrics.TN


def test_in_horizon_rejects_a_window_shorter_than_the_horizon():
    """창이 horizon 보다 짧으면 창 안 모든 행이 y=1 이라 규칙이 무너진다."""
    with pytest.raises(ValueError, match="window_days"):
        DiskLevelSpec.from_config(
            {"disk_level": {"rule": "in_horizon", "window_days": 5}},
            horizon_days=10, period=APRIL,
        )


def test_in_horizon_is_the_default_rule():
    assert DiskLevelSpec().rule == "in_horizon"
    assert DiskLevelSpec.from_config({}, horizon_days=10, period=APRIL).rule == "in_horizon"


# --------------------------------------------------------------------------
# 네 칸
# --------------------------------------------------------------------------
def test_alarm_outside_the_horizon_is_a_false_positive():
    """구간 안에서도 울렸더라도 밖에서 한 번 울렸으면 FP 다."""
    frame = _frame(
        [
            ("A", 0, 0, 0.90, pd.Timestamp("2023-04-20")),  # 구간 밖 알람
            ("A", 11, 1, 0.90, pd.Timestamp("2023-04-20")),  # 구간 안 알람
        ]
    )
    assert _outcome(frame)["A"] == metrics.FP_EARLY


def test_alarm_only_inside_the_horizon_is_a_true_positive():
    frame = _frame(
        [
            ("A", 0, 0, 0.10, pd.Timestamp("2023-04-20")),
            ("A", 11, 1, 0.90, pd.Timestamp("2023-04-20")),
        ]
    )
    assert _outcome(frame)["A"] == metrics.TP


def test_no_alarm_with_a_window_is_a_false_negative():
    frame = _frame(
        [
            ("A", 0, 0, 0.10, pd.Timestamp("2023-04-20")),
            ("A", 11, 1, 0.10, pd.Timestamp("2023-04-20")),
        ]
    )
    assert _outcome(frame)["A"] == metrics.FN


def test_no_alarm_and_no_window_is_a_true_negative():
    frame = _frame([("A", 0, 0, 0.10, pd.NaT), ("A", 1, 0, 0.20, pd.NaT)])
    assert _outcome(frame)["A"] == metrics.TN


def test_healthy_disk_that_alarms_is_a_false_positive():
    frame = _frame([("A", 0, 0, 0.90, pd.NaT), ("A", 1, 0, 0.10, pd.NaT)])
    assert _outcome(frame)["A"] == metrics.FP_HEALTHY


def test_the_four_cells_partition_every_disk():
    """어떤 조합이 와도 정확히 한 칸에 들어가야 한다."""
    rng = np.random.default_rng(0)
    rows = []
    for index in range(200):
        serial = f"D{index:03d}"
        has_window = index % 3 == 0
        for offset in range(20):
            y = int(has_window and offset >= 12)
            rows.append((serial, offset, y, float(rng.uniform()), pd.NaT))
    units = metrics.collapse_to_disks(_frame(rows), 0.5, _on_time())
    codes = metrics.outcome_codes(units, _on_time())
    counts = metrics._counts_from_codes(codes)
    assert counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"] == len(units)
    assert set(np.unique(codes)) <= {
        metrics.TN,
        metrics.FP_HEALTHY,
        metrics.FP_EARLY,
        metrics.FN,
        metrics.TP,
    }


# --------------------------------------------------------------------------
# 지표
# --------------------------------------------------------------------------
def test_false_positives_split_into_early_and_healthy():
    frame = _frame(
        [
            ("early", 0, 0, 0.90, pd.Timestamp("2023-04-20")),  # 구간 있는데 밖에서 울림
            ("early", 11, 1, 0.10, pd.Timestamp("2023-04-20")),
            ("healthy", 0, 0, 0.90, pd.NaT),  # 구간 없는데 울림
        ]
    )
    result = metrics.disk_metrics(
        metrics.collapse_to_disks(frame, 0.5, _on_time()), _on_time()
    )
    assert result["fp_early"] == 1
    assert result["fp_healthy"] == 1
    assert result["fp"] == 2


def test_detection_rate_counts_early_disks_in_its_denominator():
    """2x2 의 recall 은 fp_early 를 분모에서 뺀다. detection_rate 는 넣는다."""
    frame = _frame(
        [
            ("tp", 11, 1, 0.90, pd.Timestamp("2023-04-20")),
            ("early", 0, 0, 0.90, pd.Timestamp("2023-04-20")),
            ("early", 11, 1, 0.10, pd.Timestamp("2023-04-20")),
            ("fn", 11, 1, 0.10, pd.Timestamp("2023-04-20")),
        ]
    )
    result = metrics.disk_metrics(
        metrics.collapse_to_disks(frame, 0.5, _on_time()), _on_time()
    )
    assert result["tp"] == 1 and result["fn"] == 1 and result["fp_early"] == 1
    assert result["recall"] == pytest.approx(0.5)  # 1 / (1 + 1)
    assert result["n_with_window"] == 3
    assert result["detection_rate"] == pytest.approx(1 / 3)


def test_far_uses_the_2x2_column_while_far_healthy_uses_healthy_disks():
    frame = _frame(
        [
            ("early", 0, 0, 0.90, pd.Timestamp("2023-04-20")),
            ("early", 11, 1, 0.10, pd.Timestamp("2023-04-20")),
            ("healthy_fp", 0, 0, 0.90, pd.NaT),
            ("healthy_tn", 0, 0, 0.10, pd.NaT),
        ]
    )
    result = metrics.disk_metrics(
        metrics.collapse_to_disks(frame, 0.5, _on_time()), _on_time()
    )
    # 2x2: fp=2, tn=1
    assert result["far"] == pytest.approx(2 / 3)
    # 미고장 분모: fp_healthy=1, healthy=2
    assert result["far_healthy"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# rule: or (선행연구 계열 B)
# --------------------------------------------------------------------------
def test_or_rule_ignores_when_the_alarm_fired():
    frame = _frame(
        [
            ("A", 0, 0, 0.90, pd.Timestamp("2023-04-20")),  # 구간 밖에서만 울림
            ("A", 11, 1, 0.10, pd.Timestamp("2023-04-20")),
        ]
    )
    assert _outcome(frame, spec=_or())["A"] == metrics.TP
    assert _outcome(frame, spec=_on_time())["A"] == metrics.FP_EARLY


def test_or_rule_far_denominator_is_healthy_disks():
    frame = _frame(
        [
            ("A", 0, 1, 0.90, pd.Timestamp("2023-04-05")),
            ("B", 0, 0, 0.90, pd.NaT),
            ("B", 1, 0, 0.90, pd.NaT),
            ("C", 0, 0, 0.10, pd.NaT),
        ]
    )
    result = metrics.disk_metrics(metrics.collapse_to_disks(frame, 0.5, _or()), _or())
    assert result["fp"] == 1 and result["tn"] == 1
    assert result["far"] == 0.5


def test_calendar_ground_truth_counts_a_next_month_failure_as_a_false_alarm():
    """RODMAN: 4월 29일에 울리고 5월 1일에 고장하면 4월 평가에서 FP 다."""
    frame = _frame(
        [
            ("A", 9, 1, 0.90, pd.Timestamp("2023-04-15")),
            ("B", 28, 1, 0.90, pd.Timestamp("2023-05-01")),
        ]
    )
    calendar = metrics.disk_metrics(
        metrics.collapse_to_disks(frame, 0.5, _or(ground_truth="calendar")),
        _or(ground_truth="calendar"),
    )
    assert calendar["tp"] == 1 and calendar["fp"] == 1

    label_or = metrics.disk_metrics(
        metrics.collapse_to_disks(frame, 0.5, _or()), _or()
    )
    assert label_or["tp"] == 2 and label_or["fp"] == 0


def test_calendar_ground_truth_requires_a_period():
    frame = _frame([("A", 0, 1, 0.9, pd.Timestamp("2023-04-05"))])
    spec = DiskLevelSpec(rule="or", ground_truth="calendar")
    with pytest.raises(ValueError, match="period"):
        metrics.collapse_to_disks(frame, 0.5, spec)


# --------------------------------------------------------------------------
# 알람 정의와 창
# --------------------------------------------------------------------------
def test_consecutive_aggregation_needs_a_run():
    frame = _frame(
        [
            ("A", 10, 1, 0.90, pd.Timestamp("2023-04-20")),
            ("A", 11, 1, 0.10, pd.Timestamp("2023-04-20")),
            ("A", 12, 1, 0.90, pd.Timestamp("2023-04-20")),
            ("B", 10, 1, 0.90, pd.Timestamp("2023-04-20")),
            ("B", 11, 1, 0.90, pd.Timestamp("2023-04-20")),
        ]
    )
    spec = _on_time(aggregation="consecutive", consecutive_k=2)
    outcome = _outcome(frame, spec=spec)
    assert outcome["A"] == metrics.FN  # 연속 2일이 없어 알람으로 안 침
    assert outcome["B"] == metrics.TP


def test_consecutive_run_resets_at_group_boundaries():
    alarm = np.array([1, 1, 1, 1], dtype=np.int8)
    new_group = np.array([True, False, True, False])
    assert metrics._consecutive_run(alarm, new_group).tolist() == [1, 2, 1, 2]


def test_window_days_splits_one_disk_into_several_units():
    rows = [("A", d, 0, 0.1, pd.NaT) for d in range(30)]
    units = metrics.collapse_to_disks(
        _frame(rows), 0.5, _or(window_days=10)
    )
    assert units.index.get_level_values("window").tolist() == [0, 1, 2]
    assert units["n_samples"].tolist() == [10, 10, 10]


def test_window_origin_is_the_period_start_not_the_first_observation():
    units = metrics.collapse_to_disks(
        _frame([("A", 14, 0, 0.1, pd.NaT)]), 0.5, _or(window_days=10)
    )
    assert units.index.get_level_values("window").tolist() == [1]


# --------------------------------------------------------------------------
# 설정 검증
# --------------------------------------------------------------------------
def test_on_time_rule_rejects_a_window_at_or_below_the_horizon():
    """W <= H 면 horizon 밖 행이 없어 규칙이 OR 로 붕괴한다."""
    with pytest.raises(ValueError, match="붕괴"):
        DiskLevelSpec.from_config(
            {"disk_level": {"rule": "on_time", "window_days": 10}},
            horizon_days=10,
            period=APRIL,
        )


def test_default_config_is_in_horizon_over_the_whole_period():
    """기본 판정은 in_horizon 이다.

    선언한 horizon 이 학습 라벨과 판정에 같이 적용되어야 "H일 예측"이라는
    선언과 채점이 일치한다. 창을 달력 한 달로 두고 아무 알람이나 인정하면
    실제 인정 구간이 1~28일로 흩어진다.
    """
    spec = DiskLevelSpec.from_config({}, horizon_days=10, period=APRIL)
    assert spec.rule == "in_horizon"
    assert spec.window_days is None


def test_unknown_rule_is_rejected():
    frame = _frame([("A", 0, 0, 0.1, pd.NaT)])
    spec = DiskLevelSpec(rule="nope", period=APRIL)
    with pytest.raises(ValueError, match="disk_level.rule"):
        metrics.outcome_codes(metrics.collapse_to_disks(frame, 0.5, _or()), spec)


# --------------------------------------------------------------------------
# 신뢰구간
# --------------------------------------------------------------------------
def test_ragged_indices_concatenates_slices():
    starts = np.array([0, 5, 5], dtype=np.int64)
    lengths = np.array([2, 3, 1], dtype=np.int64)
    assert metrics._ragged_indices(starts, lengths).tolist() == [0, 1, 5, 6, 7, 5]


def test_bootstrap_returns_intervals_around_the_point_estimate():
    rng = np.random.default_rng(0)
    rows = []
    for index in range(60):
        serial = f"D{index:03d}"
        failed = index % 6 == 0
        for offset in range(20):
            y = int(failed and offset >= 12)
            score = rng.uniform(0.6, 1.0) if y else rng.uniform(0.0, 0.4)
            rows.append((serial, offset, y, score, pd.NaT))
    frame = _frame(rows)
    cfg = {"disk_level": {"rule": "on_time", "window_days": None}}
    point = metrics.evaluate(frame, 0.5, cfg, horizon_days=8, period=APRIL)
    ci = metrics.bootstrap_ci(
        frame,
        0.5,
        cfg,
        {"enabled": True, "n_resamples": 100, "unit": "disk", "ci": 0.95, "seed": 0},
        horizon_days=8,
        period=APRIL,
    )
    interval = ci["disk_level.recall"]
    assert interval["lower"] <= point["disk_level"]["recall"] <= interval["upper"]
    assert point["disk_level_spec"]["rule"] == "on_time"
