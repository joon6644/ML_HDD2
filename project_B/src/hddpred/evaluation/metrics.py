"""평가 지표.

두 층으로 잰다.

1) row 단위 (HDD-일 표본)
   학습에 쓴 라벨을 그대로 채점한다. 추가 설정이 없는 가장 기본형이다.

2) 디스크 단위 판정 (rule)
   한 달의 평가 구간 안에서 각 디스크의 일별 예측을 혼동행렬 한 칸으로 접는다.

   rule = "in_horizon" (기본)

       고장 H일 전 구간(y=1 행) 안에서 울렸으면 TP, 안 울렸으면 FN.
       구간 밖 알람은 판정을 바꾸지 않는다. 미고장 디스크는 아무 때나
       울리면 FP 다.

       선언한 horizon 이 학습 라벨과 판정에 같이 적용되므로 "10일 예측"
       이라는 선언과 채점이 일치한다. 창을 달력 한 달로 두고 아무 알람이나
       인정하면 실제 인정 구간이 1~28일로 흩어진다 (실측).

   rule = "on_time"

       out        = horizon 밖(y=0 행) 알람 >= 1
       in         = horizon 안(y=1 행) 알람 >= 1
       has_window = 창 안에 y=1 행이 있는가 (정답 구간의 존재)

       out                       -> FP   구간 밖에서 울렸다
       ~out &  in                -> TP   구간 안에서만 울렸다
       ~out & ~in &  has_window  -> FN   구간이 있는데 안 울렸다
       ~out & ~in & ~has_window  -> TN   구간도 없고 안 울렸다

       ~out & ~in 은 곧 "알람 없음"이므로 네 칸이 전체를 빈틈없이 나눈다.
       y=1 이 곧 "고장 H일 이내"이므로 이 규칙은 다음과 같다.
           TP = (행 단위 TP >= 1) AND (행 단위 FP == 0)

       FP 는 fp_early(정답 구간이 있는 디스크)와 fp_healthy(없는 디스크)로
       나눠 기록한다. 2x2 의 recall 분모(TP+FN)에는 fp_early 가 빠지므로
       detection_rate = TP / (TP + FN + fp_early) 도 함께 낸다.

   rule = "or"
       창 안에 알람이 하나라도 있으면 양성. 선행연구 계열 B의 표준 집계다.
         Murray et al. (2005)    multiple-instance 정식화 (이력 전체)
         LightGBM + CID (2021)   Algorithm 1, threshold 1 (10일 창)
         LSTM specificity (2020) "If there are alarm records in the sequence"

설정 축:

  window_days — 집계 창 W. None 이면 평가 구간 전체가 한 창이다(기본).
      rule = "on_time" 에서 W <= H 로 두면 창 안 모든 행이 고장 H일 이내가
      되어 horizon 밖 알람이 존재할 수 없고 규칙이 단순 OR 로 붕괴한다.
      그래서 그 조합은 from_config 에서 막는다.

  ground_truth — rule = "or" 에서 정답을 접는 규칙.
      calendar : 그 창 안에 실제 고장일이 있는 디스크만 양성. RODMAN 정의.
                 "the total number of actual disk failures in one-month
                 testing". 창 밖 고장을 미리 맞힌 알람은 FP 가 된다.
      label_or : 창 안 표본의 양성 라벨을 OR 한다.
      rule = "on_time" 에서는 정답 구간이 라벨로 정의되므로 쓰지 않는다.

days_in_advance 는 기술 통계로만 남기고 모델 선정에는 쓰지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)

PREDICTION_COLUMNS = ["serial_number", "record_date", "y", "score", "failure_date"]


def _brier(y_true: np.ndarray, score: np.ndarray) -> float:
    """확률 눈금을 전제하는 지표라 순위 점수 모델에는 정의되지 않는다.

    이상탐지 모델(오토인코더)의 점수는 재구성 오차라서 [0, 1] 밖으로 나간다.
    순위만 의미가 있으므로 PR-AUC / ROC-AUC / 임곗값 선정은 그대로 성립하지만
    Brier score 는 성립하지 않는다. 억지로 정규화해서 숫자를 만들어 내면
    모델 간 비교표에서 확률 예측과 나란히 놓여 오독된다. NaN 으로 둔다.
    """
    if score.size == 0 or score.min() < 0.0 or score.max() > 1.0:
        return float("nan")
    return float(brier_score_loss(y_true, score))


@dataclass(frozen=True)
class DiskLevelSpec:
    """디스크 단위 집계 규칙."""

    rule: str = "in_horizon"  # in_horizon | on_time | or
    aggregation: str = "any"  # any | consecutive
    consecutive_k: int = 2
    window_days: int | None = None  # None -> 평가 구간 전체가 한 창
    ground_truth: str = "label_or"  # label_or | calendar (rule: or 에서만 쓴다)
    period: tuple[date, date] | None = None  # 평가 구간. 창 원점과 calendar 판정에 쓴다

    @classmethod
    def from_config(
        cls,
        evaluation_cfg: dict,
        *,
        horizon_days: int | None = None,
        period: tuple[date, date] | None = None,
    ) -> "DiskLevelSpec":
        cfg = evaluation_cfg.get("disk_level", {})
        rule = cfg.get("rule", "in_horizon")
        window = cfg.get("window_days", None)
        if window == "horizon":
            if horizon_days is None:
                raise ValueError(
                    "window_days: horizon 을 쓰려면 horizon_days 를 넘겨야 한다."
                )
            window = int(horizon_days)
        elif window is not None:
            window = int(window)

        if rule in ("on_time", "in_horizon") and window is not None and horizon_days is not None:
            if window <= horizon_days:
                raise ValueError(
                    f"rule: {rule} 인데 window_days({window}) <= horizon_days"
                    f"({horizon_days}) 다. 창 안 모든 행이 고장 H일 이내가 되어"
                    " horizon 밖 알람이 존재할 수 없고, 규칙이 단순 OR 집계로"
                    " 붕괴한다. window_days 를 null(평가 구간 전체)로 두어라."
                )
        return cls(
            rule=rule,
            aggregation=cfg.get("aggregation", "any"),
            consecutive_k=int(cfg.get("consecutive_k", 2)),
            window_days=window,
            ground_truth=cfg.get("ground_truth", "label_or"),
            period=period,
        )


# --------------------------------------------------------------------------
# row 단위
# --------------------------------------------------------------------------
def row_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    predicted = (score >= threshold).astype(int)

    tp = int(((predicted == 1) & (y_true == 1)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())
    tn = int(((predicted == 0) & (y_true == 0)).sum())

    both = y_true.size > 0 and y_true.max() != y_true.min()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": int(y_true.shape[0]),
        "n_positive": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if y_true.size else 0.0,
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, score)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, score)) if both else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mcc": float(matthews_corrcoef(y_true, predicted)) if both else float("nan"),
        "brier": _brier(y_true, score),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "alarm_rate": float(predicted.mean()) if predicted.size else 0.0,
    }


# --------------------------------------------------------------------------
# 디스크 단위 집계
# --------------------------------------------------------------------------
def _consecutive_run(alarm: np.ndarray, new_group: np.ndarray) -> np.ndarray:
    """각 위치에서 끝나는 연속 양성 길이. 그룹 경계에서 초기화된다."""
    n = alarm.shape[0]
    position = np.arange(n)
    breaks = np.where(alarm == 0, position, -1)
    starts = np.flatnonzero(new_group)
    starts = starts[starts > 0]
    if starts.size:
        breaks[starts] = np.maximum(breaks[starts], starts - 1)
    run = position - np.maximum.accumulate(breaks)
    return np.where(alarm == 1, run, 0)


def collapse_to_disks(
    predictions: pd.DataFrame, threshold: float, spec: DiskLevelSpec | None = None
) -> pd.DataFrame:
    """(serial, date) 예측을 (디스크, 창) 단위로 접는다.

    predictions 에 필요한 컬럼: serial_number, record_date, y, score, failure_date
    반환 index 는 (serial_number, window) 다.
    """
    spec = spec or DiskLevelSpec()
    frame = predictions.sort_values(["serial_number", "record_date"], kind="stable")
    serial = frame["serial_number"].to_numpy()
    record_date = frame["record_date"].to_numpy().astype("datetime64[D]")
    alarm = (frame["score"].to_numpy() >= threshold).astype(np.int8)

    # 창 번호. 원점은 평가 구간 시작이다. 그래야 fold 마다 같은 위치에서 잘린다.
    if spec.window_days:
        origin = np.datetime64(spec.period[0] if spec.period else record_date.min(), "D")
        window = ((record_date - origin).astype(int) // int(spec.window_days)).astype(
            np.int32
        )
    else:
        window = np.zeros(serial.shape[0], dtype=np.int32)

    new_group = np.empty(serial.shape[0], dtype=bool)
    new_group[0] = True
    new_group[1:] = (serial[1:] != serial[:-1]) | (window[1:] != window[:-1])

    if spec.aggregation == "any":
        fires = alarm
    elif spec.aggregation == "consecutive":
        fires = (_consecutive_run(alarm, new_group) >= spec.consecutive_k).astype(np.int8)
    else:
        raise ValueError(f"알 수 없는 disk_level.aggregation: {spec.aggregation!r}")

    label = frame["y"].to_numpy()
    work = pd.DataFrame(
        {
            "serial_number": serial,
            "window": window,
            "record_date": record_date,
            "y": label,
            "fires": fires,
            # horizon 안(y=1) / 밖(y=0) 알람. on_time 판정의 재료다.
            "in_alarm": (fires * (label == 1)).astype(np.int8),
            "out_alarm": (fires * (label == 0)).astype(np.int8),
            "failure_date": frame["failure_date"].to_numpy().astype("datetime64[D]"),
        }
    )
    grouped = work.groupby(["serial_number", "window"], sort=False)
    units = grouped.agg(
        label_or=("y", "max"),
        predicted=("fires", "max"),
        n_samples=("y", "size"),
        n_alarms=("fires", "sum"),
        n_in_alarm=("in_alarm", "sum"),
        n_out_alarm=("out_alarm", "sum"),
        failure_date=("failure_date", "max"),
        window_first_day=("record_date", "min"),
        window_last_day=("record_date", "max"),
    )
    fired = work.loc[work["fires"] == 1]
    units["first_alarm_date"] = fired.groupby(
        ["serial_number", "window"], sort=False
    )["record_date"].min()

    units["actual"] = _actual_labels(units, spec)
    units["days_in_advance"] = (
        pd.to_datetime(units["failure_date"]) - pd.to_datetime(units["first_alarm_date"])
    ).dt.days
    return units


def _actual_labels(units: pd.DataFrame, spec: DiskLevelSpec) -> np.ndarray:
    """정답을 접는다."""
    if spec.ground_truth == "label_or":
        return units["label_or"].to_numpy().astype(np.int8)
    if spec.ground_truth != "calendar":
        raise ValueError(f"알 수 없는 disk_level.ground_truth: {spec.ground_truth!r}")

    # RODMAN 정의: 그 창의 달력 구간 안에 실제 고장일이 있어야 양성이다.
    # 창의 경계는 관측된 행이 아니라 구간 정의에서 와야 한다. 고장 디스크는
    # 고장 당일 이후 행이 없으므로 관측 최댓값을 경계로 쓰면 자기 고장일이
    # 창 밖으로 밀려 정답이 뒤집힌다. 그래서 period 를 필수로 요구한다.
    if spec.period is None:
        raise ValueError(
            "ground_truth: calendar 는 평가 구간(period)이 있어야 판정할 수 있다."
        )

    failure = units["failure_date"].to_numpy().astype("datetime64[D]")
    period_start = np.datetime64(spec.period[0], "D")
    period_end = np.datetime64(spec.period[1], "D")
    if spec.window_days:
        index = units.index.get_level_values("window").to_numpy()
        start = period_start + (index * spec.window_days).astype("timedelta64[D]")
        end = np.minimum(start + np.timedelta64(spec.window_days - 1, "D"), period_end)
    else:
        start = np.full(units.shape[0], period_start)
        end = np.full(units.shape[0], period_end)

    observed = ~pd.isna(failure)
    return (observed & (failure >= start) & (failure <= end)).astype(np.int8)


# 판정 결과 코드. 배열 하나로 들고 다니면 부트스트랩이 색인만으로 끝난다.
TN, FP_HEALTHY, FP_EARLY, FN, TP = 0, 1, 2, 3, 4


def outcome_codes(units: pd.DataFrame, spec: DiskLevelSpec) -> np.ndarray:
    """각 (디스크, 창) 을 혼동행렬의 한 칸에 배정한다.

    rule = "on_time" (기본):

        out        = horizon 밖(y=0 행) 알람 >= 1
        in         = horizon 안(y=1 행) 알람 >= 1
        has_window = 창 안에 y=1 행이 있는가 (정답 구간의 존재)

        out                       -> FP
        ~out &  in                -> TP
        ~out & ~in &  has_window  -> FN
        ~out & ~in & ~has_window  -> TN

    ~out & ~in 은 곧 "알람이 하나도 없음"이므로 네 칸이 전체를 빈틈없이
    나눈다. FP 는 기록을 위해 fp_early(정답 구간이 있는 디스크)와
    fp_healthy(없는 디스크)로 나눠 둔다.

    rule = "in_horizon":

        has_window & in         -> TP   정답 구간 안에서 울렸다
        has_window & ~in        -> FN   구간이 있는데 그 안에서 안 울렸다
        ~has_window & 알람      -> FP   미고장 디스크가 울렸다
        ~has_window & 무알람    -> TN

        on_time 과의 차이는 구간 밖 알람을 처벌하지 않는다는 것뿐이다.
        선언한 horizon 이 판정을 지배하면서 이중 처벌은 없다.

    rule = "or": 창 안에 알람이 하나라도 있으면 양성으로 보는 표준 OR 집계.
    """
    n = units.shape[0]
    codes = np.empty(n, dtype=np.int8)

    if spec.rule == "or":
        actual = units["actual"].to_numpy().astype(bool)
        predicted = units["predicted"].to_numpy().astype(bool)
        codes[:] = TN
        codes[predicted & actual] = TP
        codes[predicted & ~actual] = FP_HEALTHY
        codes[~predicted & actual] = FN
        return codes

    out_alarm = units["n_out_alarm"].to_numpy() > 0
    in_alarm = units["n_in_alarm"].to_numpy() > 0
    has_window = units["label_or"].to_numpy().astype(bool)

    if spec.rule == "in_horizon":
        # 정답 구간(고장 H일 전) 안에서 울렸으면 TP. 구간 밖 알람은 판정을
        # 바꾸지 않는다 — 감점도 가점도 아니다. 미고장 디스크는 아무 때나
        # 울리면 FP 이므로 오탐 부담은 그대로 측정된다.
        codes[:] = TN
        codes[has_window & in_alarm] = TP
        codes[has_window & ~in_alarm] = FN
        codes[~has_window & (out_alarm | in_alarm)] = FP_HEALTHY
        return codes

    if spec.rule != "on_time":
        raise ValueError(
            f"알 수 없는 disk_level.rule: {spec.rule!r} "
            "(in_horizon | on_time | or)"
        )

    codes[:] = TN
    codes[~out_alarm & ~in_alarm & has_window] = FN
    codes[~out_alarm & in_alarm] = TP
    codes[out_alarm & has_window] = FP_EARLY
    codes[out_alarm & ~has_window] = FP_HEALTHY
    return codes


def _counts_from_codes(codes: np.ndarray) -> dict:
    tn = int((codes == TN).sum())
    fp_healthy = int((codes == FP_HEALTHY).sum())
    fp_early = int((codes == FP_EARLY).sum())
    fn = int((codes == FN).sum())
    tp = int((codes == TP).sum())
    fp = fp_healthy + fp_early

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    # 정답 구간이 있는 디스크 전체. fp_early 는 여기 속하지만 2x2 의 recall
    # 분모에는 들어가지 않으므로 따로 낸다.
    with_window = tp + fn + fp_early
    healthy = tn + fp_healthy
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        # 2x2 열 여백
        "far": fp / (fp + tn) if fp + tn else 0.0,
        # RODMAN 식 분모(미고장 디스크)로 따로 하나 더 낸다
        "far_healthy": fp_healthy / healthy if healthy else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fp_early": fp_early,
        "fp_healthy": fp_healthy,
        "n_with_window": with_window,
        # 정답 구간이 있는 디스크 중 제때 잡은 비율. fp_early 까지 분모에 넣는다.
        "detection_rate": tp / with_window if with_window else 0.0,
    }


def disk_metrics(units: pd.DataFrame, spec: DiskLevelSpec | None = None) -> dict:
    spec = spec or DiskLevelSpec()
    codes = outcome_codes(units, spec)
    detected = units.loc[codes == TP]
    advance = detected["days_in_advance"].dropna()

    return {
        "n_units": int(units.shape[0]),
        "n_disks": int(units.index.get_level_values(0).nunique()),
        **_counts_from_codes(codes),
        # 아래는 기술 통계다. 모델 선정에 쓰지 않는다.
        "days_in_advance_median": float(advance.median()) if len(advance) else float("nan"),
        "days_in_advance_p25": float(advance.quantile(0.25)) if len(advance) else float("nan"),
        "days_in_advance_p75": float(advance.quantile(0.75)) if len(advance) else float("nan"),
        "alarm_count_per_unit": float(units["n_alarms"].mean()) if len(units) else 0.0,
    }


def evaluate(
    predictions: pd.DataFrame,
    threshold: float,
    evaluation_cfg: dict,
    *,
    horizon_days: int | None = None,
    period: tuple[date, date] | None = None,
) -> dict:
    """row 단위와 디스크 단위 지표를 한 번에 만든다."""
    spec = DiskLevelSpec.from_config(
        evaluation_cfg, horizon_days=horizon_days, period=period
    )
    units = collapse_to_disks(predictions, threshold, spec)
    return {
        "row_level": row_metrics(
            predictions["y"].to_numpy(), predictions["score"].to_numpy(), threshold
        ),
        "disk_level": disk_metrics(units, spec),
        "disk_level_spec": {
            "rule": spec.rule,
            "aggregation": spec.aggregation,
            "window_days": spec.window_days,
            "ground_truth": spec.ground_truth,
        },
    }


# --------------------------------------------------------------------------
# 신뢰구간
# --------------------------------------------------------------------------
def _ragged_indices(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """[starts[i], starts[i]+lengths[i]) 구간들을 이어붙인 인덱스 배열."""
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    steps = np.ones(total, dtype=np.int64)
    steps[0] = starts[0]
    boundaries = np.cumsum(lengths)[:-1]
    steps[boundaries] = starts[1:] - (starts[:-1] + lengths[:-1]) + 1
    return np.cumsum(steps)


def _group_slices(keys: np.ndarray, order_of: pd.Index) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """keys 를 order_of 순서의 그룹으로 묶어 (정렬 인덱스, 시작, 길이) 를 준다."""
    codes = pd.Index(order_of).get_indexer(keys)
    if (codes < 0).any():
        raise RuntimeError("그룹에 없는 키가 있습니다.")
    order = np.argsort(codes, kind="stable")
    lengths = np.bincount(codes, minlength=len(order_of)).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return order, starts, lengths


def bootstrap_ci(
    predictions: pd.DataFrame,
    threshold: float,
    evaluation_cfg: dict,
    bootstrap_cfg: dict,
    *,
    horizon_days: int | None = None,
    period: tuple[date, date] | None = None,
) -> dict:
    """디스크 단위 resample 로 신뢰구간을 낸다.

    재표집 단위는 항상 디스크다. 한 디스크가 여러 창(unit)을 내더라도 함께
    뽑힌다. row 단위로 재표집하면 같은 디스크의 행이 독립이 아니어서 구간이
    과도하게 좁아진다.
    """
    if not bootstrap_cfg.get("enabled", False):
        return {}
    if bootstrap_cfg.get("unit", "disk") != "disk":
        raise ValueError("bootstrap.unit 은 disk 만 지원한다.")

    n_resamples = int(bootstrap_cfg.get("n_resamples", 1000))
    level = float(bootstrap_cfg.get("ci", 0.95))
    rng = np.random.default_rng(int(bootstrap_cfg.get("seed", 42)))

    spec = DiskLevelSpec.from_config(
        evaluation_cfg, horizon_days=horizon_days, period=period
    )
    units = collapse_to_disks(predictions, threshold, spec)
    disks = pd.Index(units.index.get_level_values(0).unique())

    unit_order, unit_start, unit_len = _group_slices(
        units.index.get_level_values(0).to_numpy(), disks
    )
    # 판정 결과를 코드 배열로 미리 접어두면 재표집은 색인만으로 끝난다.
    codes = outcome_codes(units, spec)[unit_order]

    frame = predictions.sort_values(["serial_number", "record_date"], kind="stable")
    row_order, row_start, row_len = _group_slices(
        frame["serial_number"].to_numpy(), disks
    )
    y_sorted = frame["y"].to_numpy().astype(np.int8)[row_order]
    score_sorted = frame["score"].to_numpy().astype(np.float64)[row_order]

    n_disks = len(disks)
    samples: dict[str, list[float]] = {}
    for _ in range(n_resamples):
        drawn = rng.integers(0, n_disks, size=n_disks)

        picked = _ragged_indices(unit_start[drawn], unit_len[drawn])
        for key, value in _counts_from_codes(codes[picked]).items():
            samples.setdefault(f"disk_level.{key}", []).append(float(value))

        rows = _ragged_indices(row_start[drawn], row_len[drawn])
        y_boot, score_boot = y_sorted[rows], score_sorted[rows]
        if y_boot.size and y_boot.max() != y_boot.min():
            samples.setdefault("row_level.pr_auc", []).append(
                float(average_precision_score(y_boot, score_boot))
            )
            samples.setdefault("row_level.roc_auc", []).append(
                float(roc_auc_score(y_boot, score_boot))
            )

    alpha = (1.0 - level) / 2.0
    return {
        key: {
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1 - alpha)),
            "mean": float(np.mean(values)),
            "n_resamples": len(values),
        }
        for key, values in samples.items()
        if values
    }
