"""fold 하나의 model-ready 데이터를 조립한다.

트리 모델과 시퀀스 모델은 입력 형태가 다르지만 같은 canonical / label /
split 을 공유한다. 이 모듈이 그 접점이다.

    tree     : (serial, t) -> 평탄화된 피처 벡터 + y
    sequence : (serial, t) -> [t-L+1 .. t] 시퀀스 + y

시퀀스 쪽에서 조심할 것이 하나 있다. test 월의 첫날 표본도 lookback 이
필요하므로 test 구간 이전 L-1 일의 피처 행을 함께 읽어야 한다. 그 행들은
입력으로만 쓰이고 채점 대상이 되지 않는다.

이것은 누출이 아니다. lookback 은 관측 시점 t 이전만 본다. 누출은 t 이후를
보는 것이고, 그건 embargo 와 라벨 정의가 막는다.

메모리 주의사항 (HGST train fold 가 24M행 x 86피처다):
  - 결과를 pandas DataFrame 으로 받으면 DataFrame 과 numpy 사본이 동시에
    존재해 사용량이 두 배가 된다. Arrow record batch 로 스트리밍해서 미리
    잡아둔 행렬에 채워 넣는다.
  - 날짜는 epoch 이후 일수(int32)로 받는다. object 배열이 되면 24M행에서
    수 GB를 쓴다.
  - serial_number 문자열은 평가에 필요한 val/test 에서만 읽는다.
  - stride 표본 추출은 SQL 로 내린다. 다 읽고 나서 버리면 의미가 없다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ..features import build as features_build
from ..labeling import horizon_labels

_EPOCH = "DATE '1970-01-01'"
_BATCH_ROWS = 1_000_000


@dataclass
class FoldMatrix:
    """모델에 그대로 넣을 수 있는 상태의 fold 조각.

    serial / failure_date 는 평가에 필요한 구간에서만 채운다.
    """

    X: np.ndarray
    y: np.ndarray
    record_date: np.ndarray
    serial: np.ndarray | None = None
    failure_date: np.ndarray | None = None
    columns: list[str] = field(default_factory=list)
    sampling: dict = field(default_factory=dict)  # 표본 추출 근거 (기록용)
    window: tuple[date, date] | None = None  # 이 조각이 덮는 구간

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def healthy_mask(self) -> np.ndarray:
        """창 끝 시점에 "아직 고장이 확인되지 않은" 디스크의 행.

        이상탐지에서 정상 개체만 학습할 때 쓴다. 창 끝 이후의 고장은 그 시점에
        알 수 없는 정보이므로 정상으로 본다. failure_date 를 그대로 쓰면
        미래를 보고 학습 표본을 고르는 것이 되어 누출이다.
        """
        if self.failure_date is None:
            raise ValueError("failure_date 가 없어 정상 개체를 가려낼 수 없다.")
        healthy = np.isnat(self.failure_date)
        if self.window is not None:
            healthy |= self.failure_date > np.datetime64(self.window[1])
        return healthy

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean()) if len(self) else 0.0


@dataclass
class SequenceFold:
    """시퀀스 모델용. 행렬 하나와 표본 끝 인덱스를 들고 있는다.

    윈도우를 미리 복제하지 않는다. L=30, 표본 2천만개면 복제본은 수백 GB다.

    valid_len 이 있으면 padding 모드다. 표본 i 의 실제 행은 lookback 중
    valid_len[i] 개뿐이고 앞쪽은 그 segment 의 첫 행으로 채운다. 이때 mask
    채널 하나가 뒤에 붙으므로 n_features 가 1 늘어난다.
    """

    matrix: np.ndarray  # (n_rows, n_features) float32
    end_index: np.ndarray  # 표본마다 시퀀스 끝 행의 위치
    lookback: int
    y: np.ndarray
    record_date: np.ndarray
    serial: np.ndarray | None = None
    failure_date: np.ndarray | None = None
    columns: list[str] = field(default_factory=list)
    valid_len: np.ndarray | None = None  # None -> 전부 lookback 만큼 채워짐
    sampling: dict = field(default_factory=dict)  # 표본 추출 근거 (기록용)
    window: tuple[date, date] | None = None  # 이 조각이 덮는 구간

    def __len__(self) -> int:
        return int(self.y.shape[0])

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean()) if len(self) else 0.0

    def healthy_mask(self) -> np.ndarray:
        """창 끝 시점에 아직 고장이 확인되지 않은 디스크의 표본.

        FoldMatrix.healthy_mask 와 같은 규칙이다. 창 끝 이후의 고장은 그
        시점에 알 수 없으므로 정상으로 본다.
        """
        if self.failure_date is None:
            raise ValueError("failure_date 가 없어 정상 개체를 가려낼 수 없다.")
        healthy = np.isnat(self.failure_date)
        if self.window is not None:
            healthy |= self.failure_date > np.datetime64(self.window[1])
        return healthy

    @property
    def padded(self) -> bool:
        return self.valid_len is not None

    @property
    def n_features(self) -> int:
        return int(self.matrix.shape[1]) + (1 if self.padded else 0)

    @property
    def short_sample_rate(self) -> float:
        """lookback 을 다 못 채운 표본의 비율."""
        if self.valid_len is None or not len(self):
            return 0.0
        return float((self.valid_len < self.lookback).mean())


class Scaler:
    """train 구간에서만 적합시키는 표준화 + 분위수 절단.

    fold 밖의 통계를 절대 보지 않는다. transform 은 fit 때 저장한 값만 쓴다.
    """

    def __init__(self, method: str = "standard", clip_quantile=(0.001, 0.999)):
        self.method = method
        self.clip_quantile = tuple(clip_quantile) if clip_quantile else None
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Scaler":
        if self.method == "none":
            return self
        if self.clip_quantile:
            low, high = self.clip_quantile
            self.lower_ = np.nanquantile(X, low, axis=0).astype(np.float32)
            self.upper_ = np.nanquantile(X, high, axis=0).astype(np.float32)
            X = np.clip(X, self.lower_, self.upper_)
        if self.method == "robust":
            self.center_ = np.nanmedian(X, axis=0).astype(np.float32)
            spread = np.nanquantile(X, 0.75, axis=0) - np.nanquantile(X, 0.25, axis=0)
            self.scale_ = spread.astype(np.float32)
        elif self.method == "minmax":
            # transform 의 (X - center) / scale 이 그대로 [0, 1] 매핑이 된다.
            # 절단 뒤의 최소·최대를 쓰므로 train 구간의 극단값 하나가 나머지를
            # 0 근처로 뭉개는 일이 없다.
            self.center_ = np.nanmin(X, axis=0).astype(np.float32)
            self.scale_ = (np.nanmax(X, axis=0) - self.center_).astype(np.float32)
        else:
            self.center_ = np.nanmean(X, axis=0).astype(np.float32)
            self.scale_ = np.nanstd(X, axis=0).astype(np.float32)
        # 분산이 0인 컬럼은 그대로 둔다 (0으로 나누지 않는다).
        self.scale_ = np.where(
            ~np.isfinite(self.scale_) | (self.scale_ < 1e-8), 1.0, self.scale_
        ).astype(np.float32)
        return self

    def transform(self, X: np.ndarray, *, fill_nan: bool = False) -> np.ndarray:
        if self.method == "none":
            out = X
        else:
            if self.lower_ is not None:
                X = np.clip(X, self.lower_, self.upper_)
            out = (X - self.center_) / self.scale_
        if fill_nan:
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.astype(np.float32, copy=False)

    def state_dict(self) -> dict:
        def _list(value):
            return None if value is None else np.asarray(value).tolist()

        return {
            "method": self.method,
            "clip_quantile": list(self.clip_quantile) if self.clip_quantile else None,
            "center": _list(self.center_),
            "scale": _list(self.scale_),
            "lower": _list(self.lower_),
            "upper": _list(self.upper_),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "Scaler":
        scaler = cls(state["method"], state.get("clip_quantile"))
        for attr, key in (
            ("center_", "center"),
            ("scale_", "scale"),
            ("lower_", "lower"),
            ("upper_", "upper"),
        ):
            value = state.get(key)
            setattr(
                scaler, attr, None if value is None else np.asarray(value, np.float32)
            )
        return scaler


# --------------------------------------------------------------------------
# 읽기
# --------------------------------------------------------------------------
def _quoted(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def _connect(threads: int) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={threads}")
    con.execute("SET preserve_insertion_order=false")
    return con


def _positive_condition(
    horizon_days: int | None, end: date, scope: str, prefix: str = "l."
) -> str:
    """"이 행은 양성인가"를 판정하는 조건식.

    prefix 가 "l." 이면 피처 ⋈ 라벨 조인용(날짜는 f.record_date), 빈 문자열이면
    라벨 레이어 단독 조회용이다. NULL 을 내지 않으므로 NOT (...) 이 안전하다.
    """
    if scope == "global":
        return f"COALESCE({prefix}y, 0) = 1"
    if scope != "window":
        raise ValueError(f"알 수 없는 censoring_scope: {scope!r} (global | window)")
    if horizon_days is None:
        raise ValueError("censoring_scope: window 는 horizon_days 가 필요하다.")
    date_col = "f.record_date" if prefix else "record_date"
    return (
        f"({prefix}failure_date IS NOT NULL"
        f" AND {prefix}failure_date <= DATE '{end}'"
        f" AND date_diff('day', {date_col}, {prefix}failure_date) <= {horizon_days})"
    )


def _survived_condition(horizon_days: int | None, end: date, prefix: str = "l.") -> str:
    """창 끝까지 horizon 을 다 관측할 수 있는가."""
    date_col = "f.record_date" if prefix else "record_date"
    return f"({date_col} + INTERVAL {horizon_days} DAY <= DATE '{end}')"


def _label_expression(horizon_days: int | None, end: date, scope: str) -> str:
    """창 안에서 쓸 라벨 식.

    scope = "global" (기본)
        라벨 레이어가 만든 y 를 그대로 쓴다. 관측 전 구간을 근거로
        "horizon 끝까지 생존했는가"를 판정한 값이다.

    scope = "window"
        창 끝(end) 시점까지 알 수 있는 정보만 쓴다. 실제 운영에서 그 달이
        끝난 시점에 판정한다면 이쪽이 맞다.
          - 창 안에서 고장이 확인되고 horizon 이내면      -> 1
          - horizon 끝이 창 안에 들어오면 (생존 확인 가능) -> 0
          - 그 외 (창 끝 H일 구간)                        -> 라벨 없음, 표본에서 제외

        라벨 레이어의 전역 검열은 이미 걸려 있으므로 두 조건의 교집합이 된다.
        관측이 창 중간에 끊긴 디스크는 전역 검열이 먼저 걸러낸다.
    """
    if scope == "global":
        return "COALESCE(l.y, -1)::SMALLINT AS y_raw"
    return f"""
        CASE
            WHEN l.y IS NULL THEN -1
            WHEN {_positive_condition(horizon_days, end, scope)} THEN 1
            WHEN {_survived_condition(horizon_days, end)} THEN 0
            ELSE -1
        END::SMALLINT AS y_raw
    """


# --------------------------------------------------------------------------
# 학습 표본 구성
# --------------------------------------------------------------------------
_PPM = 1_000_000


@dataclass
class SamplingPlan:
    """train 구간을 어떻게 추릴지에 대한 계획.

    none
        전 표본을 그대로 쓴다.

    stride
        음성만 n일 간격으로 남기고 양성은 전부 유지한다.

    match_val
        확장 창(expanding) 재학습용 구성이다.

            [과거 양성 전량] + [직전 K개월 전량] + [그 이전 음성 일부]

        마지막 항의 표집률은 "전체 양성 비율이 val 창의 양성 비율과 같아지도록"
        정한다. 양성은 한 행도 버리지 않으므로 목표 비율에 도달하려면 음성만
        줄일 수 있고, 이미 목표보다 양성이 희박하면 (필요 음성 > 보유 음성)
        전부 남긴다. 그 경우 실제 비율이 목표보다 낮게 나오고 stats 에 남는다.
    """

    strategy: str = "none"
    stride: int | None = None
    recent_start: date | None = None
    keep_ppm: int = _PPM
    seed: int = 42
    stats: dict = field(default_factory=dict)


def _hash_bucket(serial_col: str, date_col: str, seed: int) -> str:
    """행 단위 결정적 표집 버킷. 같은 seed 면 항상 같은 행이 뽑힌다."""
    return f"(hash({serial_col} || '|' || {date_col}::VARCHAR || '|{seed}') % {_PPM})"


def _negative_budget(
    positives: int, recent_negatives: int, old_negatives: int, target_rate: float
) -> tuple[int, float]:
    """(과거 음성 중 남길 개수, 유지 비율).

    양성 P 를 전부 들고 목표 양성 비율 r 을 맞추려면 음성이 P(1-r)/r 개 필요하다.
    직전 K개월 음성은 무조건 들어가므로 나머지를 과거에서 채운다.
    """
    if target_rate <= 0 or positives == 0:
        return old_negatives, 1.0
    needed = positives * (1.0 - target_rate) / target_rate
    take = max(0.0, needed - recent_negatives)
    keep = min(old_negatives, int(round(take)))
    return keep, (keep / old_negatives if old_negatives else 0.0)


def recent_window_start(end: date, months: int) -> date:
    """창 끝에서 거슬러 올라간 K개월 구간의 시작일."""
    return (pd.Timestamp(end) - pd.DateOffset(months=int(months))).date() + timedelta(
        days=1
    )


def plan_from(
    sampling_cfg: dict | None,
    *,
    end: date | None = None,
    target_positive_rate: float | None = None,
) -> SamplingPlan:
    """데이터를 읽지 않고 정할 수 있는 부분까지만 채운 계획."""
    strategy = (sampling_cfg or {}).get("strategy", "none")
    if strategy == "none":
        return SamplingPlan()
    if strategy == "stride":
        return SamplingPlan(
            strategy="stride", stride=int(sampling_cfg.get("negative_stride", 7))
        )
    if strategy == "ratio":
        # 고전적 무작위 언더샘플링. 양성 전량 + 음성 (ratio x 양성) 개.
        return SamplingPlan(
            strategy="ratio",
            seed=int((sampling_cfg or {}).get("seed", 42)),
            stats={"negative_ratio": float((sampling_cfg or {}).get("negative_ratio", 50))},
        )
    if strategy != "match_val":
        raise ValueError(f"알 수 없는 train_sampling.strategy: {strategy!r}")
    if end is None:
        raise ValueError("match_val 은 창 끝 날짜가 필요하다.")
    if target_positive_rate is None:
        raise ValueError(
            "match_val 은 목표 양성 비율(val 창의 양성 비율)이 필요하다. "
            "평가 구간에는 표본 추출을 적용하지 않는다."
        )
    months = int((sampling_cfg or {}).get("keep_recent_months", 1))
    return SamplingPlan(
        strategy="match_val",
        recent_start=recent_window_start(end, months),
        seed=int((sampling_cfg or {}).get("seed", 42)),
        stats={"target_rate": float(target_positive_rate), "keep_recent_months": months},
    )


def _resolve_ratio(
    con: duckdb.DuckDBPyConnection,
    plan: SamplingPlan,
    labels_path: Path,
    start: date,
    end: date,
    *,
    horizon_days: int | None,
    censoring_scope: str,
) -> SamplingPlan:
    """양성은 전량, 음성은 (ratio x 양성) 개만 무작위로 남긴다.

    선행연구에서 가장 흔한 형태의 언더샘플링이다. stride 와 달리 시간축을
    보지 않으므로 같은 디스크의 연속된 날이 뭉쳐 뽑힐 수 있다.
    """
    positive = _positive_condition(horizon_days, end, censoring_scope, prefix="")
    keep_sql = ""
    if censoring_scope == "window":
        survived = _survived_condition(horizon_days, end, prefix="")
        keep_sql = f"AND ({positive} OR {survived})"

    row = con.execute(
        f"""
        SELECT COUNT(*) FILTER (WHERE {positive}),
               COUNT(*) FILTER (WHERE NOT ({positive}))
        FROM read_parquet('{horizon_labels.dataset_glob(labels_path)}',
                          hive_partitioning=true)
        WHERE month BETWEEN '{start:%Y-%m}' AND '{end:%Y-%m}'
          AND record_date BETWEEN DATE '{start}' AND DATE '{end}'
        {keep_sql}
        """
    ).fetchone()
    positives, negatives = int(row[0]), int(row[1])
    ratio = float(plan.stats["negative_ratio"])
    keep = min(negatives, int(round(ratio * positives)))
    fraction = keep / negatives if negatives else 0.0
    total = positives + keep
    plan.keep_ppm = _PPM if fraction >= 1.0 else int(round(fraction * _PPM))
    plan.stats.update(
        {
            "positives": positives,
            "negatives": negatives,
            "negatives_kept": keep,
            "keep_fraction": round(fraction, 6),
            "rows_before": positives + negatives,
            "rows_after": total,
            "actual_rate": positives / total if total else 0.0,
        }
    )
    s = plan.stats
    print(
        f"    [sampling] ratio 1:{ratio:g}: 양성 {positives:,} 전량 + 음성 "
        f"{keep:,}/{negatives:,} ({fraction:.2%}) -> {s['rows_before']:,}행에서 "
        f"{total:,}행, 양성률 {s['actual_rate']:.4%}"
    )
    if fraction >= 1.0 and negatives:
        print("      ⚠ 목표 비율에 필요한 음성보다 보유량이 적어 전부 남겼다.")
    return plan


def _resolve_match_val(
    con: duckdb.DuckDBPyConnection,
    plan: SamplingPlan,
    labels_path: Path,
    start: date,
    end: date,
    *,
    horizon_days: int | None,
    censoring_scope: str,
) -> SamplingPlan:
    """라벨 레이어를 세어 과거 음성 표집률을 정한다.

    피처 ⋈ 라벨은 (serial, record_date) 1:1 이라 라벨 쪽만 세어도 같은 값이다
    (tests/test_protocol.py::test_split_stats_match_what_actually_loads).
    라벨 레이어는 피처보다 훨씬 가벼워서 여는 비용이 거의 없다.
    """
    positive = _positive_condition(horizon_days, end, censoring_scope, prefix="")
    keep_sql = ""
    if censoring_scope == "window":
        survived = _survived_condition(horizon_days, end, prefix="")
        keep_sql = f"AND ({positive} OR {survived})"

    row = con.execute(
        f"""
        SELECT
            COUNT(*) FILTER (WHERE {positive}),
            COUNT(*) FILTER (WHERE NOT ({positive})
                             AND record_date >= DATE '{plan.recent_start}'),
            COUNT(*) FILTER (WHERE NOT ({positive})
                             AND record_date <  DATE '{plan.recent_start}')
        FROM read_parquet('{horizon_labels.dataset_glob(labels_path)}',
                          hive_partitioning=true)
        WHERE month BETWEEN '{start:%Y-%m}' AND '{end:%Y-%m}'
          AND record_date BETWEEN DATE '{start}' AND DATE '{end}'
        {keep_sql}
        """
    ).fetchone()
    positives, recent_negatives, old_negatives = (int(v) for v in row)

    target = float(plan.stats["target_rate"])
    keep, fraction = _negative_budget(
        positives, recent_negatives, old_negatives, target
    )
    total = positives + recent_negatives + keep
    plan.keep_ppm = _PPM if fraction >= 1.0 else int(round(fraction * _PPM))
    plan.stats.update(
        {
            "positives": positives,
            "recent_negatives": recent_negatives,
            "old_negatives": old_negatives,
            "old_negatives_kept": keep,
            "keep_fraction": round(fraction, 6),
            "rows_before": positives + recent_negatives + old_negatives,
            "rows_after": total,
            "actual_rate": positives / total if total else 0.0,
        }
    )
    _report_match_val(plan)
    return plan


def _report_match_val(plan: SamplingPlan) -> None:
    s = plan.stats
    print(
        f"    [sampling] match_val: 양성 {s['positives']:,} 전량 + 직전 "
        f"{s['keep_recent_months']}개월 음성 {s['recent_negatives']:,} 전량 + "
        f"과거 음성 {s['old_negatives_kept']:,}/{s['old_negatives']:,} "
        f"({s['keep_fraction']:.1%}) -> {s['rows_before']:,}행에서 "
        f"{s['rows_after']:,}행, 양성률 {s['actual_rate']:.4%} "
        f"(목표 {s['target_rate']:.4%})"
    )
    if s["keep_fraction"] >= 1.0 and s["old_negatives"]:
        print(
            "      ⚠ 목표 비율에 닿으려면 음성이 더 필요해 과거 음성을 전부 남겼다 "
            "(train 의 자연 불균형이 이미 val 보다 양성 희박하다)."
        )


def _sample_match_val(
    plan: SamplingPlan,
    end_index: np.ndarray,
    y: np.ndarray,
    record_date: np.ndarray,
) -> np.ndarray:
    """시퀀스 경로의 match_val. 행렬은 두고 채점 대상 인덱스만 추린다."""
    positive = y == 1
    recent = record_date >= np.datetime64(plan.recent_start)
    old = np.flatnonzero(~positive & ~recent)
    positives = int(positive.sum())
    recent_negatives = int((recent & ~positive).sum())

    keep_n, fraction = _negative_budget(
        positives, recent_negatives, int(old.size), float(plan.stats["target_rate"])
    )
    mask = positive | recent
    if keep_n >= old.size:
        mask[old] = True
    elif keep_n:
        rng = np.random.default_rng(plan.seed)
        mask[rng.choice(old, size=keep_n, replace=False)] = True

    total = int(mask.sum())
    plan.stats.update(
        {
            "positives": positives,
            "recent_negatives": recent_negatives,
            "old_negatives": int(old.size),
            "old_negatives_kept": int(min(keep_n, old.size)),
            "keep_fraction": round(fraction, 6),
            "rows_before": int(end_index.size),
            "rows_after": total,
            "actual_rate": positives / total if total else 0.0,
        }
    )
    _report_match_val(plan)
    return end_index[mask]


def _build_query(
    features_path: Path,
    labels_path: Path,
    columns: list[str],
    start: date,
    end: date,
    *,
    how: str,
    with_serial: bool,
    plan: "SamplingPlan | None" = None,
    horizon_days: int | None = None,
    censoring_scope: str = "global",
    drop_unlabelled: bool = True,
) -> str:
    """피처 + 라벨 조회 SQL.

    month 조건을 함께 걸어 hive partition pruning 이 걸리게 한다.
    날짜는 epoch 이후 일수로 내보낸다 (object 배열 방지).
    """
    feature_sql = ", ".join(f"f.{_quoted(c)}" for c in columns)
    join_type = "INNER JOIN" if how == "inner" else "LEFT JOIN"
    order = "f.serial_number, f.segment, f.record_date"
    serial_sql = "f.serial_number," if with_serial else ""

    # (serial, segment) 가 바뀌는 지점. 시퀀스 lookback 유효성 판정에 쓴다.
    # serial 문자열을 파이썬으로 가져오지 않고도 그룹 경계를 알 수 있다.
    group_start_sql = f"""
        CASE WHEN LAG(f.serial_number) OVER (ORDER BY {order})
                  IS DISTINCT FROM f.serial_number
               OR LAG(f.segment) OVER (ORDER BY {order})
                  IS DISTINCT FROM f.segment
             THEN 1 ELSE 0 END::TINYINT AS group_start
    """
    plan = plan or SamplingPlan()
    qualify = ""
    if plan.strategy == "stride" and plan.stride:
        # 학습 표본만 줄인다. 양성은 전부 남긴다.
        qualify = f"""
        QUALIFY l.y = 1
             OR (ROW_NUMBER() OVER (
                    PARTITION BY f.serial_number ORDER BY f.record_date
                 ) - 1) % {int(plan.stride)} = 0
        """

    # 창 기준 검열: 창 끝까지 판정할 수 없는 행은 표본에서 뺀다. INNER JOIN
    # (트리 입력) 은 여기서 걸러내고, 시퀀스는 행렬 재료로 남겨둔 뒤 y_raw=-1
    # 로 채점 대상에서만 제외한다.
    window_filter = ""
    if drop_unlabelled and censoring_scope == "window":
        window_filter = f"""
          AND ({_positive_condition(horizon_days, end, censoring_scope)}
             OR {_survived_condition(horizon_days, end)})
        """

    # 과거 음성 일부 표집. 양성과 직전 K개월은 조건에서 먼저 빠져나가므로
    # 해시 표집 대상이 되지 않는다. QUALIFY 가 아니라 WHERE 인 이유는
    # group_start (LAG) 가 표집 이후의 행 순서로 계산되어야 하기 때문이다.
    sample_filter = ""
    if plan.strategy == "ratio" and plan.keep_ppm < _PPM:
        sample_filter = f"""
          AND ({_positive_condition(horizon_days, end, censoring_scope)}
             OR {_hash_bucket("f.serial_number", "f.record_date", plan.seed)}
                < {plan.keep_ppm})
        """
    if plan.strategy == "match_val" and plan.keep_ppm < _PPM:
        sample_filter = f"""
          AND ({_positive_condition(horizon_days, end, censoring_scope)}
             OR f.record_date >= DATE '{plan.recent_start}'
             OR {_hash_bucket("f.serial_number", "f.record_date", plan.seed)}
                < {plan.keep_ppm})
        """

    return f"""
        SELECT
            {serial_sql}
            date_diff('day', {_EPOCH}, f.record_date)::INTEGER AS record_days,
            date_diff('day', {_EPOCH}, l.failure_date)::INTEGER AS failure_days,
            {_label_expression(horizon_days, end, censoring_scope)},
            {group_start_sql},
            {feature_sql}
        FROM read_parquet('{features_build.dataset_glob(features_path)}',
                          hive_partitioning=true) AS f
        {join_type} read_parquet('{horizon_labels.dataset_glob(labels_path)}',
                          hive_partitioning=true) AS l
          ON f.serial_number = l.serial_number
         AND f.record_date   = l.record_date
         -- LEFT JOIN 의 의미를 바꾸지 않으면서 라벨 쪽 partition 을 자른다.
         AND l.month BETWEEN '{start:%Y-%m}' AND '{end:%Y-%m}'
        WHERE f.month BETWEEN '{start:%Y-%m}' AND '{end:%Y-%m}'
          AND f.record_date BETWEEN DATE '{start}' AND DATE '{end}'
        {window_filter}
        {sample_filter}
        {qualify}
        ORDER BY {order}
    """


def _stream(
    con: duckdb.DuckDBPyConnection, query: str, columns: list[str], with_serial: bool
) -> tuple[np.ndarray, dict]:
    """Arrow record batch 로 읽어 미리 잡아둔 행렬에 채운다."""
    n_rows = con.execute(f"SELECT COUNT(*) FROM ({query}) AS q").fetchone()[0]
    X = np.empty((n_rows, len(columns)), dtype=np.float32)
    record_days = np.empty(n_rows, dtype=np.int64)
    failure_days = np.empty(n_rows, dtype=np.float64)
    y_raw = np.empty(n_rows, dtype=np.int16)
    group_start = np.empty(n_rows, dtype=np.int8)
    serial_parts: list[np.ndarray] = []

    result = con.execute(query)
    # duckdb 1.3+ 는 to_arrow_reader, 그 이전은 fetch_record_batch.
    fetch = getattr(result, "to_arrow_reader", None) or result.fetch_record_batch
    reader = fetch(_BATCH_ROWS)
    cursor = 0
    for batch in reader:
        size = batch.num_rows
        stop = cursor + size
        for position, name in enumerate(columns):
            X[cursor:stop, position] = batch.column(name).to_numpy(zero_copy_only=False)
        record_days[cursor:stop] = batch.column("record_days").to_numpy(
            zero_copy_only=False
        )
        failure_days[cursor:stop] = batch.column("failure_days").to_numpy(
            zero_copy_only=False
        )
        y_raw[cursor:stop] = batch.column("y_raw").to_numpy(zero_copy_only=False)
        group_start[cursor:stop] = batch.column("group_start").to_numpy(
            zero_copy_only=False
        )
        if with_serial:
            serial_parts.append(batch.column("serial_number").to_numpy(zero_copy_only=False))
        cursor = stop

    if cursor != n_rows:
        raise RuntimeError(f"읽은 행 수가 다릅니다: {cursor} != {n_rows}")

    meta = {
        "record_date": record_days.view("datetime64[D]"),
        "failure_date": _days_to_dates(failure_days),
        "has_label": y_raw >= 0,
        "y": np.maximum(y_raw, 0).astype(np.int8),
        "group_start": group_start,
        "serial": np.concatenate(serial_parts) if serial_parts else None,
    }
    return X, meta


def _days_to_dates(days: np.ndarray) -> np.ndarray:
    """epoch 이후 일수(결측은 NaN)를 datetime64[D] 로 바꾼다."""
    out = np.empty(days.shape[0], dtype="datetime64[D]")
    missing = np.isnan(days)
    out[~missing] = days[~missing].astype(np.int64).view("datetime64[D]")
    out[missing] = np.datetime64("NaT")
    return out


def _position_in_group(group_start: np.ndarray) -> np.ndarray:
    """각 행이 자기 (serial, segment) 그룹에서 몇 번째인지."""
    index = np.arange(group_start.shape[0])
    starts = np.where(group_start == 1, index, 0)
    return index - np.maximum.accumulate(starts)


# --------------------------------------------------------------------------
# 공개 API
# --------------------------------------------------------------------------
def load_tabular(
    features_path: Path,
    labels_path: Path,
    start: date,
    end: date,
    *,
    columns: list[str] | None = None,
    sampling_cfg: dict | None = None,
    target_positive_rate: float | None = None,
    with_meta: bool = True,
    horizon_days: int | None = None,
    censoring_scope: str = "global",
    threads: int = 8,
) -> FoldMatrix:
    columns = columns or features_build.tabular_columns(features_path)
    plan = plan_from(
        sampling_cfg, end=end, target_positive_rate=target_positive_rate
    )
    con = _connect(threads)
    try:
        if plan.strategy == "ratio":
            plan = _resolve_ratio(
                con, plan, labels_path, start, end,
                horizon_days=horizon_days, censoring_scope=censoring_scope,
            )
        if plan.strategy == "match_val":
            plan = _resolve_match_val(
                con,
                plan,
                labels_path,
                start,
                end,
                horizon_days=horizon_days,
                censoring_scope=censoring_scope,
            )
        query = _build_query(
            features_path,
            labels_path,
            columns,
            start,
            end,
            how="inner",
            with_serial=with_meta,
            plan=plan,
            horizon_days=horizon_days,
            censoring_scope=censoring_scope,
            drop_unlabelled=True,
        )
        X, meta = _stream(con, query, columns, with_serial=with_meta)
    finally:
        con.close()

    return FoldMatrix(
        X=X,
        y=meta["y"],
        record_date=meta["record_date"],
        serial=meta["serial"],
        # failure_date 는 _stream 이 이미 만들어 둔 int64 배열이라 train 구간에
        # 들고 있어도 추가 비용이 없다 (행당 8바이트). 이상탐지가 정상 개체를
        # 가려낼 때 필요하다. 수 GB 를 먹는 것은 serial 문자열 쪽이고 그건
        # 여전히 with_meta 로 막는다.
        failure_date=meta["failure_date"],
        columns=list(columns),
        sampling=dict(plan.stats, strategy=plan.strategy),
        window=(start, end),
    )


def load_sequence(
    features_path: Path,
    labels_path: Path,
    start: date,
    end: date,
    *,
    lookback: int,
    columns: list[str] | None = None,
    sampling_cfg: dict | None = None,
    target_positive_rate: float | None = None,
    short_history: str = "pad",
    with_meta: bool = True,
    horizon_days: int | None = None,
    censoring_scope: str = "global",
    threads: int = 8,
) -> SequenceFold:
    """lookback 여유분을 포함해 읽고, 채점 대상 표본만 인덱스로 남긴다.

    short_history:
        pad  - lookback 을 못 채우는 표본도 살린다. 앞쪽을 그 segment 의 첫
               행으로 복제하고 mask 채널을 붙인다. 트리 모델과 평가 모집단이
               같아진다. (기본값)
        drop - 못 채우는 표본을 버린다. segment 가 새로 시작된 디스크가
               모집단에서 빠진다.
    """
    columns = columns or features_build.sequence_columns(features_path)
    plan = plan_from(
        sampling_cfg, end=end, target_positive_rate=target_positive_rate
    )
    query = _build_query(
        features_path,
        labels_path,
        columns,
        start - timedelta(days=lookback - 1),
        end,
        how="left",
        with_serial=with_meta,
        # 시퀀스는 행렬 전체가 lookback 재료라 SQL 에서 행을 못 버린다.
        # 표본 추출은 아래에서 end_index 에만 적용한다.
        plan=None,
        horizon_days=horizon_days,
        censoring_scope=censoring_scope,
        drop_unlabelled=False,
    )
    con = _connect(threads)
    try:
        matrix, meta = _stream(con, query, columns, with_serial=with_meta)
    finally:
        con.close()

    # canonical 이 segment 안의 날짜를 빠짐없이 채워두었으므로 행 간격 = 1일이다.
    # position = 그 (serial, segment) 안에서 몇 번째 행인가.
    record_date = meta["record_date"]
    position = _position_in_group(meta["group_start"])
    in_window = (
        meta["has_label"]
        & (record_date >= np.datetime64(start))
        & (record_date <= np.datetime64(end))
    )

    if short_history == "drop":
        # lookback 을 다 채울 수 있는 위치만 남긴다. 관측 공백으로 segment 가
        # 새로 시작된 디스크가 평가 모집단에서 빠지므로 트리 모델과 분모가
        # 달라진다. 비교 실험에서는 pad 를 쓰는 편이 낫다.
        eligible = in_window & (position >= lookback - 1)
    elif short_history == "pad":
        eligible = in_window
    else:
        raise ValueError(f"알 수 없는 sequence.short_history: {short_history!r}")

    end_index = np.flatnonzero(eligible)

    # 표본 추출은 채점 대상 인덱스에만 적용한다. 행렬은 lookback 재료이므로
    # 손대지 않는다.
    if plan.strategy == "stride" and plan.stride:
        order = _position_in_group(meta["group_start"][end_index].astype(np.int8))
        positive = meta["y"][end_index] == 1
        keep = positive | (order % plan.stride == 0)
        print(
            f"    [sampling] stride={plan.stride}: 표본 {end_index.shape[0]:,} -> "
            f"{int(keep.sum()):,}개 (양성 {int(positive.sum()):,} 전량 유지)"
        )
        end_index = end_index[keep]
    elif plan.strategy == "match_val":
        end_index = _sample_match_val(
            plan, end_index, meta["y"][end_index], record_date[end_index]
        )

    valid_len = None
    if short_history == "pad":
        # 표본마다 실제로 존재하는 행 수. 앞쪽 모자란 만큼은 그 segment 의
        # 첫 행을 복제하고 mask 채널로 표시한다.
        valid_len = np.minimum(position[end_index] + 1, lookback).astype(np.int32)
        short = int((valid_len < lookback).sum())
        if short:
            print(
                f"    [sequence] lookback {lookback}일을 못 채운 표본 "
                f"{short:,}개 ({short / max(end_index.shape[0], 1):.2%}) 를 "
                "복제 패딩 + mask 채널로 채운다."
            )

    failure_date = meta["failure_date"]
    return SequenceFold(
        matrix=matrix,
        end_index=end_index,
        lookback=lookback,
        y=meta["y"][end_index],
        record_date=record_date[end_index],
        serial=meta["serial"][end_index] if meta["serial"] is not None else None,
        # failure_date 는 표본당 8바이트뿐이고 _stream 이 이미 만들어 두었다.
        # 이상탐지가 정상 개체를 가려낼 때 train 구간에서도 필요하다.
        # 수 GB 를 먹는 것은 serial 문자열 쪽이고 그건 with_meta 로 막는다.
        failure_date=failure_date[end_index],
        columns=list(columns),
        valid_len=valid_len,
        sampling=dict(plan.stats, strategy=plan.strategy),
        window=(start, end),
    )


def stride_from(sampling_cfg: dict | None) -> int | None:
    """stride 전략일 때의 간격. 학습 행렬 크기를 미리 어림할 때 쓴다."""
    strategy = (sampling_cfg or {}).get("strategy", "none")
    if strategy == "stride":
        return int(sampling_cfg.get("negative_stride", 7))
    if strategy not in ("none", "match_val", "ratio"):
        raise ValueError(f"알 수 없는 train_sampling.strategy: {strategy!r}")
    return None


def estimate_memory_gb(n_rows: int, n_columns: int) -> float:
    return n_rows * n_columns * 4 / 1024**3


def to_frame(part, score: np.ndarray) -> pd.DataFrame:
    """평가 모듈이 받는 예측 표. serial 이 없으면 만들 수 없다."""
    if part.serial is None or part.failure_date is None:
        raise ValueError(
            "with_meta=False 로 읽은 구간은 디스크 단위 평가에 쓸 수 없다."
        )
    return pd.DataFrame(
        {
            "serial_number": part.serial,
            "record_date": part.record_date,
            "y": part.y.astype(np.int8),
            "score": score,
            "failure_date": part.failure_date,
        }
    )
