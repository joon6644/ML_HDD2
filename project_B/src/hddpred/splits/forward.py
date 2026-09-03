"""labeled -> split manifest.

시간축 forward chaining split. fold 하나의 구조:

    [-------- train --------][embargo][ val ][embargo][ test ]
                                                       1개월

embargo 가 없으면 train 표본의 라벨 구간 (t, t+H] 가 val/test 구간을 덮는다.
즉 학습이 평가 구간의 고장 사건을 이미 본 상태가 된다. embargo_days >= H 로
두어야 이 누출이 막힌다. tests/test_no_leakage.py 가 이 조건을 검사한다.

split manifest 는 반드시 저장한다. 나중에 "어떤 fold였더라"를 코드에서
역산하지 않기 위해서다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from .. import config as cfg_mod
from .. import paths
from ..labeling import horizon_labels
from ..tracking import provenance

MANIFEST_FILE = "split_manifest.json"


@dataclass
class Fold:
    fold: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    test_month: str
    train_rows: int = 0
    train_positives: int = 0
    train_disks: int = 0
    val_rows: int = 0
    val_positives: int = 0
    val_disks: int = 0
    test_rows: int = 0
    test_positives: int = 0
    test_disks: int = 0
    test_failed_disks: int = 0

    def window(self, part: str) -> tuple[date, date]:
        start = date.fromisoformat(getattr(self, f"{part}_start"))
        end = date.fromisoformat(getattr(self, f"{part}_end"))
        return start, end


def split_hash(split_cfg: dict, label_hash: str) -> str:
    return cfg_mod.config_hash(split_cfg, {"label": label_hash})


def _month_start(month: str) -> date:
    return date.fromisoformat(f"{month}-01")


def _month_end(month: str) -> date:
    return (pd.Timestamp(_month_start(month)) + pd.offsets.MonthEnd(0)).date()


def _shift_months(value: date, months: int) -> date:
    return (pd.Timestamp(value) + pd.DateOffset(months=months)).date()


def plan_folds(
    months: list[str],
    split_cfg: dict,
    embargo_days: int,
    *,
    data_end: date | None = None,
    horizon_days: int | None = None,
) -> list[Fold]:
    """날짜 경계만 계산한다. 데이터는 읽지 않는다.

    data_end 와 horizon_days 를 주면 마지막 달을 test 로 쓸지 판단한다.
    관측 종료 직전 H일은 생존을 확인할 수 없어 표본에서 빠지므로, 그 구간이
    걸친 달을 test 로 쓰면 test 창이 조용히 잘린 채로 채점된다.
    """
    months = sorted(months)
    if not months:
        raise ValueError("라벨 레이어에 월이 없습니다.")

    data_start = _month_start(months[0])
    val_months = int(split_cfg["val_months"])
    test_months = int(split_cfg["test_months"])
    if test_months != 1:
        raise NotImplementedError(
            "이 연구 설정은 test 구간 1개월 고정이다. "
            "configs/split.yaml 의 test_months 를 1로 두어라."
        )
    min_train_months = int(split_cfg["min_train_months"])
    n_folds = split_cfg.get("n_folds")

    candidates = list(reversed(months))
    if (
        split_cfg.get("exclude_incomplete_tail", True)
        and data_end is not None
        and horizon_days is not None
    ):
        complete = [
            m
            for m in candidates
            if _month_end(m) + timedelta(days=horizon_days) <= data_end
        ]
        dropped = [m for m in candidates if m not in set(complete)]
        if dropped:
            print(
                f"  [splits] 라벨이 잘리는 말단 {len(dropped)}개월 제외: "
                f"{', '.join(sorted(dropped))}"
            )
        candidates = complete
    if n_folds is not None:
        candidates = candidates[: int(n_folds)]

    folds: list[Fold] = []
    for test_month in sorted(candidates):
        test_start = _month_start(test_month)
        test_end = _month_end(test_month)

        val_end = test_start - timedelta(days=1 + embargo_days)
        val_start = _shift_months(val_end, -val_months) + timedelta(days=1)
        train_end = val_start - timedelta(days=1 + embargo_days)

        if split_cfg["train_window"] == "rolling":
            train_start = _shift_months(
                train_end, -int(split_cfg["train_months"])
            ) + timedelta(days=1)
            train_start = max(train_start, data_start)
        else:
            train_start = data_start

        # 개월 수는 달력으로 센다. 일수/30.44 로 근사하면 정확히 3개월인 창이
        # 2.99 로 나와 min_train_months=3 조건에서 전부 걸러진다.
        latest_allowed_start = _shift_months(train_end, -min_train_months) + timedelta(
            days=1
        )
        if train_start > latest_allowed_start:
            continue

        folds.append(
            Fold(
                fold=len(folds),
                train_start=train_start.isoformat(),
                train_end=train_end.isoformat(),
                val_start=val_start.isoformat(),
                val_end=val_end.isoformat(),
                test_start=test_start.isoformat(),
                test_end=test_end.isoformat(),
                test_month=test_month,
            )
        )
    return folds


def _window_stats(
    con: duckdb.DuckDBPyConnection,
    source: str,
    start: date,
    end: date,
    *,
    horizon_days: int,
    censoring_scope: str,
) -> tuple[int, int, int, int]:
    """구간 통계.

    반드시 fold 적재와 같은 라벨 기준을 써야 한다. 다르면 manifest 가
    "train 양성 7" 이라고 적어놓고 실제로는 0개가 실리는 일이 생긴다.
    (창 기준 검열을 켜면 창 끝 10일과 창 밖 고장이 빠지므로 값이 달라진다.)
    """
    if censoring_scope == "window":
        confirmed = (
            f"(failure_date IS NOT NULL AND failure_date <= DATE '{end}'"
            f" AND date_diff('day', record_date, failure_date) <= {horizon_days})"
        )
        survived = f"(record_date + INTERVAL {horizon_days} DAY <= DATE '{end}')"
        label_sql = f"CASE WHEN {confirmed} THEN 1 ELSE 0 END"
        keep_sql = f"AND ({confirmed} OR {survived})"
    else:
        label_sql = "y"
        keep_sql = ""

    # month 조건을 함께 걸어 hive partition pruning 이 걸리게 한다. 이것이
    # 없으면 fold 마다 라벨 전체를 훑는다.
    row = con.execute(
        f"""
        SELECT COUNT(*), COALESCE(SUM({label_sql}), 0),
               COUNT(DISTINCT serial_number),
               COUNT(DISTINCT serial_number) FILTER (
                   WHERE failure_date BETWEEN DATE '{start}' AND DATE '{end}'
               )
        FROM read_parquet('{source}', hive_partitioning=true)
        WHERE month BETWEEN '{start:%Y-%m}' AND '{end:%Y-%m}'
          AND record_date BETWEEN DATE '{start}' AND DATE '{end}'
        {keep_sql}
        """
    ).fetchone()
    return int(row[0]), int(row[1]), int(row[2]), int(row[3])


def build(
    drive_name: str,
    labels_path: Path,
    split_cfg: dict,
    labeling_cfg: dict,
    *,
    force: bool = False,
) -> Path:
    """split manifest 를 만들고 그 디렉터리를 돌려준다."""
    lab_hash = provenance.read(labels_path)["config_hash"]
    horizon = int(labeling_cfg["horizon_days"])
    # fold 유효성 검사가 라벨 기준에 따라 달라지므로 hash 에 포함시킨다.
    # 키 이름이 censoring_scope 면 config._HASH_EXCLUDE 에 걸려 빠진다.
    scope = labeling_cfg.get("censoring_scope", "global")

    embargo = split_cfg.get("embargo_days")
    embargo = horizon if embargo is None else int(embargo)
    # embargo 는 창 사이를 비워 train 라벨 구간 (t, t+H] 가 다음 창을 덮지
    # 못하게 한다. censoring_scope: window 는 같은 누출을 다른 방법으로 막는다.
    #
    #   y=1  <=>  failure_date <= 창끝  AND  failure_date - t <= H
    #   y=0  <=>  t + H <= 창끝                (생존을 창 안에서 확인 가능)
    #   그 외 -> 라벨 없음, 표본에서 제외
    #
    # 다음 창에서 고장난 디스크는 첫 조건에 걸리지 않고 둘째 조건도 만족하지
    # 못해 그냥 빠진다. 라벨이 그 고장을 참조할 경로가 없다.
    #
    # embargo 와 다른 점은 무엇을 버리느냐다. embargo 는 경계 앞 H일을 통째로
    # (양성·음성 모두) 비우는데, window 검열은 창 안에서 고장이 확인된 디스크의
    # 행은 남긴다. 고장 사건이 수십 건뿐인 이 문제에서 그 차이가 크다.
    #
    # 그래서 scope 가 window 면 embargo < horizon 이어도 경고하지 않는다.
    if embargo < horizon and scope == "global":
        if not split_cfg.get("allow_label_overlap", False):
            raise ValueError(
                f"embargo_days({embargo}) < horizon_days({horizon}) 인데 "
                "labeling.censoring_scope 가 global 이다. train 라벨이 평가 "
                "구간의 고장을 보게 되어 누출이 발생한다. censoring_scope 를 "
                "window 로 두거나, 의도한 것이면 split.allow_label_overlap 을 "
                "true 로 두어라."
            )
        print(
            f"  [splits] ⚠ embargo({embargo}d) < horizon({horizon}d) 이고 "
            "censoring_scope 가 global 이다. 구간 경계 앞 "
            + str(horizon - embargo)
            + "일의 학습 라벨이 다음 구간의 고장을 참조한다 "
            "(allow_label_overlap 승인됨)."
        )
    resolved_split = dict(
        split_cfg, embargo_days=embargo, label_censoring_scope=scope
    )
    sp_hash = split_hash(resolved_split, lab_hash)
    out_dir = paths.splits_dir(drive_name, sp_hash)

    if provenance.is_complete(out_dir) and not force:
        print(f"[splits] 재사용: {out_dir}")
        return out_dir

    provenance.clear(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_stats = provenance.read(labels_path)["stats"]
    months = [m["month"] for m in label_stats["monthly"]]
    folds = plan_folds(
        months,
        resolved_split,
        embargo,
        data_end=date.fromisoformat(str(label_stats["date_end"])[:10]),
        horizon_days=horizon,
    )
    if not folds:
        raise ValueError(
            "조건을 만족하는 fold가 없습니다. min_train_months 를 줄이거나 "
            "n_folds 를 줄여보세요."
        )

    source = horizon_labels.dataset_glob(labels_path)
    con = duckdb.connect(database=":memory:")

    kept: list[Fold] = []
    for fold in folds:
        for part in ("train", "val", "test"):
            start, end = fold.window(part)
            rows, positives, disks, failed = _window_stats(
                con, source, start, end, horizon_days=horizon, censoring_scope=scope
            )
            setattr(fold, f"{part}_rows", rows)
            setattr(fold, f"{part}_positives", positives)
            setattr(fold, f"{part}_disks", disks)
            if part == "test":
                fold.test_failed_disks = failed

        # 세 구간 중 하나라도 양성이 없으면 그 달은 건너뛴다.
        #   train 양성 0 -> 학습할 신호가 없다
        #   val   양성 0 -> 임곗값을 고를 수 없다
        #   test  양성 0 -> TP/FN 이 정의되지 않아 recall 이 0/0 이다
        # 여기 쓰는 통계는 fold 적재와 같은 라벨 기준으로 센 값이다.
        minimums = {
            "train": int(split_cfg.get("min_train_failures", 1)),
            "val": int(split_cfg.get("min_val_failures", 1)),
            "test": int(split_cfg.get("min_test_failures", 1)),
        }
        rejected = None
        for part, minimum in minimums.items():
            positives = getattr(fold, f"{part}_positives")
            if positives < minimum:
                rejected = (part, positives, minimum)
                break
        if rejected:
            part, positives, minimum = rejected
            print(
                f"  [skip] fold {fold.test_month}: {part} 양성 {positives} < {minimum}"
            )
            continue
        kept.append(fold)

    if not kept:
        raise ValueError("유효성 검사를 통과한 fold가 없습니다.")

    for index, fold in enumerate(kept):
        fold.fold = index

    manifest = {
        "drive": drive_name,
        "protocol": resolved_split["protocol"],
        "train_window": resolved_split["train_window"],
        "horizon_days": horizon,
        "embargo_days": embargo,
        "split_hash": sp_hash,
        "label_hash": lab_hash,
        "data_months": [months[0], months[-1]],
        "n_folds": len(kept),
        "folds": [asdict(fold) for fold in kept],
    }
    with (out_dir / MANIFEST_FILE).open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"\n[splits] {drive_name}: fold {len(kept)}개")
    for fold in kept:
        print(
            f"  fold {fold.fold:02d} | train {fold.train_start}~{fold.train_end} "
            f"({fold.train_rows:,}행, 양성 {fold.train_positives:,}) | "
            f"val {fold.val_start}~{fold.val_end} ({fold.val_positives:,}) | "
            f"test {fold.test_month} ({fold.test_disks:,} 디스크, "
            f"고장 {fold.test_failed_disks})"
        )

    provenance.write(
        out_dir,
        stage="splits",
        config_hash=sp_hash,
        configs={"split": resolved_split, "labeling": labeling_cfg},
        parents={"labels": lab_hash},
        inputs=[labels_path.as_posix()],
        stats={"n_folds": len(kept), "months": months},
    )
    con.close()
    return out_dir


def load_manifest(splits_path: Path) -> dict:
    with (splits_path / MANIFEST_FILE).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_folds(splits_path: Path) -> list[Fold]:
    return [Fold(**f) for f in load_manifest(splits_path)["folds"]]
