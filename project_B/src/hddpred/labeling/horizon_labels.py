"""canonical -> labeled.

문제 정의:

    관측 시점 t 까지의 SMART 이력만 사용하여
    "이 HDD가 (t, t + H] 안에 고장나는가?" 를 예측한다.

라벨은 정의상 미래 정보를 쓴다. 그것 자체는 누출이 아니다. 누출은 feature가
t 이후를 보는 경우다. 이 모듈은 라벨만 만들고 feature는 건드리지 않는다.

여기서 처리하는 세 가지 함정:

  1. 고장 당일 이후 행
     이미 고장난 디스크의 고장을 예측할 수는 없다. t >= failure_date 인 행을
     표본에서 제외한다.

  2. 우측 검열 (censoring)
     미고장 디스크의 마지막 H일 구간은 "고장이 없었다"가 아니라 "확인할 수
     없다"이다. 이 행을 음성으로 넣으면 관측 종료 직전 구간이 전부 가짜
     음성이 된다. last_date >= t + H 인 행만 남긴다.

  3. 관측 시작부
     이력이 짧은 구간은 롤링 피처와 시퀀스 lookback을 채우지 못한다.
     min_history_days 이전 행을 제외한다.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import duckdb

from .. import config as cfg_mod
from .. import paths
from ..data import canonicalize
from ..tracking import provenance


def label_hash(labeling_cfg: dict, canon_hash: str) -> str:
    return cfg_mod.config_hash(labeling_cfg, {"canon": canon_hash})


def build(
    drive_name: str,
    canonical_path: Path,
    labeling_cfg: dict,
    preprocessing_cfg: dict,
    *,
    force: bool = False,
) -> Path:
    """라벨 레이어를 만들고 그 디렉터리를 돌려준다."""
    canon_hash = provenance.read(canonical_path)["config_hash"]
    lab_hash = label_hash(labeling_cfg, canon_hash)
    out_dir = paths.labels_dir(drive_name, canon_hash, lab_hash)

    if provenance.is_complete(out_dir) and not force:
        print(f"[labels] 재사용: {out_dir}")
        return out_dir

    provenance.clear(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"

    horizon = int(labeling_cfg["horizon_days"])
    min_history = int(labeling_cfg.get("min_history_days", 0))
    exclude_post_failure = bool(labeling_cfg.get("exclude_post_failure_rows", True))
    require_observability = bool(labeling_cfg.get("require_horizon_observability", True))

    con = duckdb.connect(database=":memory:")
    duck = preprocessing_cfg.get("duckdb", {})
    con.execute(f"PRAGMA threads={int(duck.get('threads', 8))}")
    con.execute(f"PRAGMA max_memory='{duck.get('max_memory', '16GB')}'")
    con.execute(f"PRAGMA temp_directory='{Path(duck.get('temp_dir', paths.TMP_DIR)).as_posix()}'")
    con.execute("SET preserve_insertion_order=false")

    source = canonicalize.dataset_glob(canonical_path)
    started = time.time()
    print(f"\n[labels] {drive_name}: horizon={horizon}d")

    # 디스크 단위 사실: 고장일, 최초/최종 관측일.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE disk_facts AS
        SELECT
            serial_number,
            MIN(record_date) FILTER (WHERE failure = 1) AS failure_date,
            MIN(record_date) AS first_date,
            MAX(record_date) AS last_date,
            COUNT(*)         AS n_observations
        FROM read_parquet('{source}', hive_partitioning=true)
        GROUP BY serial_number
        """
    )

    conditions = ["TRUE"]
    if exclude_post_failure:
        conditions.append(
            "(d.failure_date IS NULL OR c.record_date < d.failure_date)"
        )
    if require_observability:
        conditions.append(
            f"(d.failure_date IS NOT NULL "
            f"OR d.last_date >= c.record_date + INTERVAL {horizon} DAY)"
        )
    if min_history > 0:
        conditions.append(
            f"date_diff('day', d.first_date, c.record_date) >= {min_history}"
        )
    where_sql = "\n          AND ".join(conditions)

    con.execute(
        f"""
        CREATE OR REPLACE TABLE labels AS
        SELECT
            c.serial_number,
            c.record_date,
            c.month,
            c.segment,
            d.failure_date,
            CASE WHEN d.failure_date IS NULL THEN NULL
                 ELSE date_diff('day', c.record_date, d.failure_date)
            END::INTEGER AS days_to_failure,
            (d.failure_date IS NOT NULL)::TINYINT AS is_failed_disk,
            CASE
                WHEN d.failure_date IS NOT NULL
                 AND date_diff('day', c.record_date, d.failure_date) <= {horizon}
                THEN 1 ELSE 0
            END::TINYINT AS y
        FROM read_parquet('{source}', hive_partitioning=true) AS c
        JOIN disk_facts AS d USING (serial_number)
        WHERE {where_sql}
        """
    )

    if data_dir.exists():
        shutil.rmtree(data_dir)
    con.execute(
        f"""
        COPY (SELECT * FROM labels ORDER BY month, serial_number, record_date)
        TO '{data_dir.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (month), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
        """
    )

    total, positives, disks, failed_disks, start, end = con.execute(
        """
        SELECT COUNT(*), SUM(y), COUNT(DISTINCT serial_number),
               COUNT(DISTINCT serial_number) FILTER (WHERE is_failed_disk = 1),
               MIN(record_date), MAX(record_date)
        FROM labels
        """
    ).fetchone()
    canonical_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{source}', hive_partitioning=true)"
    ).fetchone()[0]

    positives = int(positives or 0)
    rate = positives / total if total else 0.0
    print(
        f"  {canonical_rows:,} -> {total:,}행 (표본), 양성 {positives:,} "
        f"({rate:.4%}), 디스크 {disks:,} / 고장 {failed_disks:,} "
        f"({time.time() - started:.1f}s)"
    )

    monthly = con.execute(
        """
        SELECT month, COUNT(*) AS n_rows, SUM(y) AS n_positive,
               COUNT(DISTINCT serial_number) AS n_disks,
               COUNT(DISTINCT serial_number) FILTER (
                   WHERE failure_date IS NOT NULL
                     AND strftime(failure_date, '%Y-%m') = month
               ) AS n_failures_in_month
        FROM labels GROUP BY month ORDER BY month
        """
    ).fetchall()

    output_schema = {
        row[0]: row[1]
        for row in con.execute("DESCRIBE SELECT * FROM labels").fetchall()
    }
    provenance.write(
        out_dir,
        stage="labels",
        config_hash=lab_hash,
        configs={"labeling": labeling_cfg},
        parents={"canonical": canon_hash},
        inputs=[canonical_path.as_posix()],
        output_schema=output_schema,
        stats={
            "horizon_days": horizon,
            "canonical_rows": canonical_rows,
            "sample_rows": total,
            "positive_rows": positives,
            "positive_rate": rate,
            "n_disks": disks,
            "n_failed_disks": failed_disks,
            "date_start": str(start),
            "date_end": str(end),
            "monthly": [
                {
                    "month": m,
                    "n_rows": int(r),
                    "n_positive": int(p or 0),
                    "n_disks": int(dd),
                    "n_failures_in_month": int(f),
                }
                for m, r, p, dd, f in monthly
            ],
        },
    )
    con.close()
    print(f"[labels] 완료: {out_dir}")
    return out_dir


def dataset_glob(labels_path: Path) -> str:
    return (labels_path / "data" / "**" / "*.parquet").as_posix()


def monthly_stats(labels_path: Path) -> list[dict]:
    return provenance.read(labels_path)["stats"]["monthly"]
