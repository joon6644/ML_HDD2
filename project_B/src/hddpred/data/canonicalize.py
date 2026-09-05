"""raw -> canonical.

project_A/preprocessing/common.py 의 전처리 로직을 이식했다. 단계와 순서는
같고, project_B 용으로 세 가지를 바꿨다.

  1. 원본은 모든 컬럼이 string 이다. 결측률 계산과 중복 제거 전에 DOUBLE로
     캐스팅한다. project_A는 문자열 상태로 MAX()를 걸어 사전식 비교를 했다.
     ("9" > "10") 수치 비교가 맞다.
  2. 출력을 월 단위 partition parquet으로 쓴다. 시간축 split이 필요한 달만
     scan하기 위한 것이다.
  3. 파생 폴더에 provenance.json 과 _SUCCESS 마커를 남긴다.

원본 파일은 열기만 하고 절대 쓰지 않는다.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import duckdb

from .. import config as cfg_mod
from .. import paths
from ..tracking import provenance

CANONICAL_KEYS = ["serial_number", "record_date", "month", "segment", "model", "failure"]


def _quoted(column: str) -> str:
    """안전하게 인용한 DuckDB 식별자."""
    return '"' + column.replace('"', '""') + '"'


def canonical_hash(preprocessing_cfg: dict, drive: "str | dict") -> str:
    """drive 는 이름 문자열이거나 data.yaml 의 드라이브 항목이다.

    항목을 주면 관측 구간 제한(start_date / end_date)이 hash 에 들어간다.
    구간을 바꾸면 canonical 부터 다시 만들어져야 하기 때문이다.
    """
    if isinstance(drive, dict):
        key = {"drive": drive["name"]}
        for field in ("start_date", "end_date"):
            if drive.get(field):
                key[field] = str(drive[field])
    else:
        key = {"drive": drive}
    return cfg_mod.config_hash(preprocessing_cfg, key)


def _connect(preprocessing_cfg: dict) -> duckdb.DuckDBPyConnection:
    duck = preprocessing_cfg.get("duckdb", {})
    tmp_dir = Path(duck.get("temp_dir", paths.TMP_DIR))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA max_memory='{duck.get('max_memory', '16GB')}'")
    con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}'")
    con.execute(f"PRAGMA threads={int(duck.get('threads', 8))}")
    con.execute("SET preserve_insertion_order=false")
    return con


def _observation_filter(drive: dict) -> str:
    """드라이브별 관측 구간 제한 (선택).

    함대가 퇴역하면 남은 소수의 디스크만으로 마지막 달이 구성되어 평가 표본이
    무너진다. ST12000NM0007 은 2019-11 에 37,271대였다가 2026-02 에 979대다.
    그 상태의 마지막 달을 test 로 쓰면 고장이 2건뿐이라 지표가 의미를 잃는다.

    data.yaml 의 드라이브 항목에 start_date / end_date 를 두면 여기서 자른다.
    자르는 기준은 함대 규모이지 성능이 아니므로, 어느 달을 잘랐는지 반드시
    provenance 에 남는다 (canonical hash 에 포함된다).
    """
    clauses = []
    if drive.get("start_date"):
        clauses.append(f"TRY_CAST(date AS DATE) >= DATE '{drive['start_date']}'")
    if drive.get("end_date"):
        clauses.append(f"TRY_CAST(date AS DATE) <= DATE '{drive['end_date']}'")
    return ("WHERE " + " AND ".join(clauses)) if clauses else ""


def build(
    drive: dict,
    data_cfg: dict,
    preprocessing_cfg: dict,
    *,
    force: bool = False,
) -> Path:
    """드라이브 하나의 canonical 레이어를 만들고 그 디렉터리를 돌려준다."""
    name = drive["name"]
    raw_path = Path(data_cfg["raw_root"]) / drive["file"]
    if not raw_path.exists():
        raise FileNotFoundError(f"원본 파일이 없습니다: {raw_path}")

    canon_hash = canonical_hash(preprocessing_cfg, drive)
    out_dir = paths.canonical_dir(name, canon_hash)

    if provenance.is_complete(out_dir) and not force:
        print(f"[canonical] 재사용: {out_dir}")
        return out_dir

    provenance.clear(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"

    con = _connect(preprocessing_cfg)
    raw_uri = raw_path.as_posix()
    started_all = time.time()

    # --- 0. 스키마 확인 ---------------------------------------------------
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{raw_uri}')"
    ).fetchall()
    all_columns = [row[0] for row in schema]
    missing = set(data_cfg["required_columns"]) - set(all_columns)
    if missing:
        raise ValueError(f"{name}: 필수 컬럼 누락 {sorted(missing)}")

    smart_columns = [
        c for c in all_columns if c.startswith("smart_") and c.endswith("_raw")
    ]
    if not smart_columns:
        raise ValueError(f"{name}: smart_*_raw 컬럼이 없습니다.")

    has_model = "model" in all_columns
    model_expr = "model" if has_model else f"'{name}'"

    # --- 1. 필요한 컬럼만 투영하고 수치로 캐스팅 -------------------------
    print(f"\n[1/7] {name}: 컬럼 투영 및 수치 캐스팅 ({len(smart_columns)}개 SMART)")
    cast_sql = ", ".join(
        f"TRY_CAST({_quoted(c)} AS DOUBLE) AS {_quoted(c)}" for c in smart_columns
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW typed_raw AS
        SELECT
            serial_number,
            TRY_CAST(date AS DATE)      AS record_date,
            {model_expr}                AS model,
            TRY_CAST(failure AS INTEGER) AS failure,
            {cast_sql}
        FROM read_parquet('{raw_uri}')
        {_observation_filter(drive)}
        """
    )

    # --- 2. 결측률 90% 이상 컬럼 제거 ------------------------------------
    print("[2/7] 결측률 임계 초과 SMART 컬럼 제거")
    started = time.time()
    threshold = float(preprocessing_cfg["missing_ratio_threshold"])
    count_sql = ", ".join(f"COUNT({_quoted(c)})" for c in smart_columns)
    row_counts = con.execute(
        f"SELECT COUNT(*), {count_sql} FROM typed_raw"
    ).fetchone()
    source_rows = row_counts[0]
    non_null = row_counts[1:]

    valid_columns, dropped = [], []
    for column, count in zip(smart_columns, non_null):
        ratio = 1.0 - (count / source_rows) if source_rows else 1.0
        (dropped if ratio >= threshold else valid_columns).append((column, ratio))
    valid_columns = [c for c, _ in valid_columns]
    if not valid_columns:
        raise ValueError(f"{name}: 결측률 필터 후 남은 SMART 컬럼이 없습니다.")
    print(
        f"      {len(dropped)}개 제거, {len(valid_columns)}개 보존 "
        f"({time.time() - started:.1f}s)"
    )

    # --- 2b. 상수 / 중복 컬럼 제거 ---------------------------------------
    # 결측률 필터를 통과해도 값이 하나뿐이거나 다른 컬럼과 사실상 같은 컬럼이
    # 남는다. 실측: 도시바는 26개 중 10개가 상수였고, smart_222(가동시간)는
    # smart_9(전원인가시간)와 상관 0.9997 이었다. 트리 모델은 이런 컬럼에서
    # 분할하지 않으므로 성능에는 영향이 없지만, 히스토그램을 짓는 시간과
    # 메모리는 그대로 쓴다. 여기서 걷어낸다.
    if preprocessing_cfg.get("drop_constant_columns", False) and valid_columns:
        print("[2b/7] 상수 SMART 컬럼 제거")
        started = time.time()
        distinct_sql = ", ".join(
            f"COUNT(DISTINCT {_quoted(c)})" for c in valid_columns
        )
        counts = con.execute(f"SELECT {distinct_sql} FROM typed_raw").fetchone()
        constants = [c for c, n in zip(valid_columns, counts) if (n or 0) <= 1]
        valid_columns = [c for c in valid_columns if c not in constants]
        if not valid_columns:
            raise ValueError(f"{name}: 상수 제거 후 남은 SMART 컬럼이 없습니다.")
        print(
            f"      {len(constants)}개 제거, {len(valid_columns)}개 보존"
            f" ({time.time() - started:.1f}s)"
        )
        if constants:
            print("      " + ", ".join(constants))

    if preprocessing_cfg.get("drop_duplicate_columns", False) and len(valid_columns) > 1:
        print("[2c/7] 중복 SMART 컬럼 제거 (전 행 완전 일치)")
        started = time.time()
        # 판정 기준은 project_A/preprocessing/ST12000NM0007.py 와 같다 — 모든
        # 행에서 값이 같을 때만 버린다. 상관이 높다는 이유로는 버리지 않는다.
        # 다만 후보 쌍을 손으로 적어두는 대신 전 쌍을 훑는다.
        #
        # 전수로 모든 쌍을 비교하면 스캔이 커지므로 두 단계로 나눈다.
        #   1) 결정적 표본에서 불일치가 하나도 없는 쌍만 추린다.
        #   2) 그 후보만 전 행으로 다시 확인하고, 통과한 것만 버린다.
        sample_rows = int(preprocessing_cfg.get("duplicate_sample_rows", 2_000_000))
        con.execute(
            "CREATE OR REPLACE TEMP TABLE dup_sample AS "
            f"SELECT {', '.join(_quoted(c) for c in valid_columns)} FROM typed_raw "
            f"USING SAMPLE reservoir({sample_rows} ROWS) REPEATABLE (42)"
        )
        pairs = [
            (a, b)
            for i, a in enumerate(valid_columns)
            for b in valid_columns[i + 1:]
        ]

        def _mismatch_counts(table: str, candidates):
            expr = ", ".join(
                f"COUNT(*) FILTER (WHERE {_quoted(a)} IS DISTINCT FROM {_quoted(b)})"
                for a, b in candidates
            )
            return con.execute(f"SELECT {expr} FROM {table}").fetchone()

        candidates = [
            pair for pair, bad in zip(pairs, _mismatch_counts("dup_sample", pairs))
            if (bad or 0) == 0
        ]
        con.execute("DROP TABLE IF EXISTS dup_sample")

        redundant, reasons = set(), []
        if candidates:
            for pair, bad in zip(candidates, _mismatch_counts("typed_raw", candidates)):
                a, b = pair
                if (bad or 0) != 0 or a in redundant or b in redundant:
                    continue
                # 겹치는 쌍에서는 뒤쪽을 버린다. valid_columns 가 원본 스키마
                # 순서(SMART 번호 오름차순)라 더 표준적인 속성이 남는다.
                redundant.add(b)
                reasons.append(f"{b} (== {a})")
        valid_columns = [c for c in valid_columns if c not in redundant]
        print(
            f"      후보 {len(candidates)}쌍 중 {len(redundant)}개 제거, "
            f"{len(valid_columns)}개 보존 ({time.time() - started:.1f}s)"
        )
        if reasons:
            print("      " + ", ".join(reasons))

    # --- 3. (serial_number, date) 중복 제거 ------------------------------
    print("[3/7] (serial_number, date) 중복 행 병합")
    started = time.time()
    agg_sql = ", ".join(
        [
            "MAX(model) AS model",
            "MAX(failure) AS failure",
            *[f"MAX({_quoted(c)}) AS {_quoted(c)}" for c in valid_columns],
        ]
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE deduplicated AS
        SELECT serial_number, record_date, {agg_sql}
        FROM typed_raw
        WHERE record_date IS NOT NULL
        GROUP BY serial_number, record_date
        """
    )
    dedup_rows = con.execute("SELECT COUNT(*) FROM deduplicated").fetchone()[0]
    print(
        f"      {source_rows:,} -> {dedup_rows:,}행 ({time.time() - started:.1f}s)"
    )

    # --- 4. segment 분리 + 누락일 생성 -----------------------------------
    # 관측일 차이가 (max_gap + 1)을 넘으면 새 segment로 본다.
    # segment 경계를 넘는 fill은 하지 않는다.
    max_gap = int(preprocessing_cfg["max_fillable_missing_days"])
    print(f"[4/7] 시계열 공백 분리 (누락 {max_gap}일 이하만 보간)")
    started = time.time()
    feature_sql = ", ".join(_quoted(c) for c in valid_columns)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE segmented AS
        WITH ordered AS (
            SELECT
                serial_number, record_date, model, failure, {feature_sql},
                LAG(record_date) OVER (
                    PARTITION BY serial_number ORDER BY record_date
                ) AS previous_date
            FROM deduplicated
        ),
        boundaries AS (
            SELECT *,
                CASE
                    WHEN previous_date IS NULL
                      OR (record_date - previous_date) > {max_gap + 1}
                    THEN 1 ELSE 0
                END AS starts_new_segment
            FROM ordered
        )
        SELECT
            * EXCLUDE (previous_date, starts_new_segment),
            SUM(starts_new_segment) OVER (
                PARTITION BY serial_number ORDER BY record_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) - 1 AS segment
        FROM boundaries
        """
    )
    print(f"      segment 부여 완료 ({time.time() - started:.1f}s)")

    # --- 5. forward fill (backward fill 없음) ----------------------------
    print("[5/7] Forward fill")
    started = time.time()
    joined_sql = ", ".join(f"observed.{_quoted(c)}" for c in valid_columns)
    fill_columns = ["model", "failure", *valid_columns]
    fill_sql = ", ".join(
        f"""LAST_VALUE({_quoted(c)} IGNORE NULLS) OVER (
                PARTITION BY serial_number, segment ORDER BY record_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS {_quoted(c)}"""
        for c in fill_columns
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW filled AS
        WITH ranges AS (
            SELECT serial_number, segment,
                   MIN(record_date) AS min_date, MAX(record_date) AS max_date
            FROM segmented GROUP BY serial_number, segment
        ),
        expanded AS (
            SELECT serial_number, segment,
                   UNNEST(generate_series(min_date, max_date, INTERVAL '1 day'))::DATE
                   AS record_date
            FROM ranges
        ),
        joined AS (
            SELECT expanded.serial_number, expanded.segment, expanded.record_date,
                   observed.model, observed.failure, {joined_sql}
            FROM expanded
            LEFT JOIN segmented AS observed
              ON expanded.serial_number = observed.serial_number
             AND expanded.segment       = observed.segment
             AND expanded.record_date   = observed.record_date
        )
        SELECT serial_number, record_date, segment, {fill_sql}
        FROM joined
        """
    )
    print(f"      실행 계획 생성 ({time.time() - started:.1f}s)")

    # --- 6. fill 후에도 결측이 남은 행 제거 ------------------------------
    print("[6/7] 잔여 결측 행 제거")
    started = time.time()
    not_null_sql = " AND ".join(f"{_quoted(c)} IS NOT NULL" for c in fill_columns)
    where_clause = f"WHERE {not_null_sql}" if preprocessing_cfg.get(
        "drop_incomplete_rows", True
    ) else ""
    con.execute(
        f"CREATE OR REPLACE TABLE complete_rows AS SELECT * FROM filled {where_clause}"
    )
    complete = con.execute("SELECT COUNT(*) FROM complete_rows").fetchone()[0]
    print(f"      {complete:,}행 ({time.time() - started:.1f}s)")

    # --- 7. 고장 후 정상으로 재기록된 HDD 제거 ---------------------------
    print("[7/7] 고장 후 재기록 HDD 제거 및 partition 저장")
    started = time.time()
    if preprocessing_cfg.get("drop_recovered_disks", True):
        con.execute(
            """
            CREATE OR REPLACE TABLE recovered AS
            SELECT serial_number FROM complete_rows
            GROUP BY serial_number
            HAVING MIN(record_date) FILTER (WHERE failure = 1)
                 < MAX(record_date) FILTER (WHERE failure = 0)
            """
        )
        recovered_count = con.execute("SELECT COUNT(*) FROM recovered").fetchone()[0]
        anti_join = "ANTI JOIN recovered AS bad ON rows.serial_number = bad.serial_number"
    else:
        recovered_count = 0
        anti_join = ""

    final_feature_sql = ", ".join(f"rows.{_quoted(c)}" for c in valid_columns)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE canonical AS
        SELECT
            rows.serial_number,
            rows.record_date,
            strftime(rows.record_date, '%Y-%m') AS month,
            rows.segment::INTEGER              AS segment,
            rows.model,
            rows.failure::TINYINT              AS failure,
            {final_feature_sql}
        FROM complete_rows AS rows
        {anti_join}
        """
    )

    if data_dir.exists():
        # 미완성 상태로 남은 이전 시도를 지운다. _SUCCESS 가 없는 경로만 온다.
        shutil.rmtree(data_dir)
    con.execute(
        f"""
        COPY (SELECT * FROM canonical ORDER BY month, serial_number, record_date)
        TO '{data_dir.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (month), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
        """
    )

    stats = con.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT serial_number),
               COUNT(DISTINCT serial_number) FILTER (WHERE failure = 1),
               MIN(record_date), MAX(record_date)
        FROM canonical
        """
    ).fetchone()
    print(
        f"      {stats[0]:,}행 / {stats[1]:,} 디스크 / 고장 {stats[2]:,}개 "
        f"({time.time() - started:.1f}s)"
    )

    output_schema = {
        row[0]: row[1]
        for row in con.execute("DESCRIBE SELECT * FROM canonical").fetchall()
    }
    provenance.write(
        out_dir,
        stage="canonical",
        config_hash=canon_hash,
        configs={"preprocessing": preprocessing_cfg, "drive": drive},
        inputs=[raw_path.as_posix()],
        output_schema=output_schema,
        stats={
            "source_rows": source_rows,
            "deduplicated_rows": dedup_rows,
            "final_rows": stats[0],
            "n_disks": stats[1],
            "n_failed_disks": stats[2],
            "date_start": str(stats[3]),
            "date_end": str(stats[4]),
            "smart_columns": valid_columns,
            "dropped_for_missingness": [c for c, _ in dropped],
            "recovered_disks_removed": recovered_count,
            "elapsed_seconds": round(time.time() - started_all, 1),
        },
    )
    con.close()
    print(f"[canonical] 완료: {out_dir}")
    return out_dir


def smart_columns(canonical_path: Path) -> list[str]:
    """canonical 레이어에서 살아남은 SMART 컬럼 목록."""
    return list(provenance.read(canonical_path)["stats"]["smart_columns"])


def dataset_glob(canonical_path: Path) -> str:
    """DuckDB read_parquet 에 넣을 glob 패턴."""
    return (canonical_path / "data" / "**" / "*.parquet").as_posix()
