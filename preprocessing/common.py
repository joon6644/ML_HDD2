import os
import time

import duckdb


KEY_COLUMNS = {"serial_number", "date", "model", "failure"}
MISSING_RATIO_THRESHOLD = 0.90
MAX_FILLABLE_MISSING_DAYS = 3


def _quoted(column: str) -> str:
    """Return a safely quoted DuckDB identifier."""
    return '"' + column.replace('"', '""') + '"'


def run_base_preprocessing(
    input_file: str,
    db_file: str,
    max_memory: str = "6GB",
    tmp_dir: str = ".tmp",
) -> tuple[duckdb.DuckDBPyConnection, list[str]]:
    """
    HDD SMART 원본 데이터의 공통 전처리를 수행한다.

    처리 순서:
      1. (serial_number, date) 중복 행 제거
      2. normalized SMART 및 불필요 컬럼 제거 (smart_*_raw만 보존)
      3. 결측치 비율이 90% 이상인 SMART 컬럼 제거
      4. 누락일 3일 이하는 날짜 행 생성 후 forward fill,
         누락일 4일 이상은 별도 segment로 분리
      5. forward fill 후 남은 결측치 backward fill
      6. 두 방향 채움 후에도 SMART 결측치가 남은 행 제거
      7. failure=1 이후 다시 failure=0이 관측된 HDD 전체 제거

    수치 표준화(StandardScaler)는 데이터 누수를 막기 위해 여기서 수행하지
    않고, train/validation/test 분할 후 data_loader에서 train 기준으로 수행한다.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {input_file}")

    os.makedirs(tmp_dir, exist_ok=True)
    if os.path.exists(db_file):
        os.remove(db_file)

    con = duckdb.connect(database=db_file)
    formatted_input = input_file.replace("\\", "/")
    formatted_tmp_dir = tmp_dir.replace("\\", "/")

    con.execute(f"PRAGMA max_memory='{max_memory}'")
    con.execute(f"PRAGMA temp_directory='{formatted_tmp_dir}'")
    con.execute("PRAGMA threads=4")
    con.execute("SET preserve_insertion_order=false")

    print("\n[준비] 원본 스키마 확인...")
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{formatted_input}')"
    ).fetchall()
    all_columns = [row[0] for row in schema]
    missing_keys = KEY_COLUMNS - set(all_columns)
    if missing_keys:
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing_keys)}")

    smart_raw_columns = [
        col
        for col in all_columns
        if col.startswith("smart_") and col.endswith("_raw")
    ]
    if not smart_raw_columns:
        raise ValueError("보존할 smart_*_raw 컬럼이 없습니다.")

    # 1. 동일 HDD/날짜 중복을 하나로 합친다. failure와 SMART 값은 MAX를
    # 사용하여 중복 중 고장 표시 및 비결측 측정값이 유실되지 않게 한다.
    print("\n[Step 1/7] (serial_number, date) 중복 행 제거...")
    started = time.time()
    aggregate_columns = []
    for col in all_columns:
        if col in {"serial_number", "date"}:
            continue
        quoted = _quoted(col)
        if col == "failure":
            aggregate_columns.append(
                f"MAX(TRY_CAST({quoted} AS INTEGER)) AS {quoted}"
            )
        else:
            aggregate_columns.append(f"MAX({quoted}) AS {quoted}")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE deduplicated_data AS
        SELECT
            serial_number,
            date,
            {", ".join(aggregate_columns)}
        FROM read_parquet('{formatted_input}')
        GROUP BY serial_number, date
        """
    )
    source_rows, deduplicated_rows = con.execute(
        f"""
        SELECT
            (SELECT COUNT(*) FROM read_parquet('{formatted_input}')),
            (SELECT COUNT(*) FROM deduplicated_data)
        """
    ).fetchone()
    print(
        f"  - {source_rows:,} → {deduplicated_rows:,}행 "
        f"({source_rows - deduplicated_rows:,}행 제거, {time.time() - started:.2f}초)"
    )

    # 2. 식별/라벨 컬럼과 raw SMART만 남긴다. smart_*_normalized 및 그 밖의
    # 원본 메타데이터는 이 단계에서 제거한다.
    print("\n[Step 2/7] normalized SMART 및 불필요 컬럼 제거...")
    retained_columns = [
        "serial_number",
        "date",
        "model",
        "failure",
        *smart_raw_columns,
    ]
    retained_sql = ", ".join(_quoted(col) for col in retained_columns)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE raw_features_only AS
        SELECT {retained_sql}
        FROM deduplicated_data
        """
    )
    removed_columns = [col for col in all_columns if col not in retained_columns]
    print(
        f"  - {len(removed_columns)}개 제거, SMART raw {len(smart_raw_columns)}개 보존"
    )

    # 3. 중복 제거 후의 데이터 기준으로 결측률을 계산한다.
    print("\n[Step 3/7] 결측치 비율 90% 이상 SMART 컬럼 제거...")
    started = time.time()
    total_rows = con.execute("SELECT COUNT(*) FROM raw_features_only").fetchone()[0]
    count_sql = ", ".join(
        f"COUNT({_quoted(col)}) AS {_quoted(col)}" for col in smart_raw_columns
    )
    non_null_counts = con.execute(
        f"SELECT {count_sql} FROM raw_features_only"
    ).fetchone()

    valid_smart_cols = []
    removed_for_missingness = []
    for col, non_null_count in zip(smart_raw_columns, non_null_counts):
        missing_ratio = 1.0 - (non_null_count / total_rows) if total_rows else 1.0
        if missing_ratio >= MISSING_RATIO_THRESHOLD:
            removed_for_missingness.append((col, missing_ratio))
        else:
            valid_smart_cols.append(col)

    if not valid_smart_cols:
        raise ValueError("결측률 필터 후 남은 SMART raw 컬럼이 없습니다.")

    selected_columns = [
        "serial_number",
        "date",
        "model",
        "failure",
        *valid_smart_cols,
    ]
    selected_sql = ", ".join(_quoted(col) for col in selected_columns)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE selected_features AS
        SELECT {selected_sql}
        FROM raw_features_only
        """
    )
    for col, ratio in removed_for_missingness:
        print(f"  - 제거: {col} (결측률 {ratio:.2%})")
    print(
        f"  - SMART raw {len(valid_smart_cols)}개 보존 "
        f"({time.time() - started:.2f}초)"
    )

    # 4. 관측일 차이가 4일이면 실제 누락일은 3일이다. 따라서 날짜 차이가
    # MAX_FILLABLE_MISSING_DAYS + 1보다 클 때(누락일 4일 이상) 새 segment다.
    print("\n[Step 4/7] 시계열 공백 분리, 누락 날짜 생성 및 forward fill...")
    started = time.time()
    feature_sql = ", ".join(_quoted(col) for col in valid_smart_cols)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE segment_assigned AS
        WITH ordered AS (
            SELECT
                serial_number,
                TRY_CAST(date AS DATE) AS record_date,
                model,
                TRY_CAST(failure AS INTEGER) AS failure,
                {feature_sql},
                LAG(TRY_CAST(date AS DATE)) OVER (
                    PARTITION BY serial_number
                    ORDER BY TRY_CAST(date AS DATE)
                ) AS previous_date
            FROM selected_features
        ),
        boundaries AS (
            SELECT
                *,
                CASE
                    WHEN previous_date IS NULL
                      OR (record_date - previous_date)
                         > {MAX_FILLABLE_MISSING_DAYS + 1}
                    THEN 1 ELSE 0
                END AS starts_new_segment
            FROM ordered
        )
        SELECT
            * EXCLUDE (previous_date, starts_new_segment),
            SUM(starts_new_segment) OVER (
                PARTITION BY serial_number
                ORDER BY record_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) - 1 AS segment
        FROM boundaries
        """
    )

    joined_feature_sql = ", ".join(
        f'observed.{_quoted(col)}' for col in valid_smart_cols
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW expanded_and_joined AS
        WITH ranges AS (
            SELECT
                serial_number,
                segment,
                MIN(record_date) AS min_date,
                MAX(record_date) AS max_date
            FROM segment_assigned
            GROUP BY serial_number, segment
        ),
        expanded AS (
            SELECT
                serial_number,
                segment,
                UNNEST(
                    generate_series(min_date, max_date, INTERVAL '1 day')
                )::DATE AS record_date
            FROM ranges
        )
        SELECT
            expanded.serial_number,
            expanded.segment,
            expanded.record_date,
            observed.model,
            observed.failure,
            {joined_feature_sql}
        FROM expanded
        LEFT JOIN segment_assigned AS observed
          ON expanded.serial_number = observed.serial_number
         AND expanded.segment = observed.segment
         AND expanded.record_date = observed.record_date
        """
    )

    columns_to_fill = ["model", "failure", *valid_smart_cols]
    forward_fill_sql = ", ".join(
        f"""
        LAST_VALUE({_quoted(col)} IGNORE NULLS) OVER (
            PARTITION BY serial_number, segment
            ORDER BY record_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS {_quoted(col)}
        """
        for col in columns_to_fill
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW forward_filled AS
        SELECT
            serial_number,
            record_date,
            segment,
            {forward_fill_sql}
        FROM expanded_and_joined
        """
    )
    expanded_rows = con.execute(
        "SELECT COUNT(*) FROM forward_filled"
    ).fetchone()[0]
    print(
        f"  - 누락 날짜 {expanded_rows - total_rows:,}행 생성 "
        f"({time.time() - started:.2f}초)"
    )

    # 5. segment 시작부 등 forward fill로 채우지 못한 값만 미래 관측값으로
    # backward fill한다.
    print("\n[Step 5/7] 남은 결측치 backward fill...")
    started = time.time()
    backward_fill_sql = ", ".join(
        f"""
        COALESCE(
            {_quoted(col)},
            FIRST_VALUE({_quoted(col)} IGNORE NULLS) OVER (
                PARTITION BY serial_number, segment
                ORDER BY record_date
                ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
            )
        ) AS {_quoted(col)}
        """
        for col in columns_to_fill
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW fully_filled AS
        SELECT
            serial_number,
            record_date,
            segment,
            {backward_fill_sql}
        FROM forward_filled
        """
    )
    print(f"  - 완료 ({time.time() - started:.2f}초)")

    # 6. 라벨을 임의의 0으로 바꾸지 않는다. model/failure/SMART 중 하나라도
    # 두 방향 채움 후 NULL이면 해당 행을 제거한다.
    print("\n[Step 6/7] 전처리 후에도 결측치가 남은 행 제거...")
    started = time.time()
    required_non_null_sql = " AND ".join(
        f"{_quoted(col)} IS NOT NULL" for col in columns_to_fill
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE complete_rows AS
        SELECT *
        FROM fully_filled
        WHERE {required_non_null_sql}
        """
    )
    complete_rows = con.execute("SELECT COUNT(*) FROM complete_rows").fetchone()[0]
    print(
        f"  - {expanded_rows - complete_rows:,}행 제거 "
        f"({time.time() - started:.2f}초)"
    )

    # 7. 같은 HDD에서 failure=1이 먼저 나타난 뒤 더 늦은 날짜에 failure=0이
    # 나타나면 수리/재투입 또는 라벨 이상으로 보고 그 HDD 전체를 제거한다.
    print("\n[Step 7/7] 고장 후 정상으로 재기록된 HDD 제거...")
    started = time.time()
    con.execute(
        """
        CREATE OR REPLACE TABLE invalid_recovered_hdds AS
        WITH failure_history AS (
            SELECT
                serial_number,
                record_date,
                failure,
                MAX(failure) OVER (
                    PARTITION BY serial_number
                    ORDER BY record_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS previous_max_failure
            FROM complete_rows
        )
        SELECT DISTINCT serial_number
        FROM failure_history
        WHERE previous_max_failure = 1
          AND failure = 0
        """
    )
    invalid_hdd_count = con.execute(
        "SELECT COUNT(*) FROM invalid_recovered_hdds"
    ).fetchone()[0]

    final_feature_sql = ", ".join(_quoted(col) for col in valid_smart_cols)
    con.execute(
        f"""
        CREATE OR REPLACE VIEW final_preprocessed AS
        SELECT
            rows.serial_number,
            rows.record_date::VARCHAR AS date,
            rows.segment,
            rows.model,
            TRY_CAST(rows.failure AS INTEGER) AS failure,
            {final_feature_sql}
        FROM complete_rows AS rows
        ANTI JOIN invalid_recovered_hdds AS invalid
          ON rows.serial_number = invalid.serial_number
        """
    )
    final_rows = con.execute(
        "SELECT COUNT(*) FROM final_preprocessed"
    ).fetchone()[0]
    print(
        f"  - HDD {invalid_hdd_count:,}개 제거, 최종 {final_rows:,}행 "
        f"({time.time() - started:.2f}초)"
    )

    return con, valid_smart_cols
