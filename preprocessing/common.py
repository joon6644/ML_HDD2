import os
import time
import duckdb

def run_base_preprocessing(input_file: str, db_file: str, max_memory: str = "6GB", tmp_dir: str = ".tmp") -> tuple[duckdb.DuckDBPyConnection, list[str]]:
    """
    공통 전처리 파이프라인 (Step 1 ~ Step 5) 실행 함수.
    대용량(75M+ 행) 데이터셋에서도 Out of Memory가 발생하지 않도록
    DuckDB 리소스 관리 및 그룹화 기반 중복 제거를 적용합니다.
    
    Args:
        input_file: 원본 parquet 파일 경로
        db_file: 작업용 DuckDB 파일 경로
        max_memory: DuckDB 메모리 제한
        tmp_dir: 임시 디렉토리 경로
        
    Returns:
        con: DuckDB 연결 객체 (view/table들이 생성되어 있는 상태)
        valid_smart_cols: 90% 미만 결측치를 가진 유효 SMART 컬럼 리스트
    """
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except:
            pass
            
    con = duckdb.connect(database=db_file)
    
    # DuckDB 자원 관리 및 메모리 방어 설정
    con.execute(f"PRAGMA max_memory='{max_memory}'")
    con.execute(f"PRAGMA temp_directory='{tmp_dir.replace('\\', '/')}'")
    con.execute("PRAGMA threads=4")
    con.execute("SET preserve_insertion_order=false;")
    
    formatted_input = input_file.replace('\\', '/')
    
    # 1. 컬럼 구조 및 SMART Raw 컬럼 파악
    print("\n[Step 1] 원본 파일 스키마 및 SMART Raw 컬럼 분석...")
    t0 = time.time()
    cols_info = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{formatted_input}')").fetchall()
    all_cols = [c[0] for c in cols_info]
    smart_raw_cols = [c for c in all_cols if c.startswith("smart_") and c.endswith("_raw")]
    total_raw_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{formatted_input}')").fetchone()[0]
    
    print(f"  - 원본 총 행 수: {total_raw_rows:,}개")
    print(f"  - 원본 SMART raw 컬럼 수: {len(smart_raw_cols)}개")
    
    # 2. 결측치 비율 90% 이상인 컬럼 고속 저메모리 필터링 (native COUNT 사용)
    print("\n[Step 2] 결측치 비율 90% 이상 컬럼 고속 저메모리 식별...")
    t0 = time.time()
    count_selects = ", ".join([f'COUNT("{c}") AS "{c}"' for c in smart_raw_cols])
    
    valid_smart_cols = []
    if total_raw_rows > 0 and smart_raw_cols:
        non_null_counts = con.execute(f"SELECT {count_selects} FROM read_parquet('{formatted_input}')").df().iloc[0].to_dict()
        for col, non_null_cnt in non_null_counts.items():
            null_cnt = total_raw_rows - non_null_cnt
            null_ratio = null_cnt / total_raw_rows
            if null_ratio < 0.90:
                valid_smart_cols.append(col)
            else:
                print(f"  - [제거] {col} (결측치 비율: {null_ratio:.2%})")
                
    print(f"  - 완료 (소요시간: {time.time() - t0:.2f}초)")
    print(f"  - 보존 대상 SMART raw 컬럼 수: {len(valid_smart_cols)}개 / 전체 {len(smart_raw_cols)}개")
    
    # 3. 중복 행 제거 (Deduplication) - GROUP BY 사용으로 메모리 차단 및 디스크 스필 지원
    print("\n[Step 3] (serial_number, date) 기준 중복 행 제거 및 최적화...")
    t0 = time.time()
    model_name = os.path.splitext(os.path.basename(input_file))[0]
    model_select = "FIRST(model) AS model" if "model" in all_cols else f"'{model_name}' AS model"
    
    agg_clauses = [
        "serial_number",
        "date",
        model_select,
        "MAX(TRY_CAST(failure AS INTEGER)) AS failure"
    ] + [f'MAX("{c}") AS "{c}"' for c in valid_smart_cols]
    
    agg_str = ", ".join(agg_clauses)
    
    con.execute(f"""
        CREATE OR REPLACE TABLE dedup_data AS
        SELECT {agg_str}
        FROM read_parquet('{formatted_input}')
        GROUP BY serial_number, date
    """)
    
    total_rows = con.execute("SELECT COUNT(*) FROM dedup_data").fetchone()[0]
    print(f"  - 완료 (소요시간: {time.time() - t0:.2f}초)")
    print(f"  - 중복 제거 후 총 행 수: {total_rows:,}개")
    
    # 4. 시계열 공백 분석 및 segment 번호 할당
    print("\n[Step 4] 시계열 공백 분석 및 세그먼트(segment) 분리...")
    t0 = time.time()
    keep_cols = ["date", "serial_number", "model", "failure"]
    cols_to_select = ", ".join([f'"{c}"' for c in (keep_cols + valid_smart_cols)])
    
    con.execute(f"""
        CREATE OR REPLACE VIEW segment_assigned AS
        WITH ordered AS (
            SELECT {cols_to_select},
                   TRY_CAST(date AS DATE) AS record_date,
                   LAG(TRY_CAST(date AS DATE)) OVER (PARTITION BY serial_number ORDER BY TRY_CAST(date AS DATE)) AS prev_date
            FROM dedup_data
        ),
        boundaries AS (
            SELECT *,
                   CASE WHEN prev_date IS NULL OR (record_date - prev_date) > 4 THEN 1 ELSE 0 END AS is_new_seg
            FROM ordered
        )
        SELECT 
            * EXCLUDE(record_date, prev_date, is_new_seg),
            record_date,
            SUM(is_new_seg) OVER (PARTITION BY serial_number ORDER BY record_date) - 1 AS segment
        FROM boundaries
    """)
    print(f"  - 완료 (소요시간: {time.time() - t0:.2f}초)")
    
    # 5. 시계열 날짜 확장 (3일 이하 공백 날짜 생성)
    print("  - 세그먼트 단위 연속 날짜 확장 시퀀스 생성...")
    t0 = time.time()
    con.execute("""
        CREATE OR REPLACE VIEW segment_expanded AS
        WITH segment_min_max AS (
            SELECT 
                serial_number,
                segment,
                MIN(record_date) AS min_date,
                MAX(record_date) AS max_date
            FROM segment_assigned
            GROUP BY serial_number, segment
        )
        SELECT 
            serial_number,
            segment,
            unnest(generate_series(min_date, max_date, INTERVAL '1 day'))::DATE AS record_date
        FROM segment_min_max
    """)
    print(f"  - 완료 (소요시간: {time.time() - t0:.2f}초)")
    
    # 6. 결측 데이터 Forward Fill + Backward Fill fallback 처리
    print("\n[Step 5] 시계열 데이터 결합 및 Forward Fill + Backward Fill 결측치 처리...")
    t0 = time.time()
    col_selects = ", ".join([f'o."{col}"' for col in valid_smart_cols])
    con.execute(f"""
        CREATE OR REPLACE VIEW segment_joined AS
        SELECT 
            e.serial_number,
            e.segment,
            e.record_date,
            o.model,
            o.failure,
            {col_selects}
        FROM segment_expanded e
        LEFT JOIN segment_assigned o
          ON e.serial_number = o.serial_number
             AND e.segment = o.segment
             AND e.record_date = o.record_date
    """)
    
    ffill_selects = []
    ffill_selects.append(
        'COALESCE('
        '  LAST_VALUE("model" IGNORE NULLS) OVER (PARTITION BY serial_number, segment ORDER BY record_date),'
        '  FIRST_VALUE("model" IGNORE NULLS) OVER (PARTITION BY serial_number, segment ORDER BY record_date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)'
        ') AS "model"'
    )
    ffill_selects.append('COALESCE(TRY_CAST("failure" AS INTEGER), 0) AS "failure"')
    for col in valid_smart_cols:
        ffill_selects.append(
            f'COALESCE('
            f'  LAST_VALUE("{col}" IGNORE NULLS) OVER (PARTITION BY serial_number, segment ORDER BY record_date),'
            f'  FIRST_VALUE("{col}" IGNORE NULLS) OVER (PARTITION BY serial_number, segment ORDER BY record_date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)'
            f') AS "{col}"'
        )
        
    ffill_selects_str = ", ".join(ffill_selects)
    
    null_filter = " OR ".join([f'"{col}" IS NOT NULL' for col in valid_smart_cols])
    
    con.execute(f"""
        CREATE OR REPLACE VIEW final_preprocessed AS
        SELECT 
            serial_number,
            record_date::VARCHAR AS date,
            segment,
            {ffill_selects_str}
        FROM segment_joined
        WHERE {null_filter}
    """)
    print(f"  - 완료 (소요시간: {time.time() - t0:.2f}초)")

    return con, valid_smart_cols
