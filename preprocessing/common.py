import os
import time
import duckdb

def run_base_preprocessing(input_file: str, db_file: str, max_memory: str = "6GB", tmp_dir: str = ".tmp") -> tuple[duckdb.DuckDBPyConnection, list[str]]:
    """
    공통 전처리 파이프라인 (Step 1 ~ Step 5) 실행 함수.
    
    Args:
        input_file: 원본 parquet 파일 경로
        db_file: 작업용 DuckDB 파일 경로
        max_memory: DuckDB 메모리 제한
        tmp_dir: 임시 디렉토리 경로
        
    Returns:
        con: DuckDB 연결 객체 (view들이 생성되어 있는 상태)
        valid_smart_cols: 90% 미만 결측치를 가진 유효 SMART 컬럼 리스트
    """
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except:
            pass
            
    con = duckdb.connect(database=db_file)
    
    # DuckDB 자원 관리 설정
    con.execute(f"PRAGMA max_memory='{max_memory}'")
    con.execute(f"PRAGMA temp_directory='{tmp_dir.replace('\\', '/')}'")
    
    # 1. 중복 행 제거
    # (serial_number, date)별 1행만 남김. failure가 1인 행을 우선적으로 남김.
    print("\n[Step 1] 중복 행 제거 (Deduplication)...")
    t0 = time.time()
    con.execute(f"""
        CREATE OR REPLACE VIEW dedup_data AS
        SELECT * EXCLUDE(row_num)
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY serial_number, date ORDER BY failure DESC) AS row_num
            FROM read_parquet('{input_file.replace('\\', '/')}')
        )
        WHERE row_num = 1
    """)
    total_rows = con.execute("SELECT COUNT(*) FROM dedup_data").fetchone()[0]
    print(f"  - 완료 (소요시간: {time.time() - t0:.2f}초)")
    print(f"  - 중복 제거 후 총 행 수: {total_rows:,}개")
    
    # 2. nomalization 컬럼 제거, 분석에 불필요한 컬럼 제거
    print("\n[Step 2] normalization 컬럼 및 불필요 컬럼 식별...")
    columns = con.execute("DESCRIBE dedup_data").df()["column_name"].tolist()
    keep_cols = ["date", "serial_number", "model", "failure"]
    smart_raw_cols = [c for c in columns if c.startswith("smart_") and c.endswith("_raw")]
    print(f"  - 원본 SMART raw 컬럼 수: {len(smart_raw_cols)}개")
    
    # 3. 결측치 비율 90% 이상인 컬럼 제거
    print("\n[Step 3] 결측치 비율 90% 이상인 컬럼 제거...")
    t0 = time.time()
    null_selects = ", ".join([f'SUM(CASE WHEN TRY_CAST("{c}" AS DOUBLE) IS NULL THEN 1 ELSE 0 END) AS "{c}"' for c in smart_raw_cols])
    
    valid_smart_cols = []
    if total_rows > 0 and smart_raw_cols:
        null_counts = con.execute(f"SELECT {null_selects} FROM dedup_data").df().iloc[0].to_dict()
        for col, null_cnt in null_counts.items():
            null_ratio = null_cnt / total_rows
            if null_ratio < 0.90:
                valid_smart_cols.append(col)
            else:
                print(f"  - [제거] {col} (결측치 비율: {null_ratio:.2%})")
                
    print(f"  - 완료 (소요시간: {time.time() - t0:.2f}초)")
    print(f"  - 보존 대상 SMART raw 컬럼 수: {len(valid_smart_cols)}개 / 전체 {len(smart_raw_cols)}개")
    
    # 4. 시계열 공백 처리 및 segment 번호 할당
    # (3일 이하: Forward Fill, 3일 초과: 시계열 분리하여 segment 컬럼 값 +=1)
    # record_date - prev_date > 4 이면 3일 초과 공백 (gap > 3)
    print("\n[Step 4] 시계열 공백 분석 및 세그먼트(segment) 분리...")
    t0 = time.time()
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
    
    # 5. 시계열 날짜 확장 (3일 이하 공백 날짜 생성)
    print("  - 세그먼트 단위 연속 날짜 확장 시퀀스 생성...")
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
    # - Forward Fill: LAST_VALUE(IGNORE NULLS)로 이전 유효값 전파
    # - Backward Fill fallback: 세그먼트 첫 행이 null이면(이전 값 없음) 다음 유효값으로 채움
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
    # model: forward fill → backward fill fallback (세그먼트 첫 행 null 대응)
    ffill_selects.append(
        'COALESCE('
        '  LAST_VALUE("model" IGNORE NULLS) OVER (PARTITION BY serial_number, segment ORDER BY record_date),'
        '  FIRST_VALUE("model" IGNORE NULLS) OVER (PARTITION BY serial_number, segment ORDER BY record_date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)'
        ') AS "model"'
    )
    ffill_selects.append('COALESCE(TRY_CAST("failure" AS INTEGER), 0) AS "failure"')
    for col in valid_smart_cols:
        # forward fill: 이전 유효값으로 채우기
        # backward fill fallback: 세그먼트 내 첫 행부터 null이면 다음 유효값으로 채우기
        ffill_selects.append(
            f'COALESCE('
            f'  LAST_VALUE("{col}" IGNORE NULLS) OVER (PARTITION BY serial_number, segment ORDER BY record_date),'
            f'  FIRST_VALUE("{col}" IGNORE NULLS) OVER (PARTITION BY serial_number, segment ORDER BY record_date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)'
            f') AS "{col}"'
        )
        
    ffill_selects_str = ", ".join(ffill_selects)
    
    con.execute(f"""
        CREATE OR REPLACE VIEW final_filled AS
        SELECT 
            serial_number,
            record_date::VARCHAR AS date,
            segment,
            {ffill_selects_str}
        FROM segment_joined
    """)
    print(f"  - 완료 (소요시간: {time.time() - t0:.2f}초)")

    # 6-1. SMART 컬럼 전체가 NULL인 행 제거
    # ffill + bfill 모두 적용 후에도 남아있는 null은 원본 자체가 완전히 비어있는 레코드.
    # (예: 단일 레코드 디바이스인데 그 값 자체가 null인 경우)
    # 이 행들은 복구 불가능하므로 최종 뷰에서 제거한다.
    print("\n[Step 5-1] SMART 컬럼 전체가 NULL인 행 제거...")
    t0 = time.time()
    # 하나라도 NOT NULL이면 유효한 행 → 모든 컬럼이 IS NULL인 행만 제거
    null_filter = " OR ".join([f'"{col}" IS NOT NULL' for col in valid_smart_cols])
    con.execute(f"""
        CREATE OR REPLACE VIEW final_preprocessed AS
        SELECT * FROM final_filled
        WHERE {null_filter}
    """)
    print(f"  - 완료 (소요시간: {time.time() - t0:.2f}초)")
    print(f"  - [참고] 조건: 모든 SMART 컬럼이 NULL인 행 제거 (ffill/bfill 복구 불가 레코드)")

    return con, valid_smart_cols
