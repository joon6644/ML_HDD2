import os
import argparse
import time
import duckdb

def main():
    parser = argparse.ArgumentParser(description="디스크 시계열 데이터 전처리 파이프라인")
    parser.add_argument(
        "--model", 
        type=str, 
        default="ST12000NM0007",
        choices=["ST12000NM0007", "TOSHIBA_20MG08ACA16TA", "TOSHIBA_20MG07ACA14TA"],
        help="전처리를 수행할 하드드라이브 모델명"
    )
    parser.add_argument(
        "--max-memory",
        type=str,
        default="6GB",
        help="DuckDB 사용 메모리 제한 (예: 4GB, 6GB, 8GB)"
    )
    args = parser.parse_args()
    
    model = args.model
    max_memory = args.max_memory
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")
    tmp_dir = os.path.join(project_dir, ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    input_file = os.path.join(data_dir, f"{model}.parquet")
    output_file = os.path.join(data_dir, f"{model}_preprocessed.parquet")
    db_file = os.path.join(tmp_dir, f"preprocess_{model}.db")
    
    if not os.path.exists(input_file):
        print(f"[오류] 입력 데이터 파일이 존재하지 않습니다: {input_file}")
        return

    print("=" * 80)
    print(f" 전처리 시작 - 모델: {model}")
    print(f" - 입력 파일: {input_file}")
    print(f" - 출력 파일: {output_file}")
    print(f" - 메모리 제한: {max_memory}")
    print("=" * 80)
    
    # 기존 DB 파일 제거
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except:
            pass
            
    con = duckdb.connect(database=db_file)
    t_start = time.time()
    
    try:
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
        
        # 6. 결측 데이터 Forward fill 처리
        # 조인 후 LAST_VALUE(IGNORE NULLS)를 사용해 결측 전방 채우기 진행
        print("\n[Step 5] 시계열 데이터 결합 및 Forward Fill 결측치 처리...")
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
        ffill_selects.append('LAST_VALUE("model" IGNORE NULLS) OVER (PARTITION BY serial_number, segment ORDER BY record_date) AS "model"')
        ffill_selects.append('COALESCE(TRY_CAST("failure" AS INTEGER), 0) AS "failure"')
        for col in valid_smart_cols:
            ffill_selects.append(f'LAST_VALUE("{col}" IGNORE NULLS) OVER (PARTITION BY serial_number, segment ORDER BY record_date) AS "{col}"')
            
        ffill_selects_str = ", ".join(ffill_selects)
        
        con.execute(f"""
            CREATE OR REPLACE VIEW final_preprocessed AS
            SELECT 
                serial_number,
                record_date::VARCHAR AS date,
                segment,
                {ffill_selects_str}
            FROM segment_joined
        """)
        print(f"  - 완료 (소요시간: {time.time() - t0:.2f}초)")
        
        # 7. 최종 데이터셋 Parquet로 저장
        print(f"\n[Step 6] 최종 전처리 데이터셋 저장 중: {output_file}")
        t0 = time.time()
        con.execute(f"COPY final_preprocessed TO '{output_file.replace('\\', '/')}' (FORMAT PARQUET)")
        print(f"  - 저장 완료 (소요시간: {time.time() - t0:.2f}초)")
        
        print("=" * 80)
        print(f" 전처리 완료! 총 소요시간: {time.time() - t_start:.2f}초")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[오류] 전처리 중 오류가 발생하였습니다: {e}")
    finally:
        con.close()
        # 임시 DB 파일 제거
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except:
                pass

if __name__ == "__main__":
    main()