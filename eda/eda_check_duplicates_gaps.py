import os
import sys
import argparse
import time
import duckdb
import pandas as pd

def get_selected_models():
    """분석할 모델 리스트를 결정합니다 (명령줄 인자 또는 대화형 메뉴)."""
    all_models = ["TOSHIBA_20MG08ACA16TA", "TOSHIBA_20MG07ACA14TA", "ST12000NM0007"]
    
    parser = argparse.ArgumentParser(description="중복 행 및 시계열 빈 날짜(공백) 건수 계산 스크립트")
    parser.add_argument(
        "--models", 
        nargs="+", 
        choices=all_models,
        help="분석을 수행할 하나 이상의 모델명을 지정합니다. 생략하면 전체 모델이 분석됩니다."
    )
    args, unknown = parser.parse_known_args()
    
    if args.models:
        return args.models
        
    return all_models

def main():
    selected_models = get_selected_models()
    
    project_dir = r"C:\Workspace\projects\26_2_COIN"
    data_dir = os.path.join(project_dir, "data")
    eda_dir = os.path.join(project_dir, "EDA")
    
    con = duckdb.connect(database=":memory:")
    
    print("\n" + "=" * 70)
    print("중복 행 및 시계열 공백 터미널 분석 시작")
    print("=" * 70)
    
    for model in selected_models:
        print("\n" + "-" * 70)
        print(f"모델 분석 진행 중: {model}")
        print("-" * 70)
        
        file_path = os.path.join(data_dir, f"{model}.parquet")
        output_dir = os.path.join(eda_dir, model)
        
        if not os.path.exists(file_path):
            print(f"[오류] 입력 파일이 존재하지 않습니다: {file_path}")
            continue
            
        # 1. 중복 행 및 중복 키 검사
        print("  - [1/3] 중복 데이터 분석 중...")
        t_dup = time.time()
        
        query_total = f"SELECT COUNT(*) FROM read_parquet('{file_path.replace('\\', '/')}')"
        query_dup_keys = f"""
        SELECT COALESCE(SUM(cnt - 1), 0) 
        FROM (
            SELECT serial_number, date, COUNT(*) AS cnt 
            FROM read_parquet('{file_path.replace('\\', '/')}') 
            GROUP BY serial_number, date 
            HAVING COUNT(*) > 1
        )
        """
        
        try:
            total_rows = con.execute(query_total).fetchone()[0]
            duplicate_keys = con.execute(query_dup_keys).fetchone()[0]
            
            # 최적화: 중복 키(serial_number, date)가 0개이면 전체 컬럼 중복도 0개일 수밖에 없음
            if duplicate_keys == 0:
                duplicate_rows = 0
            else:
                # 중복 키가 존재하는 경우, 중복된 키를 가지는 행들로만 대상을 좁혀 완전 중복 행 수를 계산 (초고속 연산)
                query_dup_rows_distinct = f"""
                WITH dup_keys AS (
                    SELECT serial_number, date 
                    FROM read_parquet('{file_path.replace('\\', '/')}')
                    GROUP BY serial_number, date
                    HAVING COUNT(*) > 1
                ),
                dup_rows AS (
                    SELECT p.* 
                    FROM read_parquet('{file_path.replace('\\', '/')}') p
                    JOIN dup_keys dk ON p.serial_number = dk.serial_number AND p.date = dk.date
                )
                SELECT 
                    (SELECT COUNT(*) FROM dup_rows) - (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM dup_rows)) AS dup_count
                """
                duplicate_rows = con.execute(query_dup_rows_distinct).fetchone()[0]
            
            print(f"    * 중복 분석 완료 (소요 시간: {time.time() - t_dup:.2f}초)")
        except Exception as e:
            print(f"    [오류] 중복 분석 실패: {e}")
            continue
            
        # 2. 컬럼 필터링 (결측치 10% 미만의 SMART raw 컬럼 및 상수 컬럼 제거)
        print("  - [2/3] 유효 SMART raw 컬럼 선정 중...")
        csv_report_path = os.path.join(output_dir, "missing_values_report.csv")
        selected_cols = []
        
        if os.path.exists(csv_report_path):
            df_missing = pd.read_csv(csv_report_path)
        else:
            print("    [안내] 기존 결측치 보고서가 없어 실시간으로 결측율을 연산합니다...")
            try:
                cols_info = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{file_path.replace('\\', '/')}')").fetchall()
                cols = [c[0] for c in cols_info]
                select_clause = ", ".join([f'COUNT("{col}") AS "cnt_{col}"' for col in cols])
                query = f"SELECT COUNT(*) AS total_rows, {select_clause} FROM read_parquet('{file_path.replace('\\', '/')}')"
                res = con.execute(query).fetchone()
                total_rows_val = res[0]
                
                missing_data = []
                for i, col in enumerate(cols):
                    non_null_count = res[i + 1]
                    missing_count = total_rows_val - non_null_count
                    missing_ratio = (missing_count / total_rows_val) if total_rows_val > 0 else 0.0
                    missing_data.append({"column_name": col, "missing_ratio": missing_ratio})
                df_missing = pd.DataFrame(missing_data)
            except Exception as e:
                print(f"    [오류] 결측율 계산 실패: {e}")
                continue
                
        # 조건: 'smart_' 포함, '_raw'로 끝남, 결측치 10% 미만
        filtered_df = df_missing[
            df_missing["column_name"].str.contains("smart_") & 
            df_missing["column_name"].str.endswith("_raw") & 
            (df_missing["missing_ratio"] < 0.1)
        ]
        selected_cols = filtered_df["column_name"].tolist()
        
        # 상수 컬럼 제외
        non_const_cols = []
        check_items = []
        for col in selected_cols:
            c_cast = f'TRY_CAST("{col}" AS DOUBLE)'
            check_items.append(f'(MIN({c_cast}) = MAX({c_cast}) OR COUNT({c_cast}) <= 1) AS "is_const_{col}"')
            
        query_const = f"SELECT {', '.join(check_items)} FROM read_parquet('{file_path.replace('\\', '/')}')"
        try:
            res_const = con.execute(query_const).df().to_dict(orient="records")[0]
            for col in selected_cols:
                if not res_const[f"is_const_{col}"]:
                    non_const_cols.append(col)
            selected_cols = non_const_cols
        except Exception as e:
            print(f"    [경고] 상수 컬럼 필터링 실패 (전체 컬럼 대상 진행): {e}")
            
        print(f"    * 최종 선정된 유효 SMART raw 컬럼 수: {len(selected_cols)}개")
        
        # 3. 시계열 공백(날짜 빈 곳) 집계
        print("  - [3/3] 시계열 빈 날짜(공백) 분석 중...")
        t_gap = time.time()
        
        or_conditions = " OR ".join([f'TRY_CAST("{col}" AS DOUBLE) IS NOT NULL' for col in selected_cols])
        
        gap_summary_query = f"""
        WITH valid_records AS (
            SELECT 
                serial_number,
                TRY_CAST(date AS DATE) AS record_date
            FROM read_parquet('{file_path.replace('\\', '/')}')
            WHERE {or_conditions}
        ),
        sorted_records AS (
            SELECT 
                serial_number,
                record_date,
                LEAD(record_date) OVER (PARTITION BY serial_number ORDER BY record_date) AS next_record_date
            FROM valid_records
        ),
        gaps AS (
            SELECT 
                next_record_date - record_date - 1 AS gap_days
            FROM sorted_records
            WHERE next_record_date - record_date > 1
        )
        SELECT 
            COUNT(*) AS gap_segments,
            COALESCE(SUM(gap_days), 0) AS total_gap_days
        FROM gaps
        """
        
        try:
            res_gap = con.execute(gap_summary_query).fetchone()
            gap_segments = res_gap[0]
            total_gap_days = res_gap[1]
            print(f"    * 시계열 분석 완료 (소요 시간: {time.time() - t_gap:.2f}초)")
        except Exception as e:
            print(f"    [오류] 시계열 공백 연산 실패: {e}")
            continue
            
        # 4. 분석 결과 출력
        print("\n  >>> 분석 결과 요약 <<<")
        print(f"    - 전체 행 수: {total_rows:,}행")
        print(f"    - 완전 중복 행 수: {duplicate_rows:,}행 (비율: {(duplicate_rows / total_rows) * 100:.4f}%)")
        print(f"    - 중복 키(serial_number, date) 데이터 수: {duplicate_keys:,}행")
        print(f"    - 시계열 공백 구간(세그먼트) 수: {gap_segments:,}건")
        print(f"    - 시계열 빈 날짜(공백 일수) 총합: {total_gap_days:,}일")
        
    con.close()
    print("\n" + "=" * 70)
    print("모든 모델 분석 및 터미널 출력이 완료되었습니다!")
    print("=" * 70)

if __name__ == "__main__":
    main()
