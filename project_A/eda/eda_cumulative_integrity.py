import os
import time
import argparse
import pandas as pd
import duckdb

import sys

def get_selected_model():
    """사용자가 분석할 단 하나의 모델 데이터셋을 지정하게 합니다."""
    all_models = ["HGST_20HUH721212ALN604", "TOSHIBA_20MG07ACA14TA", "ST12000NM0007"]
    
    parser = argparse.ArgumentParser(description="SMART raw 속성 누적 무결성 분석 스크립트")
    parser.add_argument("--model", choices=all_models, help="분석할 모델명 지정")
    parser.add_argument("--file", type=str, help="특정 Parquet 파일 경로 지정")
    args, _ = parser.parse_known_args()
    
    if args.file:
        return None, args.file
    if args.model:
        return args.model, None
        
    if sys.stdin.isatty():
        print("=" * 60)
        print("   SMART Raw 속성 누적 무결성 분석 - 모델 선택")
        print("=" * 60)
        for i, model in enumerate(all_models, 1):
            print(f"  {i}. {model}")
        print("=" * 60)
        try:
            choice = input("분석할 모델의 번호를 선택하세요 (1~3): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(all_models):
                    return all_models[idx], None
        except Exception:
            pass
            
    return all_models[0], None

def main():
    model, custom_file = get_selected_model()
    
    project_dir = r"C:\Workspace\projects\26_2_COIN"
    data_dir = os.path.join(project_dir, "data")
    raw_dir = os.path.join(data_dir, "raw")
    eda_dir = os.path.join(project_dir, "EDA")
    
    if custom_file:
        file_path = custom_file
        filename = os.path.basename(file_path)
        model = os.path.splitext(filename)[0]
    else:
        file_path = os.path.join(raw_dir, f"{model}.parquet")
        
    output_dir = os.path.join(eda_dir, model)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(file_path):
        print(f"[오류] 데이터 파일이 존재하지 않습니다: {file_path}")
        return

    print("\n" + "=" * 70)
    print(f"SMART Raw 속성 누적 무결성 분석 시작: {model}")
    print("=" * 70)

    con = duckdb.connect(database=":memory:")
    
    # 1. 결측치 보고서 참조 또는 실시간 결측률 연산으로 유효 SMART raw 컬럼 탐색
    print(f"  - 스키마 분석 및 유효 SMART raw 컬럼 탐색 중: {file_path}")
    csv_report_path = os.path.join(output_dir, "missing_values_report.csv")
    
    if os.path.exists(csv_report_path):
        df_missing = pd.read_csv(csv_report_path)
    else:
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
            print(f"  [오류] 결측률 연산 실패: {e}")
            return

    # 조건: 'smart_' 포함, '_raw'로 끝남, 결측치 10% 미만
    filtered_df = df_missing[
        df_missing["column_name"].str.contains("smart_") & 
        df_missing["column_name"].str.endswith("_raw") & 
        (df_missing["missing_ratio"] < 0.1)
    ]
    candidate_cols = filtered_df["column_name"].tolist()

    # 상수 컬럼 제외
    non_const_cols = []
    check_items = []
    for col in candidate_cols:
        c_cast = f'TRY_CAST("{col}" AS DOUBLE)'
        check_items.append(f'(MIN({c_cast}) = MAX({c_cast}) OR COUNT({c_cast}) <= 1) AS "is_const_{col}"')
        
    query_const = f"SELECT {', '.join(check_items)} FROM read_parquet('{file_path.replace('\\', '/')}')"
    try:
        res_const = con.execute(query_const).df().to_dict(orient="records")[0]
        for col in candidate_cols:
            if not res_const[f"is_const_{col}"]:
                non_const_cols.append(col)
        candidate_cols = non_const_cols
    except Exception as e:
        print(f"  [경고] 상수 컬럼 탐색 필터링 실패 (전체 후보 컬럼 대상 진행): {e}")

    # SMART 속성 번호 순서대로 정렬 (smart_1_raw, smart_4_raw, ...)
    def sort_key(col_name):
        try:
            return int(col_name.split("_")[1])
        except Exception:
            return 999

    run_cols = sorted(candidate_cols, key=sort_key)
            
    print(f"\n[자동 탐색] 범용 분석 대상 SMART raw 컬럼 ({len(run_cols)}개): {run_cols}")
    print("=" * 110)
    print(f"{'Column':<16} | {'Total Devices':<13} | {'Total Records':<13} | {'Dec. Devices':<12} | {'Dec. Events':<11} | {'Avg Decrease':<13} | {'Max Decrease':<13} | {'Strictly Cum.':<13}")
    print("-" * 110)
    
    results = []
    
    for idx, col in enumerate(run_cols, 1):
        t0 = time.time()
        print(f"  [{idx}/{len(run_cols)}] {col} 누적 무결성 연산 중...", end="", flush=True)
        
        # 각 디바이스(serial_number)의 날짜(date) 순 흐름에서 이전 값 대비 감소했는지 검증하는 쿼리
        # 데이터가 이미 정렬되어 있으므로 윈도우 함수가 추가 정렬 없이 효율적으로 작동합니다.
        query = f"""
        WITH sorted_data AS (
            SELECT 
                serial_number,
                TRY_CAST(date AS DATE) AS record_date,
                TRY_CAST("{col}" AS DOUBLE) AS val
            FROM read_parquet('{file_path.replace('\\', '/')}')
            WHERE TRY_CAST("{col}" AS DOUBLE) IS NOT NULL
        ),
        prev_values AS (
            SELECT 
                serial_number,
                record_date,
                val,
                LAG(val) OVER (PARTITION BY serial_number ORDER BY record_date) AS prev_val
            FROM sorted_data
        ),
        stats AS (
            SELECT
                COUNT(DISTINCT serial_number) AS total_devices,
                COUNT(*) AS total_records,
                SUM(CASE WHEN prev_val IS NOT NULL THEN 1 ELSE 0 END) AS compared_transitions,
                SUM(CASE WHEN prev_val IS NOT NULL AND val < prev_val THEN 1 ELSE 0 END) AS decrease_events,
                COUNT(DISTINCT CASE WHEN prev_val IS NOT NULL AND val < prev_val THEN serial_number END) AS decrease_devices
            FROM prev_values
        ),
        decreases AS (
            SELECT 
                (prev_val - val) AS decrease_amount
            FROM prev_values
            WHERE prev_val IS NOT NULL AND val < prev_val
        )
        SELECT 
            s.total_devices,
            s.total_records,
            s.compared_transitions,
            s.decrease_events,
            s.decrease_devices,
            COALESCE((SELECT AVG(decrease_amount) FROM decreases), 0) AS avg_decrease,
            COALESCE((SELECT MAX(decrease_amount) FROM decreases), 0) AS max_decrease
        FROM stats s
        """
        
        try:
            row = con.execute(query).fetchone()
            total_devices, total_records, compared_transitions, decrease_events, decrease_devices, avg_decrease, max_decrease = row
            
            is_strictly_cum = "True" if decrease_events == 0 else "False"
            
            print(f"\r{col:<16} | {total_devices:<13,} | {total_records:<13,} | {decrease_devices:<12,} | {decrease_events:<11,} | {avg_decrease:<13.2f} | {max_decrease:<13,.1f} | {is_strictly_cum:<13}", flush=True)
            
            results.append({
                "column": col,
                "total_devices": total_devices,
                "total_records": total_records,
                "compared_transitions": compared_transitions,
                "decrease_devices": decrease_devices,
                "decrease_events": decrease_events,
                "avg_decrease": avg_decrease,
                "max_decrease": max_decrease,
                "is_strictly_cumulative": is_strictly_cum,
                "elapsed_seconds": round(time.time() - t0, 2)
            })
        except Exception as e:
            print(f"{col:<16} | 오류 발생: {e}")
            
    print("=" * 110)
    
    # CSV 결과 저장
    df_res = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "cumulative_integrity_report.csv")
    df_res.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[완료] 무결성 검증 리포트 저장 완료: {csv_path}")
    
    con.close()

if __name__ == "__main__":
    main()
