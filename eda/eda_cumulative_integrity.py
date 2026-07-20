import os
import time
import argparse
import pandas as pd
import duckdb

def main():
    parser = argparse.ArgumentParser(description="SMART raw 속성 누적 무결성 분석 스크립트")
    parser.add_argument(
        "--file",
        type=str,
        help="분석할 특정 Parquet 파일 경로 지정. 지정할 경우 해당 파일의 이름을 기준으로 EDA 결과 폴더가 생성됩니다."
    )
    args = parser.parse_args()
    
    project_dir = r"C:\Workspace\projects\26_2_COIN"
    data_dir = os.path.join(project_dir, "data")
    eda_dir = os.path.join(project_dir, "eda")
    
    if args.file:
        file_path = args.file
        filename = os.path.basename(file_path)
        model = os.path.splitext(filename)[0]
    else:
        model = "ST12000NM0007"
        file_path = os.path.join(data_dir, f"{model}.parquet")
        
    output_dir = os.path.join(eda_dir, model)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(file_path):
        print(f"[오류] 데이터 파일이 존재하지 않습니다: {file_path}")
        return

    # 사용자 요청 SMART 컬럼 리스트
    target_smart_nums = [1, 4, 5, 7, 9, 12, 187, 188, 192, 193, 195, 199, 240, 241, 242]
    target_cols = [f"smart_{num}_raw" for num in target_smart_nums]
    
    con = duckdb.connect(database=":memory:")
    
    # 1. Parquet 스키마 조회하여 실제 존재하는 컬럼 필터링
    print(f"스키마 확인 중: {file_path}")
    schema_df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{file_path.replace('\\', '/')}') LIMIT 1").df()
    available_cols = set(schema_df["column_name"].tolist())
    
    run_cols = []
    for col in target_cols:
        if col in available_cols:
            run_cols.append(col)
        else:
            print(f"  [정보] {col} 컬럼이 데이터셋 스키마에 존재하지 않아 생략합니다.")
            
    print(f"\n분석 대상 컬럼 ({len(run_cols)}개): {run_cols}")
    print("=" * 110)
    print(f"{'Column':<16} | {'Total Devices':<13} | {'Total Records':<13} | {'Dec. Devices':<12} | {'Dec. Events':<11} | {'Avg Decrease':<13} | {'Max Decrease':<13} | {'Strictly Cum.':<13}")
    print("-" * 110)
    
    results = []
    
    for col in run_cols:
        t0 = time.time()
        
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
            
            print(f"{col:<16} | {total_devices:<13,} | {total_records:<13,} | {decrease_devices:<12,} | {decrease_events:<11,} | {avg_decrease:<13.2f} | {max_decrease:<13,.1f} | {is_strictly_cum:<13}")
            
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
