import os
import sys
import argparse
import time
import duckdb
import pandas as pd

def get_selected_model():
    """사용자가 분석할 단 하나의 모델 데이터셋을 지정하게 합니다."""
    all_models = ["HGST_20HUH721212ALN604", "TOSHIBA_20MG07ACA14TA", "ST12000NM0007"]
    
    parser = argparse.ArgumentParser(description="컬럼별 결측치 비율을 계산하고 CSV 보고서로 저장합니다.")
    parser.add_argument("--model", choices=all_models, help="분석할 모델명 지정")
    parser.add_argument("--models", nargs="+", choices=all_models, help="분석할 모델 리스트")
    args, _ = parser.parse_known_args()
    
    if args.model:
        return args.model
    if args.models and len(args.models) > 0:
        return args.models[0]
        
    if sys.stdin.isatty():
        print("=" * 60)
        print("   컬럼별 결측치 비율 분석 - 모델 선택")
        print("=" * 60)
        for i, model in enumerate(all_models, 1):
            print(f"  {i}. {model}")
        print("=" * 60)
        try:
            choice = input("분석할 모델의 번호를 선택하세요 (1~3): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(all_models):
                    return all_models[idx]
        except Exception:
            pass
            
    return all_models[0]

def main():
    model = get_selected_model()
    selected_models = [model]
    
    project_dir = r"C:\Workspace\projects\26_2_COIN"
    data_dir = os.path.join(project_dir, "data")
    raw_dir = os.path.join(data_dir, "raw")
    eda_dir = os.path.join(project_dir, "EDA")
    
    con = duckdb.connect(database=":memory:")
    
    for model in selected_models:
        print("\n" + "=" * 70)
        print(f"컬럼별 결측치 비율 분석 시작: {model}")
        print("=" * 70)
        
        file_path = os.path.join(raw_dir, f"{model}.parquet")
        output_dir = os.path.join(eda_dir, model)
        
        if not os.path.exists(file_path):
            print(f"[오류] 입력 파일이 존재하지 않습니다: {file_path}")
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. DuckDB를 사용하여 스키마를 확인하고 결측치 계산 쿼리 생성
        print("  - 스키마 분석 중...")
        try:
            cols_info = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{file_path.replace('\\', '/')}')").fetchall()
            cols = [c[0] for c in cols_info]
        except Exception as e:
            print(f"  [오류] 스키마 읽기 실패: {e}")
            continue
            
        total_cols = len(cols)
        print(f"  - 총 {total_cols}개 컬럼 발견. 결측치 비율 계산 중...")
        
        select_clause = ", ".join([f'COUNT("{col}") AS "cnt_{col}"' for col in cols])
        query = f"SELECT COUNT(*) AS total_rows, {select_clause} FROM read_parquet('{file_path.replace('\\', '/')}')"
        
        t0 = time.time()
        try:
            res = con.execute(query).fetchone()
            total_rows = res[0]
            print(f"  - {total_rows:,}개 행 읽기 완료 (소요 시간: {time.time() - t0:.2f}초).")
        except Exception as e:
            print(f"  [오류] 쿼리 실행 실패: {e}")
            continue
            
        # 2. 컬럼별 결측치 비율 계산
        missing_data = []
        for i, col in enumerate(cols):
            non_null_count = res[i + 1]
            missing_count = total_rows - non_null_count
            missing_ratio = (missing_count / total_rows) if total_rows > 0 else 0.0
            missing_data.append({
                "column_name": col,
                "non_null_count": non_null_count,
                "missing_count": missing_count,
                "missing_ratio": missing_ratio,
                "missing_ratio_pct": missing_ratio * 100
            })
            
        df = pd.DataFrame(missing_data)
        
        # 결측비율 기준 내림차순 정렬
        df = df.sort_values(by="missing_count", ascending=False).reset_index(drop=True)
        
        # 3. 상세 CSV 보고서 저장
        csv_path = os.path.join(output_dir, "missing_values_report.csv")
        df.to_csv(csv_path, index=False)
        print(f"  - CSV 보고서 저장 완료: {csv_path}")
        
    con.close()
    print("\n" + "=" * 70)
    print("요청한 모든 결측치 비율 분석 작업이 성공적으로 완료되었습니다!")
    print("=" * 70)

if __name__ == "__main__":
    main()
