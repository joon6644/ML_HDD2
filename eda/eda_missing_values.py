import os
import sys
import argparse
import time
import duckdb
import pandas as pd

def get_selected_models():
    """명령줄 인자(CLI arguments) 또는 대화형 입력을 기반으로 실행할 모델 리스트를 결정합니다."""
    models = ["TOSHIBA_20MG08ACA16TA", "TOSHIBA_20MG07ACA14TA", "ST12000NM0007"]
    
    # 1. 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="컬럼별 결측치 비율을 계산하고 CSV 보고서로 저장합니다.")
    parser.add_argument(
        "--models", 
        nargs="+", 
        choices=models,
        help="분석을 수행할 하나 이상의 모델명을 지정합니다. 생략하면 대화형 프롬프트가 실행되거나 전체 모델이 분석됩니다."
    )
    args, unknown = parser.parse_known_args()
    
    if args.models:
        return args.models
        
    # 2. 터미널이 대화형 입력을 지원하는 경우 선택 메뉴 표시
    if sys.stdin.isatty():
        print("=" * 60)
        print("   결측치 비율 분석 - 모델 선택 메뉴")
        print("=" * 60)
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        print("  4. 전체 모델 실행")
        print("=" * 60)
        try:
            choice = input("분석할 모델을 선택하세요 (예: 1,3 또는 4) [기본값: 4]: ").strip()
            if not choice or choice == "4":
                return models
            
            selected = []
            for part in choice.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(models):
                        selected.append(models[idx])
            
            if selected:
                return selected
        except Exception:
            pass
            
    # 3. 비대화형 환경이거나 입력에 실패했을 경우 전체 모델을 기본값으로 반환
    return models

def main():
    selected_models = get_selected_models()
    print(f"\n분석을 진행할 모델: {selected_models}")
    
    project_dir = r"C:\Workspace\projects\26_2_COIN"
    data_dir = os.path.join(project_dir, "data")
    eda_dir = os.path.join(project_dir, "EDA")
    
    con = duckdb.connect(database=":memory:")
    
    for model in selected_models:
        print("\n" + "=" * 70)
        print(f"모델 처리 중: {model}")
        print("=" * 70)
        
        file_path = os.path.join(data_dir, f"{model}.parquet")
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
