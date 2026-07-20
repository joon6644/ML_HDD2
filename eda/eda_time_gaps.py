import os
import sys
import argparse
import time
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_selected_models():
    """분석할 모델 리스트를 결정합니다 (명령줄 인자 또는 대화형 메뉴)."""
    all_models = ["TOSHIBA_20MG08ACA16TA", "TOSHIBA_20MG07ACA14TA", "ST12000NM0007"]
    
    parser = argparse.ArgumentParser(description="모델별 시계열 공백 개체 수 및 기간 분포 분석 스크립트")
    parser.add_argument(
        "--models", 
        nargs="+", 
        choices=all_models,
        help="분석을 수행할 하나 이상의 모델명을 지정합니다. 생략하면 대화형 프롬프트가 실행되거나 전체 모델이 분석됩니다."
    )
    args, unknown = parser.parse_known_args()
    
    if args.models:
        return args.models
        
    if sys.stdin.isatty():
        print("=" * 60)
        print("   시계열 공백 분석 - 모델 선택 메뉴")
        print("=" * 60)
        for i, model in enumerate(all_models, 1):
            print(f"  {i}. {model}")
        print("  4. 전체 모델 실행")
        print("=" * 60)
        try:
            choice = input("분석할 모델을 선택하세요 (예: 1,3 또는 4) [기본값: 4]: ").strip()
            if not choice or choice == "4":
                return all_models
            
            selected = []
            for part in choice.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(all_models):
                        selected.append(all_models[idx])
            if selected:
                return selected
        except Exception:
            pass
            
    return all_models

def main():
    selected_models = get_selected_models()
    print(f"\n분석을 진행할 모델 리스트: {selected_models}")
    
    project_dir = r"C:\Workspace\projects\26_2_COIN"
    data_dir = os.path.join(project_dir, "data")
    eda_dir = os.path.join(project_dir, "EDA")
    
    con = duckdb.connect(database=":memory:")
    
    # 한글 폰트 설정 (Windows 기본 맑은 고딕 사용)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Malgun Gothic", "Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    
    for model in selected_models:
        print("\n" + "=" * 70)
        print(f"모델 분석 시작: {model}")
        print("=" * 70)
        
        file_path = os.path.join(data_dir, f"{model}.parquet")
        output_dir = os.path.join(eda_dir, model)
        
        if not os.path.exists(file_path):
            print(f"[오류] 입력 파일이 존재하지 않습니다: {file_path}")
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 컬럼 필터링 (결측치 10% 미만의 SMART raw 컬럼 및 상수 컬럼 제거)
        csv_report_path = os.path.join(output_dir, "missing_values_report.csv")
        selected_cols = []
        
        if os.path.exists(csv_report_path):
            print("  - 기존 결측치 보고서 로드 중...")
            df_missing = pd.read_csv(csv_report_path)
        else:
            print("  - 결측치 보고서가 없어 결측율을 즉시 연산합니다...")
            try:
                cols_info = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{file_path.replace('\\', '/')}')").fetchall()
                cols = [c[0] for c in cols_info]
                select_clause = ", ".join([f'COUNT("{col}") AS "cnt_{col}"' for col in cols])
                query = f"SELECT COUNT(*) AS total_rows, {select_clause} FROM read_parquet('{file_path.replace('\\', '/')}')"
                res = con.execute(query).fetchone()
                total_rows = res[0]
                
                missing_data = []
                for i, col in enumerate(cols):
                    non_null_count = res[i + 1]
                    missing_count = total_rows - non_null_count
                    missing_ratio = (missing_count / total_rows) if total_rows > 0 else 0.0
                    missing_data.append({"column_name": col, "missing_ratio": missing_ratio})
                df_missing = pd.DataFrame(missing_data)
            except Exception as e:
                print(f"  [오류] 결측율 계산 실패: {e}")
                continue
                
        # 조건 필터: 'smart_' 포함, '_raw'로 끝남, 결측치 10% 미만
        filtered_df = df_missing[
            df_missing["column_name"].str.contains("smart_") & 
            df_missing["column_name"].str.endswith("_raw") & 
            (df_missing["missing_ratio"] < 0.1)
        ]
        selected_cols = filtered_df["column_name"].tolist()
        
        if not selected_cols:
            print("  [경고] 분석 대상 SMART raw 컬럼을 찾지 못했습니다.")
            continue
            
        # 상수 컬럼 제외
        print("  - 상수 컬럼 탐색 및 필터링 중...")
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
            print(f"  [경고] 상수 컬럼 탐색 중 오류 발생 (전체 컬럼 대상 진행): {e}")
            
        if not selected_cols:
            print("  [경고] 상수 컬럼을 제외한 후 분석할 SMART raw 컬럼이 존재하지 않습니다.")
            continue
            
        print(f"  - 최종 선정된 SMART raw 컬럼 수: {len(selected_cols)}개")
        
        # 2. DuckDB를 통한 시계열 공백 분석 쿼리 실행
        print("  - 시계열 공백 및 개체별 공백 분포 연산 중...")
        t0 = time.time()
        
        # 필터링 조건: 선정된 컬럼 중 최소한 하나는 유효한(Non-null) 값을 가져야 함
        or_conditions = " OR ".join([f'TRY_CAST("{col}" AS DOUBLE) IS NOT NULL' for col in selected_cols])
        
        # 전체 공백 리스트 쿼리 (Lead 윈도우 함수 사용)
        gap_list_query = f"""
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
                serial_number,
                record_date,
                next_record_date,
                next_record_date - record_date - 1 AS gap_days
            FROM sorted_records
            WHERE next_record_date - record_date > 1
        )
        SELECT * FROM gaps
        """
        
        # 개체(serial_number)별 누적 공백 횟수 분포 쿼리 (공백이 없는 0인 개체 포함)
        device_gaps_query = f"""
        WITH unique_devices AS (
            SELECT DISTINCT serial_number FROM read_parquet('{file_path.replace('\\', '/')}')
        ),
        gaps AS (
            {gap_list_query}
        ),
        device_gaps AS (
            SELECT 
                u.serial_number,
                COALESCE(g.num_gaps, 0) AS num_gaps
            FROM unique_devices u
            LEFT JOIN (
                SELECT serial_number, COUNT(*) AS num_gaps
                FROM gaps
                GROUP BY serial_number
            ) g ON u.serial_number = g.serial_number
        )
        SELECT num_gaps, COUNT(*) AS count_devices
        FROM device_gaps
        GROUP BY num_gaps
        ORDER BY num_gaps
        """
        
        try:
            df_gaps = con.execute(gap_list_query).df()
            df_device_gaps = con.execute(device_gaps_query).df()
            print(f"  - 쿼리 완료 (소요 시간: {time.time() - t0:.2f}초)")
            print(f"  - 총 발견된 공백 구간 수: {len(df_gaps):,}개")
        except Exception as e:
            print(f"  [오류] 공백 데이터 쿼리 실패: {e}")
            continue
            
        # 3. 하나의 이미지로 병합 시각화 작성 (1x2 Subplots)
        print("  - 시각화 이미지 렌더링 중...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 3a. 왼쪽 그래프: 개체별 공백 횟수 분포 (Bar Plot)
        total_devices = df_device_gaps["count_devices"].sum()
        devices_with_gaps = df_device_gaps[df_device_gaps["num_gaps"] > 0]["count_devices"].sum()
        gap_device_ratio = (devices_with_gaps / total_devices) * 100 if total_devices > 0 else 0.0
        
        sns.barplot(
            x="num_gaps", 
            y="count_devices", 
            data=df_device_gaps, 
            ax=axes[0], 
            color="#3498db", 
            edgecolor="#2c3e50",
            alpha=0.85
        )
        axes[0].set_title("개체별 시계열 공백 발생 횟수 분포", fontsize=14, fontweight="bold", pad=15)
        axes[0].set_xlabel("개체당 공백 발생 횟수 (회)", fontsize=11)
        axes[0].set_ylabel("개체 수 (HDD 수)", fontsize=11)
        axes[0].tick_params(labelsize=10)
        
        # 개체 요약 통계 텍스트 박스
        summary_text_left = (
            f"• 전체 개체 수: {total_devices:,}개\n"
            f"• 공백 발생 개체 수: {devices_with_gaps:,}개\n"
            f"• 공백 발생 개체 비율: {gap_device_ratio:.2f}%"
        )
        props_left = dict(boxstyle="round,pad=0.5", facecolor="#eaf2f8", edgecolor="#a9cce3", alpha=0.9)
        axes[0].text(
            0.95, 0.95, summary_text_left, 
            transform=axes[0].transAxes, 
            ha="right", va="top", 
            fontsize=10.5, bbox=props_left
        )
        
        # 3b. 오른쪽 그래프: 시계열 공백의 기간 분포 (Histogram with Log Y-scale)
        if len(df_gaps) > 0:
            max_gap = df_gaps["gap_days"].max()
            mean_gap = df_gaps["gap_days"].mean()
            median_gap = df_gaps["gap_days"].median()
            
            # 히스토그램 그리기 (Y축 로그 스케일 적용하여 롱테일 시각화)
            axes[1].hist(
                df_gaps["gap_days"], 
                bins=min(50, int(max_gap)), 
                color="#e74c3c", 
                edgecolor="#2c3e50", 
                alpha=0.85, 
                log=True
            )
            axes[1].set_title("시계열 공백 기간(일수) 분포 (Y축 Log 스케일)", fontsize=14, fontweight="bold", pad=15)
            axes[1].set_xlabel("공백 기간 (일)", fontsize=11)
            axes[1].set_ylabel("공백 발생 횟수 (건)", fontsize=11)
            axes[1].tick_params(labelsize=10)
            
            # 공백 기간 요약 통계 텍스트 박스
            summary_text_right = (
                f"• 총 공백 구간 건수: {len(df_gaps):,}건\n"
                f"• 평균 공백 기간: {mean_gap:.2f}일\n"
                f"• 중앙값 공백 기간: {median_gap:.1f}일\n"
                f"• 최대 공백 기간: {max_gap:,}일"
            )
            props_right = dict(boxstyle="round,pad=0.5", facecolor="#fdebd0", edgecolor="#f5cba7", alpha=0.9)
            axes[1].text(
                0.95, 0.95, summary_text_right, 
                transform=axes[1].transAxes, 
                ha="right", va="top", 
                fontsize=10.5, bbox=props_right
            )
        else:
            axes[1].text(0.5, 0.5, "발견된 공백 구간이 없습니다.", ha="center", va="center", fontsize=14)
            axes[1].set_title("시계열 공백 기간(일수) 분포", fontsize=14, fontweight="bold", pad=15)
            
        plt.suptitle(f"Time Series Gap Analysis Dashboard - {model}", fontsize=18, fontweight="bold", color="#2c3e50", y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        plot_path = os.path.join(output_dir, "time_gaps_plots.png")
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  - 시각화 이미지 저장 완료: {plot_path}")
        
    con.close()
    print("\n" + "=" * 70)
    print("모든 모델에 대한 시계열 공백 분석 및 시각화 이미지 생성이 완료되었습니다!")
    print("=" * 70)

if __name__ == "__main__":
    main()
