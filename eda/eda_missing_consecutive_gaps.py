import os
import sys
import argparse
import time
import math
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_selected_models():
    """분석할 모델 리스트를 결정합니다 (명령줄 인자 또는 대화형 메뉴)."""
    all_models = ["ST12000NM0007"]
    
    parser = argparse.ArgumentParser(description="모델별 컬럼별 결측치 연속 공백 일수 분포 시각화 스크립트")
    parser.add_argument(
        "--models", 
        nargs="+", 
        choices=["TOSHIBA_20MG08ACA16TA", "TOSHIBA_20MG07ACA14TA", "ST12000NM0007"],
        help="분석을 수행할 하나 이상의 모델명을 지정합니다. 생략하면 전체 모델이 분석됩니다."
    )
    args, unknown = parser.parse_known_args()
    
    if args.models:
        return args.models
        
    return all_models

def main():
    selected_models = get_selected_models()
    print(f"\n분석을 진행할 모델 리스트: {selected_models}")
    
    project_dir = r"C:\Workspace\projects\26_2_COIN"
    data_dir = os.path.join(project_dir, "data")
    eda_dir = os.path.join(project_dir, "EDA")
    
    # 한글 폰트 설정 (Windows 맑은 고딕 사용)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Malgun Gothic", "Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    
    for model in selected_models:
        print("\n" + "=" * 70)
        print(f"모델 분석 시작: {model}")
        print("=" * 70)
        
        con = duckdb.connect(database=":memory:")
        
        file_path = os.path.join(data_dir, f"{model}.parquet")
        output_dir = os.path.join(eda_dir, model)
        
        if not os.path.exists(file_path):
            print(f"[오류] 입력 파일이 존재하지 않습니다: {file_path}")
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 컬럼 필터링 (결측치 10% 미만의 SMART raw 컬럼 및 상수 컬럼 제외)
        csv_report_path = os.path.join(output_dir, "missing_values_report.csv")
        selected_cols = []
        
        if os.path.exists(csv_report_path):
            print("  - 기존 결측치 보고서 로드 중...")
            df_missing = pd.read_csv(csv_report_path)
        else:
            print("  - 결측치 보고서가 없어 실시간으로 결측율을 연산합니다...")
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
                
        # 조건: 'smart_' 포함, '_raw'로 끝남, 결측치 10% 미만
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
        
        # 2. 각 컬럼별 절대적인 결측치 수 구하기 (초고속 배치 쿼리)
        print("  - 컬럼별 절대적인 결측치(Null) 개수 연산 중...")
        cnt_selects = ", ".join([f'COUNT(TRY_CAST("{col}" AS DOUBLE)) AS "cnt_{col}"' for col in selected_cols])
        query_missing_counts = f"SELECT COUNT(*) AS total_rows, {cnt_selects} FROM read_parquet('{file_path.replace('\\', '/')}')"
        
        try:
            res_counts = con.execute(query_missing_counts).fetchone()
            total_rows = res_counts[0]
            col_missing_counts = {}
            for i, col in enumerate(selected_cols):
                non_null_cnt = res_counts[i + 1]
                missing_cnt = total_rows - non_null_cnt
                col_missing_counts[col] = missing_cnt
        except Exception as e:
            print(f"  [오류] 결측치 개수 연산 실패: {e}")
            continue
            
        # 3. 결측치가 존재하는 컬럼에 대해 연속 결측치(공백) 길이 분포 연산
        print("  - 컬럼별 결측치 연속 공백 일수 분포 연산 중...")
        consecutive_gaps = {}
        
        for col in selected_cols:
            missing_cnt = col_missing_counts[col]
            if missing_cnt == 0:
                consecutive_gaps[col] = []
                continue
                
            # Gaps and Islands 윈도우 함수 쿼리 (이미 정렬되어 있으므로 parquet 파일에서 직접 연산)
            query_gap_dist = f"""
            WITH marked AS (
                SELECT 
                    serial_number,
                    date,
                    CASE WHEN TRY_CAST("{col}" AS DOUBLE) IS NULL THEN 1 ELSE 0 END AS is_null,
                    SUM(CASE WHEN TRY_CAST("{col}" AS DOUBLE) IS NOT NULL THEN 1 ELSE 0 END) 
                        OVER (PARTITION BY serial_number ORDER BY date) AS grp
                FROM read_parquet('{file_path.replace('\\', '/')}')
            )
            SELECT 
                consecutive_nulls,
                COUNT(*) AS occurrence_count
            FROM (
                SELECT 
                    serial_number,
                    grp,
                    COUNT(*) AS consecutive_nulls
                FROM marked
                WHERE is_null = 1
                GROUP BY serial_number, grp
            )
            GROUP BY consecutive_nulls
            ORDER BY consecutive_nulls
            """
            
            t_col = time.time()
            try:
                df_col_gaps = con.execute(query_gap_dist).df()
                consecutive_gaps[col] = df_col_gaps
                print(f"    * {col} 분석 완료 ({time.time() - t_col:.2f}초, 공백 세그먼트: {len(df_col_gaps)}종류)")
            except Exception as e:
                print(f"    [경고] {col} 분석 중 오류 발생: {e}")
                consecutive_gaps[col] = []
                
        # 4. 시각화 캔버스 작성 (한 이미지에 모든 컬럼 이어붙이기)
        print("  - 통합 시각화 대시보드 이미지 구성 중...")
        num_cols = len(selected_cols)
        ncols_grid = 4
        nrows_grid = math.ceil(num_cols / ncols_grid)
        
        fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(20, 4.5 * nrows_grid))
        axes = axes.flatten()
        
        for idx, col in enumerate(selected_cols):
            ax = axes[idx]
            missing_cnt = col_missing_counts[col]
            missing_ratio = (missing_cnt / total_rows) * 100
            
            # 타이틀에 절대적인 결측치 수와 비율 표기
            title_text = f"{col}\n(결측치: {missing_cnt:,}개, {missing_ratio:.4f}%)"
            
            df_col_gaps = consecutive_gaps[col]
            
            if missing_cnt == 0 or len(df_col_gaps) == 0:
                # 결측치가 전혀 없는 경우
                ax.text(0.5, 0.5, "결측치 없음 (0개)", ha="center", va="center", fontsize=14, color="#7f8c8d", weight="bold")
                ax.set_title(title_text, fontsize=12, fontweight="bold", pad=10)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor("#f9f9f9")
                continue
                
            # 결측치 공백 일수 분포 그리기
            # x축: 연속 공백 일수, y축: 발생 빈도(로그 스케일 또는 일반 스케일)
            # 대부분 1일~4일 집중되므로 바 플롯으로 명확하게 표현
            # 데이터가 너무 길 경우 상위 10개만 표현하되 요약 텍스트 추가
            df_plot = df_col_gaps.copy()
            if len(df_plot) > 10:
                # 상위 9개 외 나머지는 '기타'로 합산
                df_top = df_plot.head(9).copy()
                other_count = df_plot.iloc[9:]["occurrence_count"].sum()
                # concat을 사용해 안전하게 행 추가
                other_row = pd.DataFrame([{"consecutive_nulls": "10+", "occurrence_count": other_count}])
                df_top["consecutive_nulls"] = df_top["consecutive_nulls"].astype(str)
                df_plot = pd.concat([df_top, other_row], ignore_index=True)
            else:
                df_plot["consecutive_nulls"] = df_plot["consecutive_nulls"].astype(str)
                
            sns.barplot(
                x="consecutive_nulls",
                y="occurrence_count",
                data=df_plot,
                ax=ax,
                color="#e67e22",
                edgecolor="#d35400",
                alpha=0.85
            )
            
            ax.set_title(title_text, fontsize=12, fontweight="bold", pad=10, color="#2c3e50")
            ax.set_xlabel("연속 결측 기간 (일)", fontsize=9)
            ax.set_ylabel("발생 빈도 (건)", fontsize=9)
            ax.tick_params(labelsize=9.5)
            
            # 바 위에 수치 텍스트 표시
            for p in ax.patches:
                val = p.get_height()
                if val > 0:
                    # Y축 값이 클 때 텍스트가 겹치지 않게 천 단위 콤마 포맷팅
                    ax.annotate(
                        f"{int(val):,}", 
                        (p.get_x() + p.get_width() / 2., val), 
                        ha="center", va="bottom", 
                        fontsize=8, color="#2c3e50", xytext=(0, 2),
                        textcoords="offset points"
                    )
            
            # Y축의 스케일 차이가 클 수 있으므로 자동 스케일 조절 및 필요시 상단 여유 공간 확보
            max_val = df_plot["occurrence_count"].max()
            ax.set_ylim(0, max_val * 1.15)
            
        # 남는 빈 서브플롯들 숨기기
        for idx in range(num_cols, len(axes)):
            axes[idx].axis("off")
            
        plt.suptitle(
            f"SMART Raw Attributes Consecutive Missing Days (Null Gaps) Distribution Dashboard - {model}", 
            fontsize=20, fontweight="bold", color="#2c3e50", y=0.99
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        plot_path = os.path.join(output_dir, "missing_consecutive_gaps_plots.png")
        plt.savefig(plot_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  - 시각화 이미지 저장 완료: {plot_path}")
        
        con.close()
        
    print("\n" + "=" * 70)
    print("모든 모델에 대한 결측치 연속 공백 분포 분석 및 시각화 이미지 생성이 완료되었습니다!")
    print("=" * 70)

if __name__ == "__main__":
    main()
