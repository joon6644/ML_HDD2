import os
import sys
import argparse
import time
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_selected_model():
    """사용자가 분석할 단 하나의 모델 데이터셋을 지정하게 합니다."""
    all_models = ["TOSHIBA_20MG08ACA16TA", "TOSHIBA_20MG07ACA14TA", "ST12000NM0007"]
    
    # 1. 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="지정된 모델의 전체 데이터를 사용하여 SMART Raw 컬럼별 기술통계량 계산 및 분포/사분위수 통합 시각화를 수행합니다.")
    parser.add_argument(
        "--model", 
        choices=all_models,
        help="분석할 모델명을 입력합니다. 생략 시 대화형 선택 메뉴를 띄웁니다."
    )
    args, unknown = parser.parse_known_args()
    
    if args.model:
        return args.model
        
    # 2. 대화형 선택 메뉴 제공
    if sys.stdin.isatty():
        print("=" * 60)
        print("   기술통계량 및 시각화 분석 (전체 데이터 사용) - 모델 선택")
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
            
    # 3. 기본값으로 첫 번째 모델 지정
    print(f"  - 모델이 지정되지 않아 기본 모델({all_models[0]})로 분석을 진행합니다.")
    return all_models[0]

def main():
    model = get_selected_model()
    
    project_dir = r"C:\Workspace\projects\26_2_COIN"
    data_dir = os.path.join(project_dir, "data")
    eda_dir = os.path.join(project_dir, "EDA")
    
    file_path = os.path.join(data_dir, f"{model}.parquet")
    output_dir = os.path.join(eda_dir, model)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(file_path):
        print(f"[오류] 입력 파일이 존재하지 않습니다: {file_path}")
        return
        
    con = duckdb.connect(database=":memory:")
    
    print("\n" + "=" * 70)
    print(f"기술통계량 및 시각화 분석 시작 (전체 데이터 사용): {model}")
    print("=" * 70)
    
    # 1. 결측치 비율 보고서 로드 또는 즉시 계산
    csv_report_path = os.path.join(output_dir, "missing_values_report.csv")
    selected_cols = []
    
    if os.path.exists(csv_report_path):
        print("  - 기존 결측치 보고서 로드 중...")
        df_missing = pd.read_csv(csv_report_path)
    else:
        print("  - 결측치 보고서가 발견되지 않아 결측율을 즉시 계산합니다...")
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
                missing_data.append({
                    "column_name": col,
                    "missing_ratio": missing_ratio
                })
            df_missing = pd.DataFrame(missing_data)
        except Exception as e:
            print(f"  [오류] 결측비율 계산 실패: {e}")
            return
            
    # 조건 만족하는 컬럼 필터링: 'smart_' 포함, '_raw'로 끝남, 결측치 10% 미만(ratio < 0.1)
    filtered_df = df_missing[
        df_missing["column_name"].str.contains("smart_") & 
        df_missing["column_name"].str.endswith("_raw") & 
        (df_missing["missing_ratio"] < 0.1)
    ]
    selected_cols = filtered_df["column_name"].tolist()
    
    if not selected_cols:
        print("  [경고] 결측치 10% 미만의 SMART raw 컬럼을 찾지 못했습니다.")
        return
        
    print(f"  - 조건에 부합하는 컬럼 수: {len(selected_cols)}개")
    
    # 2. 상수 컬럼(모든 값이 동일하거나 결측치를 제외한 유효 값이 1개 이하인 컬럼) 식별 및 제거
    print("  - 상수 컬럼 식별 중...")
    non_const_cols = []
    constant_cols = []
    
    check_items = []
    for col in selected_cols:
        c_cast = f'TRY_CAST("{col}" AS DOUBLE)'
        check_items.append(f'(MIN({c_cast}) = MAX({c_cast}) OR COUNT({c_cast}) <= 1) AS "is_const_{col}"')
        
    query_const = f"SELECT {', '.join(check_items)} FROM read_parquet('{file_path.replace('\\', '/')}')"
    try:
        res_const = con.execute(query_const).df().to_dict(orient="records")[0]
        for col in selected_cols:
            if res_const[f"is_const_{col}"]:
                constant_cols.append(col)
            else:
                non_const_cols.append(col)
        
        print(f"  - 제거된 상수 컬럼 ({len(constant_cols)}개): {constant_cols}")
        selected_cols = non_const_cols
    except Exception as e:
        print(f"  [경고] 상수 컬럼 식별 중 오류 발생 (전체 컬럼 대상 진행): {e}")
        
    if not selected_cols:
        print("  [경고] 상수 컬럼을 제외한 후 분석 대상 SMART raw 컬럼이 존재하지 않습니다.")
        return
        
    print(f"  - 최종 분석 대상 컬럼 수 (상수 제외): {len(selected_cols)}개")
    
    # 3. 시각화 이미지 캔버스 구성
    print("  - 시각화 이미지 캔버스 구성 중...")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Malgun Gothic", "Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    
    nrows = len(selected_cols)
    ncols = 2
    fig_height = nrows * 3.2 + 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, fig_height))
    
    if nrows == 1:
        axes = axes.reshape(1, 2)
        
    stats_data = []
    
    # 4. 컬럼별 순회 분석 및 시각화 (메모리 보호를 위해 1개의 컬럼씩 로딩 및 처리)
    print("  - 컬럼별 전체 데이터 로드 및 통계/시각화 연산 중...")
    t_start = time.time()
    
    for idx, col in enumerate(selected_cols):
        t_col = time.time()
        print(f"    [{idx+1}/{nrows}] {col} 처리 중...", end="", flush=True)
        
        # DuckDB를 사용하여 단일 컬럼만 캐스팅하여 로드
        query = f'SELECT TRY_CAST("{col}" AS DOUBLE) AS "{col}" FROM read_parquet(\'{file_path.replace("\\", "/")}\')'
        col_df = con.execute(query).df()
        data_series = col_df[col].dropna().values
        
        count_val = len(data_series)
        if count_val == 0:
            # 결측 데이터만 있는 예외 상황 처리
            stats_data.append({
                "column_name": col, "count": 0, "mean": np.nan, "std": np.nan,
                "min": np.nan, "25%": np.nan, "50%": np.nan, "75%": np.nan, "max": np.nan
            })
            ax_hist = axes[idx, 0]
            ax_box = axes[idx, 1]
            ax_hist.text(0.5, 0.5, "No Data (All Null)", ha="center", va="center")
            ax_box.text(0.5, 0.5, "No Data (All Null)", ha="center", va="center")
            del col_df
            print(" (결측치로 인해 건너뜀)")
            continue
            
        # 정확한 기술통계량 산출 (NumPy C-extension 활용으로 초고속 연산)
        min_val = float(np.min(data_series))
        max_val = float(np.max(data_series))
        mean_val = float(np.mean(data_series))
        std_val = float(np.std(data_series))
        q25, q50, q75 = np.percentile(data_series, [25, 50, 75])
        q25, q50, q75 = float(q25), float(q50), float(q75)
        
        stats_data.append({
            "column_name": col,
            "count": count_val,
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "25%": q25,
            "50%": q50,
            "75%": q75,
            "max": max_val
        })
        
        # 4. 시각화 그리기
        ax_hist = axes[idx, 0]
        ax_box = axes[idx, 1]
        
        # 4a. 값 분포 (히스토그램) 그리기
        # 편차가 없는 단일 값 컬럼 분기 (최소값 == 최대값)
        if min_val == max_val:
            ax_hist.bar([min_val], [count_val], width=0.1, color="#3498db", edgecolor="#2c3e50", alpha=0.7)
            ax_hist.set_title(f"{col} - Constant Value ({min_val})", fontsize=11, fontweight="bold", color="#2c3e50")
        else:
            # 고성능 numpy.histogram 연산 후 stairs로 드로잉 (메모리 점유율 최소화)
            counts, bins = np.histogram(data_series, bins=50)
            ax_hist.stairs(counts, bins, fill=True, color="#3498db", edgecolor="#2c3e50", alpha=0.7)
            ax_hist.set_title(f"{col} - Value Distribution (All Data)", fontsize=11, fontweight="bold", color="#2c3e50")
            
        # 4b. 사분위수(boxplot) 그리기
        # 박스플롯 이상치 기준 계산
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        # 위스커 경계값 계산 (이상치가 아닌 값들의 최소/최대)
        non_outliers = data_series[(data_series >= lower_bound) & (data_series <= upper_bound)]
        whislo = float(non_outliers.min()) if len(non_outliers) > 0 else q25
        whishi = float(non_outliers.max()) if len(non_outliers) > 0 else q75
        
        # 아웃라이어 필터링
        outliers = data_series[(data_series < lower_bound) | (data_series > upper_bound)]
        total_outliers = len(outliers)
        
        # 아웃라이어 드로잉 병목을 피하기 위해 시각화용 샘플링 진행 (통계값에는 전체 데이터 반영됨)
        if total_outliers > 1000:
            np.random.seed(42)
            sampled_fliers = np.random.choice(outliers, size=1000, replace=False)
        else:
            sampled_fliers = outliers
            
        bxp_stats = [{
            'med': q50,
            'q1': q25,
            'q3': q75,
            'whislo': whislo,
            'whishi': whishi,
            'fliers': sampled_fliers
        }]
        
        # ax.bxp로 정밀하고 즉각적인 박스플롯 드로잉
        ax_box.bxp(bxp_stats, orientation='horizontal', widths=0.4, patch_artist=True,
                   boxprops=dict(facecolor="#e74c3c", edgecolor="#2c3e50"),
                   medianprops=dict(color="#2c3e50", linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor='#e74c3c', markeredgecolor='none', alpha=0.25, markersize=3))
        ax_box.set_title(f"{col} - Boxplot (All Data, Outliers Sampled)", fontsize=11, fontweight="bold", color="#2c3e50")
        
        # 디테일 조정
        ax_hist.tick_params(labelsize=9)
        ax_box.tick_params(labelsize=9)
        ax_hist.set_xlabel("")
        ax_hist.set_ylabel("Count", fontsize=9)
        ax_box.set_xlabel("")
        ax_box.set_yticks([])  # 불필요한 Y축 틱 레이블 제거
        
        # 메모리 해제
        del col_df
        print(f" 완료 ({time.time() - t_col:.2f}초)")
        
    # 5. 기술통계량 CSV 저장
    df_stats = pd.DataFrame(stats_data)
    stats_csv_path = os.path.join(output_dir, "descriptive_stats.csv")
    df_stats.to_csv(stats_csv_path, index=False)
    print(f"  - 전체 데이터 기준 기술통계량 CSV 저장 완료: {stats_csv_path}")
    
    # 6. 이미지 저장
    print("  - 하나의 고해상도 이미지 파일로 렌더링 및 저장 중...")
    fig.suptitle(
        f"Descriptive Statistics Visualization - {model} (All Data Used)",
        fontsize=22,
        fontweight="bold",
        color="#2c3e50",
        y=0.995
    )
    
    subtitle_text = (
        f"• 분석 모델: {model}\n"
        f"• 생성일시: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    props = dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#bdc3c7", alpha=0.9)
    fig.text(0.5, 0.985, subtitle_text, ha="center", va="top", fontsize=11, bbox=props, color="#2c3e50")
    
    plt.subplots_adjust(top=0.97, hspace=0.45, wspace=0.25)
    
    plot_img_path = os.path.join(output_dir, "descriptive_stats_plots.png")
    plt.savefig(plot_img_path, dpi=120, bbox_inches="tight")
    plt.close()
    
    print(f"  - 시각화 이미지 통합 저장 완료: {plot_img_path}")
    print(f"  - 총 소요 시간: {time.time() - t_start:.2f}초")
    
    con.close()
    print("\n" + "=" * 70)
    print("전체 데이터를 반영한 기술통계량 분석 및 시각화 통합 파일 저장이 완료되었습니다!")
    print("=" * 70)

if __name__ == "__main__":
    main()
