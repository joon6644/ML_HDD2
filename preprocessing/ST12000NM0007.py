import os
import sys
import argparse
import time
import duckdb

# 동일 폴더 내의 common 모듈을 임포트하기 위해 sys.path 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import run_base_preprocessing

def main():
    parser = argparse.ArgumentParser(description="ST12000NM0007 디스크 시계열 데이터 전처리 파이프라인")
    parser.add_argument(
        "--max-memory",
        type=str,
        default="6GB",
        help="DuckDB 사용 메모리 제한 (예: 4GB, 6GB, 8GB)"
    )
    args = parser.parse_args()
    
    model = "ST12000NM0007"
    max_memory = args.max_memory
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")
    tmp_dir = os.path.join(project_dir, ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    input_file = os.path.join(data_dir, f"{model}.parquet")
    output_file = os.path.join(data_dir, f"{model}_preprocessed.parquet")
    temp_output_file = os.path.join(data_dir, f"{model}_preprocessed_temp.parquet")
    db_file = os.path.join(tmp_dir, f"preprocess_{model}.db")
    
    if not os.path.exists(input_file):
        print(f"[오류] 입력 데이터 파일이 존재하지 않습니다: {input_file}")
        return

    print("=" * 80)
    print(f" ST12000NM0007 전용 전처리 시작")
    print(f" - 입력 파일: {input_file}")
    print(f" - 출력 파일: {output_file}")
    print(f" - 메모리 제한: {max_memory}")
    print("=" * 80)
    
    # 기존 임시 파일 제거
    if os.path.exists(temp_output_file):
        try:
            os.remove(temp_output_file)
        except:
            pass
            
    con = None
    t_start = time.time()
    
    try:
        # 1 ~ 5단계 공통 전처리 파이프라인 실행
        con, valid_smart_cols = run_base_preprocessing(
            input_file=input_file,
            db_file=db_file,
            max_memory=max_memory,
            tmp_dir=tmp_dir
        )
        
        # 6. 임시 데이터셋 Parquet로 먼저 저장 (이후 검증의 속도 단축을 위함)
        print(f"\n[Step 6] 임시 preprocessed 파일 저장 중 (1차 스캔 및 저장): {temp_output_file}")
        t0 = time.time()
        con.execute(f"COPY final_preprocessed TO '{temp_output_file.replace('\\', '/')}' (FORMAT PARQUET)")
        print(f"  - 저장 완료 (소요시간: {time.time() - t0:.2f}초)")
        
        # 7. 중복 컬럼 검증
        print("\n[Step 7] 물리적 임시 Parquet 파일을 이용한 중복 컬럼 검증...")
        t0 = time.time()
        
        # smart_1_raw == smart_195_raw 및 smart_197_raw == smart_198_raw 검증
        validation_query = f"""
            SELECT 
                COUNT(*) AS total_rows,
                SUM(CASE WHEN "smart_1_raw" = "smart_195_raw" OR ("smart_1_raw" IS NULL AND "smart_195_raw" IS NULL) THEN 1 ELSE 0 END) AS eq_1_195,
                SUM(CASE WHEN "smart_197_raw" = "smart_198_raw" OR ("smart_197_raw" IS NULL AND "smart_198_raw" IS NULL) THEN 1 ELSE 0 END) AS eq_197_198
            FROM read_parquet('{temp_output_file.replace('\\', '/')}')
        """
        val_row = con.execute(validation_query).fetchone()
        total_rows, eq_1_195, eq_197_198 = val_row
        
        is_1_195_equal = (eq_1_195 == total_rows)
        is_197_198_equal = (eq_197_198 == total_rows)
        
        print(f"  - smart_1_raw == smart_195_raw 검증 결과: {is_1_195_equal} ({eq_1_195:,} / {total_rows:,} 일치)")
        print(f"  - smart_197_raw == smart_198_raw 검증 결과: {is_197_198_equal} ({eq_197_198:,} / {total_rows:,} 일치)")
        
        final_smart_cols = list(valid_smart_cols)
        
        # 8. 검증 결과에 따라 중복 컬럼 제거하여 최종 Parquet 저장
        if is_1_195_equal and is_197_198_equal:
            print("  - [검증 통과] 중복 컬럼 smart_195_raw 및 smart_198_raw를 최종 데이터셋에서 제거합니다.")
            if "smart_195_raw" in final_smart_cols:
                final_smart_cols.remove("smart_195_raw")
            if "smart_198_raw" in final_smart_cols:
                final_smart_cols.remove("smart_198_raw")
            
            print(f"\n[Step 8] 최종 컬럼 필터링 및 최종 저장 중: {output_file}")
            t_final = time.time()
            
            final_cols = ["serial_number", "date", "segment", "model", "failure"] + final_smart_cols
            final_cols_str = ", ".join([f'"{c}"' for c in final_cols])
            
            # 기존 최종 파일이 존재할 경우 삭제 후 저장
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except:
                    pass
            con.execute(f"""
                COPY (
                    SELECT {final_cols_str} 
                    FROM read_parquet('{temp_output_file.replace('\\', '/')}')
                ) TO '{output_file.replace('\\', '/')}' (FORMAT PARQUET)
            """)
            print(f"  - 최종 저장 완료 (소요시간: {time.time() - t_final:.2f}초)")
            
            # 임시 파일 삭제
            if os.path.exists(temp_output_file):
                try:
                    os.remove(temp_output_file)
                except:
                    pass
        else:
            print("  - [검증 실패] 컬럼 값이 일치하지 않으므로, 중복 컬럼을 제거하지 않고 기존 임시 파일을 최종본으로 대체합니다.")
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except:
                    pass
            os.rename(temp_output_file, output_file)
            
        print(f"  - 검증 및 최종 처리 완료 (소요시간: {time.time() - t0:.2f}초)")
        
        print("=" * 80)
        print(f" 전처리 완료! 총 소요시간: {time.time() - t_start:.2f}초")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[오류] 전처리 중 오류가 발생하였습니다: {e}")
    finally:
        if con is not None:
            con.close()
        # 임시 DB 파일 제거
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except:
                pass
        # 남아 있는 임시 파일 제거
        if os.path.exists(temp_output_file):
            try:
                os.remove(temp_output_file)
            except:
                pass

if __name__ == "__main__":
    main()