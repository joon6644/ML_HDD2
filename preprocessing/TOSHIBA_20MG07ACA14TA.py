import os
import sys
import argparse
import time
import duckdb

# 동일 폴더 내의 common 모듈을 임포트하기 위해 sys.path 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import run_base_preprocessing

def main():
    parser = argparse.ArgumentParser(description="TOSHIBA_20MG07ACA14TA 디스크 시계열 데이터 전처리 파이프라인")
    parser.add_argument(
        "--max-memory",
        type=str,
        default="48GB",
        help="DuckDB 사용 메모리 제한 (예: 4GB, 6GB, 8GB)"
    )
    args = parser.parse_args()
    
    model = "TOSHIBA_20MG07ACA14TA"
    max_memory = args.max_memory
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "preprocessed")
    os.makedirs(processed_dir, exist_ok=True)
    tmp_dir = os.path.join(project_dir, ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    input_file = os.path.join(raw_dir, f"{model}.parquet")
    output_file = os.path.join(processed_dir, f"{model}_preprocessed.parquet")
    formatted_output_file = output_file.replace('\\', '/')
    temp_output_file = os.path.join(tmp_dir, f"{model}_preprocessed_temp.parquet")
    db_file = os.path.join(tmp_dir, f"preprocess_{model}.db")
    
    if not os.path.exists(input_file):
        print(f"[오류] 입력 데이터 파일이 존재하지 않습니다: {input_file}")
        return

    print("=" * 80)
    print(f" TOSHIBA_20MG07ACA14TA 전용 전처리 시작")
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
        
        # 6. 전처리 완료 데이터 ZSTD 압축 Parquet로 최종 저장
        print(f"\n[Step 6] 전처리 데이터셋 최종 저장 중: {output_file}")
        t0 = time.time()
        
        final_cols = ["serial_number", "date", "segment", "model", "failure"] + valid_smart_cols
        final_cols_str = ", ".join([f'"{c}"' for c in final_cols])
        
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except:
                pass
                
        con.execute(f"""
            COPY (
                SELECT {final_cols_str}
                FROM final_preprocessed
            ) TO '{formatted_output_file}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        print(f"  - 최종 저장 완료 (소요시간: {time.time() - t0:.2f}초)")
        
        print("=" * 80)
        print(f" TOSHIBA 전처리 완료! 총 소요시간: {time.time() - t_start:.2f}초")
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
        if os.path.exists(temp_output_file):
            try:
                os.remove(temp_output_file)
            except:
                pass

if __name__ == "__main__":
    main()
