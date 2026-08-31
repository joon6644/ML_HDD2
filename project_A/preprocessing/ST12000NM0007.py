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
        default="16GB",
        help="DuckDB 사용 메모리 제한 (예: 4GB, 6GB, 8GB)"
    )
    args = parser.parse_args()
    
    model = "ST12000NM0007"
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
    temp_output_file = os.path.join(tmp_dir, f"{model}_preprocessed_temp.parquet")
    formatted_output_file = output_file.replace('\\', '/')
    formatted_temp_output_file = temp_output_file.replace('\\', '/')
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
        
        # final_preprocessed는 common에서 이미 물리화된 TABLE이다. 임시
        # Parquet로 썼다가 다시 읽지 않고 TABLE에서 직접 중복 컬럼을 검증한다.
        print("\n[후처리] 중복 SMART 컬럼 검증...")
        t0 = time.time()
        final_smart_cols = list(valid_smart_cols)
        duplicate_pairs = [
            ("smart_1_raw", "smart_195_raw"),
            ("smart_197_raw", "smart_198_raw"),
        ]
        available_pairs = [
            pair for pair in duplicate_pairs
            if pair[0] in final_smart_cols and pair[1] in final_smart_cols
        ]

        if available_pairs:
            match_expressions = [
                (
                    f'COUNT(*) FILTER (WHERE "{left}" '
                    f'IS NOT DISTINCT FROM "{right}")'
                )
                for left, right in available_pairs
            ]
            validation_row = con.execute(
                f"""
                SELECT COUNT(*), {", ".join(match_expressions)}
                FROM final_preprocessed
                """
            ).fetchone()
            total_rows = validation_row[0]
            for (left, right), match_count in zip(
                available_pairs, validation_row[1:]
            ):
                is_equal = match_count == total_rows
                print(
                    f"  - {left} == {right}: {is_equal} "
                    f"({match_count:,} / {total_rows:,})"
                )
                if is_equal:
                    final_smart_cols.remove(right)
        else:
            print("  - 결측률 필터 후 비교 가능한 중복 컬럼 쌍이 없습니다.")

        # 최종 Parquet은 한 번만 기록한다.
        print(f"\n[저장] 최종 Parquet 저장 중: {output_file}")
        final_cols = [
            "serial_number", "date", "segment", "model", "failure",
            *final_smart_cols,
        ]
        final_cols_str = ", ".join(f'"{col}"' for col in final_cols)
        if os.path.exists(output_file):
            os.remove(output_file)
        con.execute(
            f"""
            COPY (
                SELECT {final_cols_str}
                FROM final_preprocessed
            ) TO '{formatted_output_file}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        print(f"  - 검증 및 단일 저장 완료 ({time.time() - t0:.2f}초)")
        
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
