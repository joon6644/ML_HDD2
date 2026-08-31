import os
import sys
import argparse
import time
import subprocess

def get_selected_model():
    """전처리할 모델(단일 또는 전체)을 선택합니다."""
    all_models = ["HGST_20HUH721212ALN604", "TOSHIBA_20MG07ACA14TA", "ST12000NM0007"]
    
    parser = argparse.ArgumentParser(description="전체 데이터셋 전처리 파이프라인 연달아 실행 스크립트")
    parser.add_argument("--model", choices=all_models + ["ALL"], help="전처리를 수행할 모델 선택")
    parser.add_argument("--models", nargs="+", choices=all_models + ["ALL"], help="전처리를 수행할 모델 리스트")
    parser.add_argument("--max-memory", type=str, default=None, help="DuckDB 사용 메모리 제한 (기본값: 각 모델별 지정값)")
    args, _ = parser.parse_known_args()
    
    if args.model:
        return args.model, args.max_memory
    if args.models and len(args.models) > 0:
        return args.models[0], args.max_memory
        
    if sys.stdin.isatty():
        print("=" * 60)
        print("   데이터셋 전처리 파이프라인 - 모델 선택")
        print("=" * 60)
        for i, model in enumerate(all_models, 1):
            print(f"  {i}. {model}")
        print(f"  {len(all_models) + 1}. 전체 모델 실행 (ALL)")
        print("=" * 60)
        try:
            choice = input(f"전처리할 모델의 번호를 선택하세요 (1~{len(all_models) + 1}): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(all_models):
                    return all_models[idx], args.max_memory
                elif idx == len(all_models):
                    return "ALL", args.max_memory
        except Exception:
            pass
            
    print(f"  - 번호 선택이 없거나 비대화형 환경이므로 기본값(전체 모델 실행)으로 진행합니다.")
    return "ALL", args.max_memory

def main():
    model, max_memory = get_selected_model()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    all_models = ["HGST_20HUH721212ALN604", "TOSHIBA_20MG07ACA14TA", "ST12000NM0007"]
    target_models = all_models if model == "ALL" else [model]
    
    python_exe = sys.executable
    total_start = time.time()
    results = {}
    
    print("\n" + "=" * 80)
    print(f" 선택된 전처리 대상: {', '.join(target_models)}")
    print("=" * 80)
    
    for idx, m in enumerate(target_models, 1):
        script_path = os.path.join(curr_dir, f"{m}.py")
        if not os.path.exists(script_path):
            print(f"[오류] 전처리 스크립트가 존재하지 않습니다: {script_path}")
            results[m] = "파일 없음"
            continue
            
        print(f"\n[{idx}/{len(target_models)}] {m} 전처리 스크립트 실행 시작...")
        cmd = [python_exe, script_path]
        if max_memory:
            cmd.extend(["--max-memory", max_memory])
            
        t_start = time.time()
        res = subprocess.run(cmd)
        elapsed = time.time() - t_start
        
        if res.returncode != 0:
            print(f"\n[경고] {m} 전처리 도중 오류 발생 (종료 코드: {res.returncode}, 소요시간: {elapsed:.2f}초)")
            results[m] = f"실패 (코드 {res.returncode})"
        else:
            print(f"\n>>> [{m}] 전처리 정상 완료! (소요시간: {elapsed:.2f}초)")
            results[m] = f"성공 ({elapsed:.2f}초)"
            
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(" >>> 전체 데이터셋 전처리 실행 요약 <<<")
    print("=" * 80)
    for m, status in results.items():
        print(f"  - {m}: {status}")
    print(f"  - 총 소요 시간: {total_elapsed:.2f}초")
    print("=" * 80)

if __name__ == "__main__":
    main()

