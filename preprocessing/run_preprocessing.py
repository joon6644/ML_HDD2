import os
import sys
import argparse
import subprocess

def get_selected_model():
    all_models = ["HGST_20HUH721212ALN604", "TOSHIBA_20MG07ACA14TA", "ST12000NM0007", "ALL"]
    
    parser = argparse.ArgumentParser(description="전체 데이터셋 전처리 파이프라인 실행 스크립트")
    parser.add_argument("--model", choices=all_models, help="전처리를 수행할 모델 선택")
    parser.add_argument("--max-memory", type=str, default="6GB", help="DuckDB 메모리 제한")
    args, _ = parser.parse_known_args()
    
    if args.model:
        return args.model, args.max_memory
        
    if sys.stdin.isatty():
        print("=" * 60)
        print("   데이터셋 전처리 파이프라인 - 모델 선택")
        print("=" * 60)
        print("  1. HGST_20HUH721212ALN604")
        print("  2. TOSHIBA_20MG07ACA14TA")
        print("  3. ST12000NM0007")
        print("  4. 전체 모델 실행 (ALL)")
        print("=" * 60)
        try:
            choice = input("전처리할 모델의 번호를 선택하세요 (1~4): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(all_models):
                    return all_models[idx], args.max_memory
        except Exception:
            pass
            
    return "ALL", args.max_memory

def main():
    model, max_memory = get_selected_model()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    target_models = ["HGST_20HUH721212ALN604", "TOSHIBA_20MG07ACA14TA", "ST12000NM0007"] if model == "ALL" else [model]
    
    python_exe = sys.executable
    
    for m in target_models:
        script_path = os.path.join(curr_dir, f"{m}.py")
        if not os.path.exists(script_path):
            print(f"[오류] 전처리 스크립트가 존재하지 않습니다: {script_path}")
            continue
            
        print(f"\n>>> [{m}] 전처리 스크립트 실행 시작...")
        cmd = [python_exe, script_path, "--max-memory", max_memory]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[경고] {m} 전처리 도중 오류 발생 (종료 코드: {res.returncode})")
        else:
            print(f">>> [{m}] 전처리 정상 완료!")

if __name__ == "__main__":
    main()
