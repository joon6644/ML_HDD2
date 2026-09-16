"""탐색을 마친 비시퀀스 4종을 학습·채점한다 — 표 1 의 탐색 후 행.

    python scripts/run_tuned_baselines.py                    # 전부
    python scripts/run_tuned_baselines.py --models xgboost,lightgbm

run_optuna_tabular.py 가 만든 configs/models/{name}_tuned.yaml 을 읽어
실험 설정을 생성하고, tos_select_pauc 과 같은 분할·피처로 돌린 뒤 두 임곗값
기준으로 채점한다.

  val_quantile    검증에서 정한 임곗값을 그대로 (표 1 계열)
  month_quantile  각 test 월에서 목표 오탐률로 다시 자른 값 (표 2 계열)

GRU 판(tos_proposed)과 같은 절차다. 다른 점은 모델뿐이다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
BASE = "tos_select_pauc"
NAMES = ["randomforest", "xgboost", "lightgbm", "mlp"]


def run(cmd: list[str]) -> bool:
    print(f"\n$ {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    print(f"  → rc={rc} ({time.time() - t0:.0f}s)", flush=True)
    return rc == 0


def write_experiment(name: str, tag: str = "", base_exp: str = BASE) -> str | None:
    """탐색 1위 설정 하나만 담은 실험 yaml 을 만든다.

    tag 는 run_optuna_tabular.py 의 --tag 와 같아야 한다 (예: roll28).
    base_exp 는 분할·피처를 가져올 실험이다 — 파생변수 판에서는 그 피처
    설정을 담은 실험(tosfeat_roll28 등)을 줘야 학습 입력이 일치한다.
    """
    suffix = f"_{tag}" if tag else ""
    model_cfg = ROOT / "configs" / "models" / f"{name}_pauc_tuned{suffix}.yaml"
    if not model_cfg.exists():
        print(f"[건너뜀] {model_cfg} 없음 — 탐색을 먼저 끝내라", flush=True)
        return None
    text = (ROOT / "configs" / "experiments" / f"{base_exp}.yaml").read_text(encoding="utf-8")
    base = yaml.safe_load("experiment:" + text.split("experiment:", 1)[1])
    exp = f"tos_tuned_{name}{suffix}"
    base["experiment"] = exp
    base["models"] = [f"configs/models/{name}_pauc_tuned{suffix}.yaml"]
    # 기반 실험의 models.* 오버라이드는 다른 모델 이름을 가리키므로 지운다.
    o = base["overrides"]
    for k in [k for k in o if k.startswith("models.")]:
        del o[k]
    (ROOT / "configs" / "experiments" / f"{exp}.yaml").write_text(
        f"# {name} 탐색 1위 설정으로 돌리는 판. scripts/run_tuned_baselines.py 생성.\n\n"
        + yaml.safe_dump(base, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    return exp


def main() -> int:
    ap = argparse.ArgumentParser(description="탐색 후 비시퀀스 모델 채점")
    ap.add_argument("--models", default=",".join(NAMES))
    ap.add_argument("--tag", default="", help="run_optuna_tabular --tag 와 같은 값")
    ap.add_argument("--base-experiment", default=BASE,
                    help="분할·피처를 가져올 실험 (파생변수 판이면 tosfeat_*)")
    args = ap.parse_args()
    want = [m.strip() for m in args.models.split(",") if m.strip()]

    started, failed = time.time(), []
    for i, name in enumerate(want, 1):
        print(f"\n{'=' * 66}\n[{i}/{len(want)}] {name}\n{'=' * 66}", flush=True)
        exp = write_experiment(name, args.tag, args.base_experiment)
        if exp is None:
            failed.append(name)
            continue
        if not run([PY, "scripts/run_experiment.py",
                    f"configs/experiments/{exp}.yaml"]):
            # CUDA 종료 잡음으로 rc 가 0이 아닐 수 있어 산출물로 판단한다.
            print("  (rc 비정상 — 산출물로 재확인)", flush=True)
        ok = run([PY, "scripts/export_results.py", exp,
                  "--threshold-source", "val_quantile"])
        ok &= run([PY, "scripts/export_results.py", exp,
                   "--threshold-source", "month_quantile",
                   "--out", f"results/{exp}_oracle"])
        if not ok:
            failed.append(name)

    print(f"\n[완료] {len(want) - len(failed)}/{len(want)} "
          f"({(time.time() - started) / 3600:.1f}시간)", flush=True)
    if failed:
        print(f"[실패] {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
