"""TCN 탐색 이후의 후속 실행을 한 번에 돌린다.

    python scripts/run_tcn_followup.py               # 전부
    python scripts/run_tcn_followup.py --only proposed
    python scripts/run_tcn_followup.py --only sweep

run_optuna_tcn.py 가 configs/models/tcn_tuned.yaml 을 만든 뒤에 돌린다.

━━ 무엇을 뽑는가 ━━

1) tos_proposed_tcn — 탐색으로 고른 TCN. 논문 표 1의 "Optimized GRU" 행에
   대응하는 TCN 행이다. 이게 결정에 필요한 유일한 수치다: 같은 절차를
   TCN 에 적용했을 때 튜닝된 GRU(pAUC 0.879 / 재현율 0.778)를 넘는가.

2) tcnlb_* — 입력 윈도우 민감도(표 2)를 TCN 으로 다시 뽑는다.

   레벨 수는 수용 영역 규칙에 맞춰 길이마다 바꾼다 (7일 n=2, 14·21일 n=3,
   30·60일 n=4). RF = 1 + 2(k-1)(2^n - 1) 가 창을 덮어야 하기 때문이다.

   레벨 수를 고정하지 않는 이유: 그 길이로 실제 쓴다면 구조도 그 길이에
   맞춰 잡는다. 따라서 이 판이 "커널 크기를 적절히 잡았다는 전제에서 창
   길이를 바꿨을 때 기대되는 성능" 이다. n 을 고정하면 짧은 창에 과한
   구조를, 긴 창에 모자란 구조를 강요하게 되어 오히려 현실과 멀어진다.

   대신 용량이 길이와 함께 움직이므로, GRU 판(toslb_*)처럼 "길이만 바꾸고
   나머지 동일" 이라고는 쓸 수 없다. 표에 붙일 설명에서 이 점을 밝혀야 한다.

━━ 채점 ━━

기존 스윕과 같게 두 기준을 모두 뽑는다.
  val_quantile    검증에서 정한 임곗값을 그대로 (표 1 계열)
  month_quantile  각 test 월에서 목표 오탐률로 다시 자른 값 (표 2 계열,
                  파일명 접미사 _oracle)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

PROPOSED = ["tos_proposed_tcn"]
SWEEP = [f"tcnlb_{L}_pauc" for L in (7, 14, 21, 30, 60)]


def run(cmd: list[str]) -> bool:
    print(f"\n$ {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT)
    ok = proc.returncode == 0
    print(f"  → {'OK' if ok else '실패(rc=%d)' % proc.returncode}  ({time.time()-t0:.0f}s)",
          flush=True)
    return ok


def do(experiment: str) -> bool:
    """학습 후 두 임곗값 기준으로 채점한다.

    run_experiment.py 는 yaml **경로**를 받고 export_results.py 는 실험
    **이름**을 받는다. 둘의 인자 규약이 달라서 여기서 맞춰 준다.
    """
    cfg_path = f"configs/experiments/{experiment}.yaml"
    if not (ROOT / cfg_path).exists():
        print(f"  → 건너뜀: {cfg_path} 없음")
        return False
    if not run([PY, "scripts/run_experiment.py", cfg_path]):
        return False
    ok = run([PY, "scripts/export_results.py", experiment,
              "--threshold-source", "val_quantile"])
    ok &= run([PY, "scripts/export_results.py", experiment,
               "--threshold-source", "month_quantile",
               "--out", f"results/{experiment}_oracle"])
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="TCN 후속 실행")
    ap.add_argument("--only", choices=["proposed", "sweep", "figures"], default=None)
    args = ap.parse_args()

    tuned = ROOT / "configs" / "models" / "tcn_tuned.yaml"
    if args.only in (None, "proposed") and not tuned.exists():
        print(f"[중단] {tuned} 가 없다. 먼저 run_optuna_tcn.py 를 끝내라.")
        return 1

    if args.only == "proposed":
        plan, figures = PROPOSED, False
    elif args.only == "sweep":
        plan, figures = SWEEP, False
    elif args.only == "figures":
        plan, figures = [], True
    else:
        plan, figures = PROPOSED + SWEEP, True

    print(f"[계획] 실험 {len(plan)}개"
          + (f": {', '.join(plan)}" if plan else "")
          + (" + 그림 2종" if figures else ""), flush=True)
    started = time.time()
    failed = []
    for i, exp in enumerate(plan, 1):
        print(f"\n{'='*66}\n[{i}/{len(plan)}] {exp}\n{'='*66}", flush=True)
        if not do(exp):
            failed.append(exp)

    if figures:
        # 그림 1 — 탐색 전후 TCN 의 부분 ROC 곡선. tcnlb_14_pauc 와
        # tos_proposed_tcn 이 모두 끝난 뒤라야 그릴 수 있다.
        print(f"\n{'='*66}\n[그림 1] 부분 ROC 곡선\n{'='*66}", flush=True)
        if not run([PY, "scripts/run_curve.py", "--arms", "tcn"]):
            failed.append("run_curve(tcn)")
        # 그림 2 — 시점 x 속성 IG 히트맵. GRU 판과 같은 설정
        # (기준점 = 학습 구간 속성별 평균, 대상 = 고장 임박 양성 표본).
        print(f"\n{'='*66}\n[그림 2] IG 히트맵\n{'='*66}", flush=True)
        if not run([PY, "scripts/run_ig.py",
                    "--experiment", "tos_proposed_tcn", "--model", "tcn_tuned",
                    "--baseline", "train_mean", "--positives-only"]):
            failed.append("run_ig(tcn)")

    total = len(plan) + (2 if figures else 0)
    print(f"\n[완료] {total-len(failed)}/{total} 성공, "
          f"총 {(time.time()-started)/3600:.1f}시간", flush=True)
    if failed:
        print(f"[실패] {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
