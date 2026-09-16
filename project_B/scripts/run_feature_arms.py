"""파생변수 구성별 GRU 성능 — 어떤 파생이 도시바+GRU 에서도 듣는지 본다.

    python scripts/run_feature_arms.py
    python scripts/run_feature_arms.py --arms asfd7,roll30

현재 논문은 원본 SMART 16개만 쓴다 (tos_select_pauc 의 features.base.* 전부 off).
옛 계보(HGST + XGBoost)에서는 파생변수가 FAR 1% 재현율을 0.532 -> 0.638 로
끌어올렸다. 드라이브도 모델도 다르므로 그대로 옮겨간다는 보장이 없어, 같은
구성 몇 가지를 도시바 + GRU 로 다시 재는 것이 이 스크립트다.

━━ 조건 ━━

모델은 gru_pauc (표 1 의 GRU 행과 같은 하이퍼파라미터, BCE). 피처 외의 축을
전부 고정해야 차이가 파생변수로 귀속된다. 하이퍼파라미터는 원본 피처 기준으로
고른 값이므로, 파생이 효과를 보이면 그 위에서 다시 탐색해야 한다.

━━ 채점 ━━

두 임곗값 기준을 모두 뽑는다 (val_quantile / month_quantile).
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
MODEL = "configs/models/gru_pauc.yaml"

# 옛 계보 상위 구성. 값은 features.base.* 오버라이드다.
ARMS: dict[str, dict] = {
    "asfd7": {
        "features.base.asfd_windows": [7],
    },
    "roll30": {
        "features.base.rolling_windows": [30],
        "features.base.rolling_stats": ["mean", "std"],
    },
    # 4주. 3.3 의 "일주일 단위의 변동" 근거와 눈금이 맞는다. 이동통계는
    # 윈도우를 자르기 전에 전체 타임라인에서 계산되므로, 14일 입력 윈도우
    # 안의 각 행이 자기 시점 기준 28일치 통계를 들고 들어간다 — 즉 모델이
    # 창 밖 과거까지 참조하게 된다. 미래는 보지 않으므로 누출이 아니다.
    "roll28": {
        "features.base.rolling_windows": [28],
        "features.base.rolling_stats": ["mean", "std"],
    },
    # 평균만. roll28 은 속성당 2개(평균·표준편차)를 붙여 채널이 17 -> 31 로
    # 늘고 학습이 2~3배 느려진다. 표준편차가 실제로 기여하는지 갈라 본다.
    "roll28mean": {
        "features.base.rolling_windows": [28],
        "features.base.rolling_stats": ["mean"],
    },
    # 표준편차만. 평균/표준편차 중 어느 쪽이 이득을 내는지 가른다.
    "roll28std": {
        "features.base.rolling_windows": [28],
        "features.base.rolling_stats": ["std"],
    },
    "diff": {
        "features.base.diff_lags": [1, 7, 30],
    },
    "combo_roll30_asfd7": {
        "features.base.rolling_windows": [30],
        "features.base.rolling_stats": ["mean", "std"],
        "features.base.asfd_windows": [7],
    },
}


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    print(f"  → rc={rc} ({time.time() - t0:.0f}s)", flush=True)
    return rc


def write_experiment(arm: str, overrides: dict) -> str:
    text = (ROOT / "configs" / "experiments" / f"{BASE}.yaml").read_text(encoding="utf-8")
    base = yaml.safe_load("experiment:" + text.split("experiment:", 1)[1])
    exp = f"tosfeat_{arm}"
    base["experiment"] = exp
    base["models"] = [MODEL]
    o = base["overrides"]
    # 다른 모델을 가리키는 오버라이드는 지우고, gru_pauc 것만 남긴다.
    for k in [k for k in o if k.startswith("models.")]:
        del o[k]
    o["models.gru_pauc.training.loss"] = "bce"
    o["models.gru_pauc.training.auto_pos_weight"] = False
    # 파생 설정을 덮어쓴다. 켜지 않은 축은 기반 실험대로 꺼진 채 남는다.
    o.update(overrides)
    (ROOT / "configs" / "experiments" / f"{exp}.yaml").write_text(
        f"# 파생변수 구성 '{arm}'. scripts/run_feature_arms.py 가 생성한다.\n"
        f"# 켠 것: {', '.join(overrides)}\n\n"
        + yaml.safe_dump(base, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    return exp


def main() -> int:
    ap = argparse.ArgumentParser(description="파생변수 구성별 GRU 성능")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--python", default=PY, help="학습에 쓸 인터프리터 (GPU venv)")
    args = ap.parse_args()
    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    py = args.python

    started, failed = time.time(), []
    for i, arm in enumerate(want, 1):
        if arm not in ARMS:
            print(f"[건너뜀] 알 수 없는 구성: {arm}")
            continue
        print(f"\n{'=' * 66}\n[{i}/{len(want)}] {arm}: {ARMS[arm]}\n{'=' * 66}", flush=True)
        exp = write_experiment(arm, ARMS[arm])
        run([py, "scripts/run_experiment.py", f"configs/experiments/{exp}.yaml"])
        metrics = (ROOT / "runs" / exp / "TOSHIBA_20MG07ACA14TA" / "gru_pauc"
                   / "seed42" / "fold00" / "fold_metrics.json")
        if not metrics.exists():
            print("  → 학습 산출물 없음", flush=True)
            failed.append(arm)
            continue
        run([py, "scripts/export_results.py", exp, "--threshold-source", "val_quantile"])
        run([py, "scripts/export_results.py", exp, "--threshold-source",
             "month_quantile", "--out", f"results/{exp}_oracle"])

    print(f"\n[완료] {len(want) - len(failed)}/{len(want)} "
          f"({(time.time() - started) / 60:.0f}분)", flush=True)
    if failed:
        print(f"[실패] {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
