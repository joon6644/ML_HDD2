"""XGBoost 탐색의 개별 시행을 테스트 구간에서 채점한다.

    python scripts/score_xgb_trials.py --trials 34,16,4,0

탐색은 val pAUC 로 한 시행(trial 34)만 고르는데, 그 시행의 테스트 pAUC 가
탐색 전(0.8637)보다 낮았다. 탐색 도중 한동안 1위였던 조합들은 어떤지 본다.

주의: 이 표는 **선정 근거로 쓸 수 없다.** test 를 여러 번 들여다본 결과라
그중 최댓값은 "이 탐색 공간에서 도달 가능한 상한" 이지 held-out 성능이
아니다. score_trials_936.py 의 경고와 같은 취지다.

채점은 논문 3.4 와 같다 — 달마다 디스크 단위 pAUC@FAR<=5% 를 내고 평균한다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

PY = sys.executable
TRIALS_CSV = ROOT / "results" / "optuna_xgboost_trials.csv"
BASE_EXPERIMENT = "tos_select_pauc"
PAUC_MAX_FPR = 0.05

from run_optuna_tabular import FIXED, suggest  # noqa: E402


class _Replay:
    """기록된 파라미터로 suggest() 를 그대로 재생한다."""

    def __init__(self, params: dict):
        self.p = params

    def suggest_categorical(self, k, _):
        return self._get(k)

    def suggest_int(self, k, *a, **kw):
        return int(self._get(k))

    def suggest_float(self, k, *a, **kw):
        return float(self._get(k))

    def _get(self, k):
        v = self.p[k]
        return v.item() if hasattr(v, "item") else v


def write_configs(row, n: int) -> tuple[str, str]:
    """시행 하나의 모델·실험 설정을 쓴다. (실험 이름, 모델 이름)."""
    model_name = f"xgb_t{n}"
    params, training = suggest(_Replay(row), "xgboost")
    header = (
        f"# XGBoost 탐색 trial {n} (val pAUC {row['pauc']:.4f}).\n"
        "# 탐색이 고른 시행이 아니라, test 로 줄 세워 보기 위한 재현용이다.\n"
        "# scripts/score_xgb_trials.py 가 생성한다.\n\n"
    )
    (ROOT / "configs" / "models" / f"{model_name}.yaml").write_text(
        header + yaml.safe_dump(
            {"name": model_name, "family": "tabular",
             "class": "hddpred.models.trees.XGBoostModel",
             "params": params, "training": training},
            allow_unicode=True, sort_keys=False),
        encoding="utf-8")

    text = (ROOT / "configs" / "experiments" / f"{BASE_EXPERIMENT}.yaml"
            ).read_text(encoding="utf-8")
    base = yaml.safe_load("experiment:" + text.split("experiment:", 1)[1])
    exp = f"tosxgb_t{n}"
    base["experiment"] = exp
    base["models"] = [f"configs/models/{model_name}.yaml"]
    for k in [k for k in base["overrides"] if k.startswith("models.")]:
        del base["overrides"][k]
    (ROOT / "configs" / "experiments" / f"{exp}.yaml").write_text(
        header + yaml.safe_dump(base, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    return exp, model_name


def main() -> int:
    ap = argparse.ArgumentParser(description="XGBoost 시행별 테스트 채점")
    ap.add_argument("--trials", default="34,16,4,0")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    frame = pd.read_csv(TRIALS_CSV).set_index("trial")
    want = [int(x) for x in args.trials.split(",") if x.strip()]

    from run_curve import disk_scores

    rows, started = [], time.time()
    for i, n in enumerate(want, 1):
        if n not in frame.index:
            print(f"[건너뜀] trial {n}: 기록 없음", flush=True)
            continue
        row = frame.loc[n].to_dict()
        row["pauc"] = float(frame.loc[n, "pauc"])
        exp, model_name = write_configs(row, n)
        print(f"\n{'=' * 66}\n[{i}/{len(want)}] trial {n} "
              f"(val pAUC {row['pauc']:.4f})\n{'=' * 66}", flush=True)
        if not args.skip_train:
            subprocess.run([PY, "scripts/run_experiment.py",
                            f"configs/experiments/{exp}.yaml"], cwd=ROOT)
        metrics = (ROOT / "runs" / exp / "TOSHIBA_20MG07ACA14TA" / model_name
                   / "seed42" / "fold00" / "fold_metrics.json")
        if not metrics.exists():
            print("  → 학습 산출물 없음, 건너뜀", flush=True)
            continue
        months = disk_scores(exp, model_name)
        per = [float(roc_auc_score(f.astype(int), s, max_fpr=PAUC_MAX_FPR))
               for s, f in months if 0 < f.sum() < len(f)]
        rows.append({"trial": n, "val_pauc": row["pauc"],
                     "test_pauc": float(np.mean(per)),
                     "test_sd": float(np.std(per, ddof=1))})
        print(f"  → test pAUC {np.mean(per):.4f}", flush=True)

    if not rows:
        print("\n채점된 시행이 없다.")
        return 1
    out = pd.DataFrame(rows).sort_values("test_pauc", ascending=False)
    path = ROOT / "results" / "xgb_trial_test_pauc.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n=== XGBoost 시행별 테스트 pAUC "
          f"({(time.time() - started) / 60:.0f}분) ===")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n비교 기준: XGBoost 기본 0.8637 / Optimized GRU 0.8792")
    print("주의: 이 표는 test 를 여러 번 본 결과다. 선정 근거로 쓸 수 없다.")
    print(f"[저장] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
