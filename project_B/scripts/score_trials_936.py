"""9-3-6 GRU 탐색의 개별 시행을 테스트 구간에서 채점한다.

    python scripts/score_trials_936.py --trials 0,1,2,3,4,10,11,39

탐색은 val pAUC 로 한 시행만 고른다. 그런데 9-3-6 브랜치에서 그렇게 고른
시행(trial 6)의 테스트 pAUC 가 0.8671 로 LSTM(0.8756)에 못 미쳤다. 그래서
"val 이 아니라 test 로 줄 세우면 이기는 시행이 있는가" 를 확인한다.

주의: 이 표는 **선정 근거로 쓸 수 없다.** test 를 여러 번 들여다본 결과라
그중 최댓값은 "이 탐색 공간에서 도달 가능한 상한" 이지 held-out 성능이
아니다. rank_trials_on_test.py 의 경고와 같은 취지다.

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
TRIALS_CSV = ROOT / "results" / "optuna_gru_936_trials.csv"
BASE_EXPERIMENT = "tos936_nn"          # 9-3-6 분할·피처를 가져온다
PAUC_MAX_FPR = 0.05

# run_optuna_rnn.FIXED_TRAINING 과 동일해야 탐색 조건이 재현된다.
FIXED_TRAINING = {
    "epochs": 30,
    "optimizer": "adamw",
    "loss": "bce",
    "auto_pos_weight": False,
    "early_stopping_patience": 5,
    "early_stopping_metric": "val_pauc",
    "grad_clip": 1.0,
    "num_workers": 0,
    "amp": True,
}


def write_configs(row) -> tuple[str, str]:
    """시행 하나의 모델·실험 설정을 쓴다. (실험 이름, 모델 이름)."""
    n = int(row.trial)
    model_name = f"gru_936_t{n}"
    layers = int(row.num_layers)
    dropout = 0.0 if layers == 1 or pd.isna(row.dropout) else float(row.dropout)
    model = {
        "name": model_name,
        "family": "sequence",
        "class": "hddpred.models.sequence.RNNModel",
        "params": {
            "cell": "gru",
            "hidden_size": int(row.hidden_size),
            "num_layers": layers,
            "dropout": dropout,
            "bidirectional": False,
        },
        "training": {
            **FIXED_TRAINING,
            "learning_rate": float(row.learning_rate),
            "weight_decay": float(row.weight_decay),
            "batch_size": int(row.batch_size),
        },
    }
    header = (
        f"# 9-3-6 GRU 탐색 trial {n} 의 하이퍼파라미터 (val pAUC {row.pauc:.4f}).\n"
        "# 탐색이 고른 시행이 아니라, test 로 줄 세워 보기 위한 재현용이다.\n"
        "# scripts/score_trials_936.py 가 생성한다.\n\n"
    )
    (ROOT / "configs" / "models" / f"{model_name}.yaml").write_text(
        header + yaml.safe_dump(model, allow_unicode=True, sort_keys=False),
        encoding="utf-8")

    base = yaml.safe_load(
        "experiment:" + (ROOT / "configs" / "experiments" / f"{BASE_EXPERIMENT}.yaml")
        .read_text(encoding="utf-8").split("experiment:", 1)[1])
    exp_name = f"tos936_t{n}"
    base["experiment"] = exp_name
    base["models"] = [f"configs/models/{model_name}.yaml"]
    o = base["overrides"]
    for k in [k for k in o if k.startswith("models.")]:
        del o[k]
    o[f"models.{model_name}.training.loss"] = "bce"
    o[f"models.{model_name}.training.auto_pos_weight"] = False
    (ROOT / "configs" / "experiments" / f"{exp_name}.yaml").write_text(
        header + yaml.safe_dump(base, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    return exp_name, model_name


def main() -> int:
    ap = argparse.ArgumentParser(description="9-3-6 GRU 시행별 테스트 채점")
    ap.add_argument("--trials", default="0,1,2,3,4,10,11,39")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    frame = pd.read_csv(TRIALS_CSV).set_index("trial")
    want = [int(x) for x in args.trials.split(",") if x.strip()]

    from run_curve import disk_scores   # 학습 뒤에 임포트해도 되지만 경로만 확인

    rows, started = [], time.time()
    for i, n in enumerate(want, 1):
        if n not in frame.index:
            print(f"[건너뜀] trial {n}: 완주 기록 없음", flush=True)
            continue
        r = frame.loc[n]
        # write_configs 가 속성 접근을 쓰므로 얇은 래퍼로 감싼다.
        row = type("Row", (), {"trial": n, **{k: r[k] for k in frame.columns}})()
        exp, model_name = write_configs(row)
        print(f"\n{'='*66}\n[{i}/{len(want)}] trial {n}  "
              f"(val pAUC {r.pauc:.4f}, hidden {int(r.hidden_size)}, "
              f"layers {int(r.num_layers)}, batch {int(r.batch_size)})\n{'='*66}",
              flush=True)
        if not args.skip_train:
            cmd = [PY, "scripts/run_experiment.py",
                   f"configs/experiments/{exp}.yaml"]
            print(f"$ {' '.join(cmd[1:])}", flush=True)
            subprocess.run(cmd, cwd=ROOT)   # 종료 코드는 무시한다 (CUDA 종료 잡음)
        metrics = ROOT / "runs" / exp / "TOSHIBA_20MG07ACA14TA" / model_name \
            / "seed42" / "fold00" / "fold_metrics.json"
        if not metrics.exists():
            print(f"  → 학습 산출물 없음, 건너뜀", flush=True)
            continue
        months = disk_scores(exp, model_name)
        per = [float(roc_auc_score(f.astype(int), s, max_fpr=PAUC_MAX_FPR))
               for s, f in months if 0 < f.sum() < len(f)]
        rows.append({"trial": n, "val_pauc": float(r.pauc),
                     "test_pauc": float(np.mean(per)),
                     "test_sd": float(np.std(per, ddof=1)),
                     **{f"m{j+1}": v for j, v in enumerate(per)}})
        print(f"  → test pAUC {np.mean(per):.4f}", flush=True)

    if not rows:
        print("\n채점된 시행이 없다.")
        return 1
    out = pd.DataFrame(rows).sort_values("test_pauc", ascending=False)
    path = ROOT / "results" / "gru936_trial_test_pauc.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n=== 9-3-6 GRU 시행별 테스트 pAUC ({(time.time()-started)/3600:.1f}시간) ===")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n비교 기준: LSTM 0.8756 / Optimized GRU(trial 6) 0.8671")
    print("⚠ 이 표는 test 를 여러 번 본 결과다. 선정 근거로 쓸 수 없다.")
    print(f"[저장] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
