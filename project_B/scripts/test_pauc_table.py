"""테스트 구간 pAUC 순위표 — 한 브랜치의 후보들을 한 줄로 세운다.

    python scripts/test_pauc_table.py --arms 936
    python scripts/test_pauc_table.py --arms 1026

논문 표 1 의 pAUC 열에 해당하는 값을 학습 없이 다시 뽑는다. 저장된 모델로
테스트 6개월을 채점하고, 달마다 디스크 단위 pAUC@FAR<=5% 를 낸 뒤 평균한다
(3.4 의 집계 규칙 — 풀링하지 않는다).

run_curve.disk_scores 를 그대로 재사용하므로 채점 경로가 곡선 그림과 같다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_curve import disk_scores  # noqa: E402

PAUC_MAX_FPR = 0.05

# (표기, experiment, model). 표기는 논문 표 1 의 Model 열과 맞춘다.
ARM_SETS = {
    "936": [
        ("RandomForest",  "tos936_trees",    "randomforest"),
        ("XGBoost",       "tos936_trees",    "xgboost"),
        ("LightGBM",      "tos936_trees",    "lightgbm"),
        ("MLP",           "tos936_nn",       "mlp_pauc"),
        ("LSTM",          "tos936_nn",       "lstm_pauc"),
        ("GRU",           "tos936_nn",       "gru_pauc"),
        ("TCN",           "tos936_nn",       "tcn_pauc"),
        ("Optimized GRU", "tos936_proposed", "gru_tuned_936"),
    ],
    "1026": [
        ("RandomForest",  "tos_select",      "randomforest"),
        ("XGBoost",       "tos_select",      "xgboost"),
        ("LightGBM",      "tos_select",      "lightgbm"),
        ("MLP",           "tos_select_pauc", "mlp_pauc"),
        ("LSTM",          "tos_select_pauc", "lstm_pauc"),
        ("GRU",           "tos_select_pauc", "gru_pauc"),
        ("TCN",           "tos_select_pauc", "tcn_pauc"),
        ("Optimized GRU", "tos_proposed",    "gru_tuned"),
        ("Optimized TCN", "tos_proposed_tcn", "tcn_tuned"),
    ],
}


def month_pauc(months) -> list[float]:
    """달마다 디스크 단위 pAUC. 한쪽 클래스뿐인 달은 건너뛴다."""
    out = []
    for score, failed in months:
        y = failed.astype(int)
        if 0 < y.sum() < len(y):
            out.append(float(roc_auc_score(y, score, max_fpr=PAUC_MAX_FPR)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="테스트 pAUC 순위표")
    ap.add_argument("--arms", choices=sorted(ARM_SETS), default="936")
    ap.add_argument("--out", default=None, help="CSV 저장 경로")
    args = ap.parse_args()

    rows = []
    for label, experiment, model_name in ARM_SETS[args.arms]:
        run = ROOT / "runs" / experiment
        if not run.exists():
            print(f"[건너뜀] {label}: {run} 없음", flush=True)
            continue
        print(f"[채점] {label} ({experiment} / {model_name})", flush=True)
        months = disk_scores(experiment, model_name)
        per = month_pauc(months)
        rows.append({
            "model": label,
            "pauc": float(np.mean(per)),
            "pauc_sd": float(np.std(per, ddof=1)) if len(per) > 1 else 0.0,
            "n_months": len(per),
            "n_failed": sum(int(f.sum()) for _, f in months),
            **{f"m{i+1}": v for i, v in enumerate(per)},
        })

    frame = pd.DataFrame(rows).sort_values("pauc", ascending=False)
    out = Path(args.out or ROOT / "results" / f"test_pauc_{args.arms}.csv")
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n=== 테스트 pAUC@FAR<={PAUC_MAX_FPR:.0%} (달별 산출 후 평균) ===")
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n[저장] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
