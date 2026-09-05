"""후보 모델 비교 — notion.md 4.1 의 표.

    python scripts/run_selection.py

원본 SMART 피처 18개만 쓰고 불균형 처리도 하지 않은 상태에서 후보 9종을
같은 조건으로 비교한다. 학습은 이미 끝나 있고(runs/baseline, baseline_seq14),
여기서는 저장된 모델로 다시 채점만 한다.

━━ 무엇을 기준으로 고르는가 ━━

선정은 val 에서 한다. test 는 선정이 끝난 뒤 성능을 보고하는 데만 쓴다.
val 기준은 FAR 0~1% 구간의 부분 AUC 로, 하이퍼파라미터 탐색(run_optuna.py)이
쓰는 것과 같은 기준이다. 논문이 관심 있는 구간이 저오탐 영역이기 때문이다.

전 구간 ROC-AUC 로 고르면 저오탐 구간에서의 우열이 묻힌다. 실제로 두 지표는
순위를 다르게 매긴다.

━━ 임곗값 ━━

검증셋에서 디스크 FAR 1% 가 되는 지점을 임곗값으로 잡고, test 세 달에 그대로
적용한다. test 에서 FAR 이 정확히 1% 로 떨어지지는 않으며, 그 값을 그대로
표에 싣는다.

━━ 시드 ━━

시드 42 하나다. 논문의 모든 수치와 같은 기준이다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hddpred import config as cfg_mod  # noqa: E402
from hddpred import paths  # noqa: E402
from hddpred.features import fold as fold_mod  # noqa: E402
from hddpred.experiments.runner import Pipeline, prepare_drive  # noqa: E402
from hddpred.models import registry  # noqa: E402
from export_results import disk_rank, load_part, rescale  # noqa: E402

DRIVE = "HGST_20HUH721212ALN604"
SEED = 42
RULE = "in_horizon"
FAR_TARGET = 0.01
PAUC_MAX_FPR = 0.01
# 표에 싣는 이름, 실험, 모델
MODELS = [
    ("RandomForest", "baseline", "randomforest"),
    ("XGBoost", "baseline", "xgboost"),
    ("LightGBM", "baseline", "lightgbm"),
    ("MLP", "baseline", "mlp"),
    ("LSTM", "baseline_seq14", "lstm"),
    ("GRU", "baseline_seq14", "gru"),
    ("CNN1D", "baseline_seq14", "cnn1d"),
    ("TCN", "baseline_seq14", "tcn"),
]


def evaluate(experiment: str, model_name: str) -> dict:
    cfg = cfg_mod.load_yaml(paths.CONFIG_DIR / "experiments" / f"{experiment}.yaml")
    pipeline = Pipeline.from_experiment(cfg)
    prepared = prepare_drive(DRIVE, pipeline)
    folds = sorted(prepared.folds, key=lambda f: f.fold)
    mcfg = next(
        m for m in (registry.load_model_config(x) for x in cfg["models"])
        if m["name"] == model_name
    )
    horizon = int(pipeline.labeling["horizon_days"])
    run = paths.run_dir(experiment, DRIVE, model_name, SEED, 0)
    model = registry.resolve_class(mcfg["class"]).load(run / "model")
    # 시퀀스 계열은 학습 때 fold 의 train 구간 스케일러를 거쳤다. 채점할 때도
    # 같은 스케일러를 먹여야 한다 — 빼먹으면 모델이 상수를 뱉는다.
    with (run / "fold_metrics.json").open(encoding="utf-8") as fh:
        record = json.load(fh)
    scaler = (fold_mod.Scaler.from_state_dict(record["scaler"])
              if record.get("scaler") else None)

    def part_of(split: str, start, end):
        part = load_part(prepared, pipeline, mcfg["family"], start, end, horizon, 8)
        return rescale(part, scaler) if mcfg["family"] == "sequence" else part

    # --- val: 선정 기준과 임곗값을 여기서 잡는다 ---
    vstart, vend = folds[0].window("val")
    val_rank, val_failed = disk_rank(model, part_of("val", vstart, vend), RULE)
    val_score = val_rank.to_numpy()
    val_failed = val_failed.to_numpy().astype(bool)
    threshold = float(np.quantile(val_score[~val_failed], 1.0 - FAR_TARGET))
    val_pauc = float(roc_auc_score(val_failed.astype(int), val_score,
                                   max_fpr=PAUC_MAX_FPR))

    # --- test: 모델 하나로 세 달을 각각 예측해 통합한다 ---
    scores, failed = [], []
    for fold in folds:
        start, end = fold.window("test")
        rank, has_window = disk_rank(model, part_of("test", start, end), RULE)
        scores.append(rank.to_numpy())
        failed.append(has_window.to_numpy())
    score = np.concatenate(scores)
    is_failed = np.concatenate(failed).astype(bool)

    alarm = score >= threshold
    return {
        "val_pAUC": val_pauc,
        "recall": float((alarm & is_failed).sum() / is_failed.sum()),
        "far": float((alarm & ~is_failed).sum() / (~is_failed).sum()),
        "roc_auc": float(roc_auc_score(is_failed.astype(int), score)),
        "pauc": float(roc_auc_score(is_failed.astype(int), score,
                                    max_fpr=PAUC_MAX_FPR)),
        "n_failed": int(is_failed.sum()),
        "n_healthy": int((~is_failed).sum()),
    }


def main() -> int:
    rows = []
    for label, experiment, model_name in MODELS:
        print(f"[selection] {label}", flush=True)
        info = evaluate(experiment, model_name)
        rows.append({"model": label, **info})
        print(f"  val pAUC {info['val_pAUC']:.4f} | test recall {info['recall']:.3f} "
              f"FAR {info['far']:.4f} pAUC {info['pauc']:.4f}", flush=True)

    frame = pd.DataFrame(rows).sort_values("val_pAUC", ascending=False)
    out = ROOT / "results" / "selection_summary.csv"
    frame.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n=== 후보 모델 비교 (seed {SEED}, test {frame.n_failed.iloc[0]}대 고장 "
          f"/ {frame.n_healthy.iloc[0]:,}대 정상) ===")
    print(f"{'모델':<14}{'val pAUC':>10}{'Recall':>9}{'FAR':>8}{'pAUC':>8}{'ROC-AUC':>9}")
    for _, r in frame.iterrows():
        print(f"{r.model:<14}{r.val_pAUC:>10.3f}{r.recall:>9.3f}"
              f"{r.far * 100:>7.2f}%{r.pauc:>8.3f}{r.roc_auc:>9.3f}")
    print(f"\n[저장] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
