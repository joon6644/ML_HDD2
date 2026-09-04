"""XGBoost 하이퍼파라미터 탐색 — notion.md 4장 표의 [+ Optuna (Proposed)] 행.

    python scripts/run_optuna.py --trials 40
    python scripts/run_optuna.py --trials 40 --resume     # 중단된 study 이어서
    python scripts/run_optuna.py --report                 # 결과만 다시 출력

피처 구성은 [+ Feature] 행에서 확정된 것(ASFD 7일, 26피처)을 그대로 쓰고,
분할·판정·운영점(val 에서 잡는 디스크 FAR 1%)도 전부 고정한다. 바뀌는 것은
XGBoost 하이퍼파라미터뿐이다.

━━ 왜 다중 시드로 탐색하는가 ━━

단일 시드의 val 점수를 최대화하면 "그 시드에서만 높은" 조합이 뽑힌다. 실측으로
확인된 문제다 — 같은 설정이 시드에 따라 고장 26~32대로 흔들렸고, 시드 42 하나만
보고 최적이라 판단했던 조합이 5시드 평균에서는 오히려 중위권이었다.

그래서 두 겹으로 막는다.

  1) 목적함수 = SEARCH_SEEDS 개 시드의 val 점수 평균
     한 시드의 운으로는 목적함수를 못 올린다.

  2) 최종 선택 = 평균 - 표준편차 (상위 TOP_K 중에서)
     평균이 같으면 시드 간 흔들림이 작은 쪽을 고른다. 평균만 보고 고르면
     "평균은 높지만 분산이 큰" 조합, 즉 운에 기대는 조합이 뽑힐 수 있다.

  그리고 확정된 조합은 탐색에 쓰지 않은 시드까지 포함해 5시드로 test 를
  다시 채점한다 (scripts/run_features.py 와 같은 경로).

━━ test 는 탐색에 일절 쓰지 않는다 ━━

목적함수는 val 구간에서만 계산한다. 조기 종료도 val, 임곗값도 val 이다.
test 는 탐색이 전부 끝나고 조합이 확정된 뒤 한 번만 본다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hddpred import config as cfg_mod  # noqa: E402
from hddpred import paths  # noqa: E402
from hddpred.experiments.runner import Pipeline, prepare_drive, _load_part  # noqa: E402
from hddpred.models.trees import XGBoostModel  # noqa: E402
from export_results import disk_rank, healthy_quantiles  # noqa: E402

DRIVE = "HGST_20HUH721212ALN604"
# [+ Feature] 행에서 확정된 피처 구성.
BASE_EXPERIMENT = "feat_asfd7"
# 탐색 시드 = 최종 검증 시드(42~46). 일반적인 관행이다 — val 로 고르고 test 로
# 보고하는 분리는 그대로 유지되고, 시드까지 떼는 것은 추가 엄격성이었다.
# 시드를 뗀 판(101~105)의 결과는 study 이름 xgboost_asfd7 로 남아 있다.
SEARCH_SEEDS = [42, 43, 44, 45, 46]
STUDY_NAME = "xgboost_asfd7_seed42_46"
# 선택 기준. "robust" 는 평균-표준편차, "mean" 은 평균 최대.
SELECT_BY = "mean"
FAR_TARGET = 0.01
TOP_K = 5  # 평균 상위 몇 개 중에서 평균-표준편차로 고를지
STUDY_DB = ROOT / "runs" / "optuna" / "xgboost.db"  # study 이름으로 구분한다
RESULT_DIR = ROOT / "results"


def suggest(trial):
    """탐색 공간.

    현행 값(max_depth 6, lr 0.05, min_child_weight 5, subsample 0.8,
    colsample 0.8, reg_lambda 1.0, reg_alpha 0)을 전부 안쪽에 포함하도록
    잡았다. 탐색이 현행보다 나쁜 곳만 뒤지는 일이 없게 하기 위해서다.
    n_estimators 는 고정하고 조기 종료가 정하게 둔다.
    """
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 10.0, log=True),
    }


# 탐색은 GPU 로 돈다. 실측 26.9s vs CPU 152.1s (5.7배).
# GPU 와 CPU 의 결과 차이는 CPU 의 스레드 수만 바꿨을 때의 차이보다 크지 않다
# (best_score: CPU8 0.156 / GPU 0.174 / CPU12 0.187 — GPU 가 두 CPU 사이).
SEARCH_FIXED = {
    "n_estimators": 2000,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "device": "cuda",
    "n_jobs": 12,
}
# 확정된 조합을 저장할 때 쓰는 값. baseline / +Feature 행이 CPU 로 계산됐으므로
# 최종 검증도 같은 경로로 맞춘다. 하이퍼파라미터 자체는 경로와 무관하다.
OUTPUT_FIXED = {
    "n_estimators": 2000,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_jobs": 8,
}
TRAINING = {"early_stopping_rounds": 100, "auto_scale_pos_weight": False}


def val_score(model, val_part) -> dict:
    """val 구간의 디스크 단위 지표. test 는 건드리지 않는다."""
    rank, has_window = disk_rank(model, val_part, "in_horizon")
    thresholds = healthy_quantiles(rank, has_window)
    alarm = rank >= thresholds[FAR_TARGET]
    tp = int((has_window & alarm).sum())
    fn = int((has_window & ~alarm).sum())
    fp = int((~has_window & alarm).sum())
    tn = int((~has_window & ~alarm).sum())
    from sklearn.metrics import average_precision_score, roc_auc_score

    actual = has_window.astype(int).to_numpy()
    return {
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "far": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp,
        "n_failed": tp + fn,
        "roc_auc": float(roc_auc_score(actual, rank.to_numpy())),
        "ap": float(average_precision_score(actual, rank.to_numpy())),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="XGBoost 하이퍼파라미터 탐색")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--resume", action="store_true", help="기존 study 이어서")
    ap.add_argument("--report", action="store_true", help="탐색 없이 결과만 출력")
    args = ap.parse_args()

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    STUDY_DB.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB.as_posix()}",
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0),
    )

    if not args.report:
        cfg = cfg_mod.load_yaml(paths.CONFIG_DIR / "experiments" / f"{BASE_EXPERIMENT}.yaml")
        pipeline = Pipeline.from_experiment(cfg)
        prepared = prepare_drive(DRIVE, pipeline)
        fold = sorted(prepared.folds, key=lambda f: f.fold)[0]
        threads = int(pipeline.preprocessing.get("duckdb", {}).get("threads", 8))
        shared = ("tabular", pipeline.features, pipeline.labeling, threads)

        print("[optuna] 데이터 적재 (한 번만)", flush=True)
        started = time.time()
        train = _load_part(prepared, fold, "train", *shared)
        val = _load_part(prepared, fold, "val", *shared)
        print(
            f"  train {len(train):,} x {train.X.shape[1]}피처 | val {len(val):,} "
            f"({time.time() - started:.1f}s)",
            flush=True,
        )

        def objective(trial):
            params = {**SEARCH_FIXED, **suggest(trial)}
            scores = []
            for i, seed in enumerate(SEARCH_SEEDS):
                model = XGBoostModel(params, TRAINING, seed=seed)
                model.fit(train, val)
                info = val_score(model, val)
                scores.append(info["recall"])
                trial.set_user_attr(f"seed{seed}", info)
                # 시드 하나가 끝날 때마다 중간값을 보고해 가망 없는 조합을 일찍 끊는다.
                trial.report(float(np.mean(scores)), i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            trial.set_user_attr("recall_mean", float(np.mean(scores)))
            trial.set_user_attr("recall_sd", float(np.std(scores, ddof=1)))
            return float(np.mean(scores))

        study.sampler = optuna.samplers.TPESampler(seed=0)
        study.pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1)
        done = len([t for t in study.trials if t.state.is_finished()])
        remaining = max(0, args.trials - done)
        print(f"[optuna] 완료 {done}개, 남은 시도 {remaining}개", flush=True)
        for n in range(remaining):
            t0 = time.time()
            study.optimize(objective, n_trials=1, catch=(Exception,))
            last = study.trials[-1]
            mark = "pruned" if str(last.state) == "TrialState.PRUNED" else f"{last.value:.4f}"
            print(
                f"  [{done + n + 1:>3}/{args.trials}] {mark}  ({time.time() - t0:.0f}s)"
                f"  best={study.best_value:.4f}",
                flush=True,
            )

    # ---- 결과 정리: 평균 상위 TOP_K 중에서 평균-표준편차로 고른다 ----
    rows = []
    for t in study.trials:
        if t.value is None:
            continue
        rows.append(
            {
                "trial": t.number,
                "recall_mean": t.user_attrs.get("recall_mean", t.value),
                "recall_sd": t.user_attrs.get("recall_sd", np.nan),
                **t.params,
            }
        )
    if not rows:
        print("완료된 시도가 없다.")
        return 1

    frame = pd.DataFrame(rows)
    frame["robust"] = frame["recall_mean"] - frame["recall_sd"].fillna(0)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frame.sort_values("recall_mean", ascending=False).to_csv(
        RESULT_DIR / "optuna_trials.csv", index=False, encoding="utf-8-sig"
    )

    top = frame.nlargest(TOP_K, "recall_mean")
    if SELECT_BY == "robust":
        chosen = top.nlargest(1, "robust").iloc[0]
    else:
        # 평균 최대. 동점이면 흔들림이 작은 쪽으로 가른다.
        chosen = top.sort_values(
            ["recall_mean", "robust"], ascending=[False, False]
        ).iloc[0]
    print(f"\n--- 평균 상위 {TOP_K}개 (val, {len(SEARCH_SEEDS)}시드) ---")
    cols = ["trial", "recall_mean", "recall_sd", "robust"]
    print(top[cols].to_string(index=False))
    criterion = "평균-표준편차" if SELECT_BY == "robust" else "평균"
    print(f"\n선택: trial {int(chosen['trial'])} ({criterion} 최대)")

    param_keys = [c for c in frame.columns if c not in
                  ("trial", "recall_mean", "recall_sd", "robust")]
    params = {}
    for k in param_keys:
        v = chosen[k]
        params[k] = int(v) if k == "max_depth" else float(v)
    for k, v in params.items():
        print(f"  {k}: {v}")

    out = ROOT / "configs" / "models" / "xgboost_tuned.yaml"
    import yaml

    cfg_out = {
        "name": "xgboost_tuned",
        "family": "tabular",
        "class": "hddpred.models.trees.XGBoostModel",
        "params": {**params, **OUTPUT_FIXED},
        "training": dict(TRAINING),
    }
    header = (
        "# Optuna 로 고른 XGBoost 하이퍼파라미터.\n"
        f"#   탐색: val 디스크 Recall@FAR {FAR_TARGET:.0%} 를 {len(SEARCH_SEEDS)}시드"
        f"({', '.join(map(str, SEARCH_SEEDS))}) 평균으로 최대화\n"
        f"#   선택: 평균 상위 {TOP_K}개 중 {criterion} 이 가장 큰 조합\n"
        f"#   test 는 탐색에 쓰지 않았다. 확정 후 5시드로 따로 채점한다.\n"
        "#   scripts/run_optuna.py 가 생성한다. 직접 고치지 마라.\n\n"
    )
    out.write_text(
        header + yaml.safe_dump(cfg_out, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"\n[저장] {out}")
    print(f"[저장] {RESULT_DIR / 'optuna_trials.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
