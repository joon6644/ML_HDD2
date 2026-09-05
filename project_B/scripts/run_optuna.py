"""XGBoost 하이퍼파라미터 탐색 — notion.md 4장 표의 [Proposed] 행.

    python scripts/run_optuna.py --trials 200
    python scripts/run_optuna.py --trials 200 --resume     # 중단된 study 이어서
    python scripts/run_optuna.py --report                 # 탐색 없이 결과만 출력

피처 구성은 [+ Feature] 행에서 확정된 것(ASFD 7일, 26피처)을 그대로 쓰고,
분할·판정·운영점(val 에서 잡는 디스크 FAR 1%)도 전부 고정한다. 바뀌는 것은
XGBoost 하이퍼파라미터뿐이다.

━━ 단일 시드로 탐색한다 ━━

논문에 싣는 모든 수치를 시드 42 하나로 통일했다. 반복 학습의 평균과 표준편차를
싣는 방식은 이 분야 선행연구에서 흔치 않고, 알려주는 바에 비해 표를 무겁게
만든다. 그래서 탐색도 시드 42 의 val 점수를 최대화하는 방식으로 맞춘다.

val 로 고르고 test 로 보고하는 분리는 그대로다. test 는 조합이 확정된 뒤
한 번만 본다.

━━ 목적함수: FAR 1% 이하 구간의 부분 AUC ━━

한 점의 Recall@FAR 1% 를 최대화하면 동점이 쏟아진다. val 고장 디스크가 23대라
Recall 의 눈금이 1/23(=0.0435)뿐이기 때문이다. 눈금이 굵으면 상위 시도 수십
개가 같은 값을 받고, 그중 무엇을 고를지는 결국 임의의 규칙이 정하게 된다.

그래서 점 하나가 아니라 구간을 본다. FAR 0 부터 1% 까지의 부분 ROC 면적
(partial AUC)을 목적함수로 쓴다.

  - 연속값이라 동점이 사실상 없다. 디스크 점수의 순서가 한 쌍만 바뀌어도
    값이 움직인다.
  - 보고하는 운영점(FAR 1%)과 같은 영역을 본다. 서론에서 밝힌 "낮은 오탐률
    에서의 고장 탐지 성능" 이라는 목표와도 일치한다.
  - 한 임곗값에 과적합되지 않는다. 하필 1% 지점만 좋은 조합이 아니라 저오탐
    구간 전체에서 좋은 조합이 뽑힌다.

Recall@FAR 1% 도 시도마다 같이 기록해 두므로 나중에 대조할 수 있다. 전부
val 에서만 계산한다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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
# 논문의 모든 수치가 시드 42 다. 탐색도 같은 시드로 맞춘다.
SEED = 42
STUDY_NAME = "xgboost_asfd7_single42_pauc_cpu"
FAR_TARGET = 0.01
# 부분 AUC 를 읽을 FAR 상한. 보고 운영점과 같은 1% 로 둔다.
PAUC_MAX_FPR = 0.01
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


# 탐색과 최종 학습이 같은 값을 쓴다. configs/models/xgboost.yaml (Baseline /
# + Feature 행) 과도 같으므로 표의 세 행이 전부 한 경로에서 나온다.
#
# 예전에는 탐색만 GPU(device: cuda, n_jobs 12)로 돌려 5.7배 빠르게 훑고 최종
# 학습은 CPU 로 했다. 그러면 안 된다 — tree_method: hist 는 계산 경로마다
# 히스토그램 분할점이 달라져 같은 하이퍼파라미터가 다른 모델이 된다. 실측:
#
#   같은 조합, 같은 시드   GPU 0.7586 / CPU n_jobs=12 0.7390 / CPU n_jobs=8 0.7332
#   탐색 상위 8개의 폭     0.0100
#
# 경로 차이가 탐색이 가르려던 차이보다 크다. 게다가 GPU 는 가용 메모리에 따라
# 스케치 배치가 달라져서, 다른 작업이 GPU 를 함께 쓰면 같은 조합의 점수까지
# 바뀐다 (기록 0.7929 -> 재현 0.7586). CPU 는 n_jobs 를 고정하면 리덕션 순서가
# 고정되어 프로세스가 달라도 재현된다.
SEARCH_FIXED = {
    "n_estimators": 2000,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_jobs": 8,
}
# 확정된 조합을 저장할 때 쓰는 값. 위와 같아야 한다.
OUTPUT_FIXED = dict(SEARCH_FIXED)
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
    from sklearn.metrics import roc_auc_score

    actual = has_window.astype(int).to_numpy()
    return {
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "far": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp,
        "n_failed": tp + fn,
        "roc_auc": float(roc_auc_score(actual, rank.to_numpy())),
        # 목적함수. sklearn 은 McClish 보정본을 주는데, 원 면적의 단조변환이라
        # 순위를 매기는 용도로는 같다.
        "pauc": float(roc_auc_score(actual, rank.to_numpy(), max_fpr=PAUC_MAX_FPR)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="XGBoost 하이퍼파라미터 탐색")
    ap.add_argument("--trials", type=int, default=200)
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
            model = XGBoostModel(params, TRAINING, seed=SEED)
            model.fit(train, val)
            info = val_score(model, val)
            for key, value in info.items():
                trial.set_user_attr(key, value)
            # FAR 1% 이하 구간의 부분 AUC. 상세는 모듈 docstring 참고.
            return info["pauc"]

        # 시드가 하나라 중간 보고 지점이 없다. 가지치기는 쓰지 않는다.
        study.sampler = optuna.samplers.TPESampler(seed=0)
        done = len([t for t in study.trials if t.state.is_finished()])
        remaining = max(0, args.trials - done)
        print(f"[optuna] 완료 {done}개, 남은 시도 {remaining}개 (seed {SEED} 단일)",
              flush=True)
        for n in range(remaining):
            t0 = time.time()
            study.optimize(objective, n_trials=1, catch=(Exception,))
            last = study.trials[-1]
            if last.value is None:
                mark = "failed"
            else:
                mark = (f"pauc={last.user_attrs['pauc']:.4f} "
                        f"recall={last.user_attrs['recall']:.4f}")
            print(
                f"  [{done + n + 1:>3}/{args.trials}] {mark}  ({time.time() - t0:.0f}s)"
                f"  best={study.best_value:.4f}",
                flush=True,
            )

    # ---- 결과 정리: val 부분 AUC 가 가장 큰 조합 ----
    rows = []
    for t in study.trials:
        if t.value is None:
            continue
        rows.append(
            {
                "trial": t.number,
                "pauc": t.user_attrs.get("pauc"),
                "recall": t.user_attrs.get("recall"),
                "far": t.user_attrs.get("far"),
                "roc_auc": t.user_attrs.get("roc_auc"),
                "tp": t.user_attrs.get("tp"),
                "n_failed": t.user_attrs.get("n_failed"),
                **t.params,
            }
        )
    if not rows:
        print("완료된 시도가 없다.")
        return 1

    frame = pd.DataFrame(rows).sort_values("pauc", ascending=False)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULT_DIR / "optuna_trials.csv", index=False, encoding="utf-8-sig")

    chosen = frame.iloc[0]
    n_tied = int((frame["pauc"] == chosen["pauc"]).sum())
    print(f"\n--- val 상위 8개 (seed {SEED}, 고장 {int(chosen['n_failed'])}대) ---")
    print(frame[["trial", "pauc", "recall", "far", "roc_auc", "tp"]].head(8)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n선택: trial {int(chosen['trial'])} "
          f"(val 부분 AUC {chosen['pauc']:.4f} 최대, 동점 {n_tied}개)")

    param_keys = [c for c in frame.columns if c not in
                  ("trial", "pauc", "recall", "far", "roc_auc", "tp", "n_failed")]
    params = {}
    for key in param_keys:
        value = chosen[key]
        params[key] = int(value) if key == "max_depth" else float(value)
    for key, value in params.items():
        print(f"  {key}: {value}")

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
        f"#   탐색: 시드 {SEED} 의 val 디스크 부분 AUC(FAR <= {PAUC_MAX_FPR:.0%}) 최대화\n"
        f"#   같은 값을 받은 시도: {n_tied}개\n"
        "#   test 는 탐색에 쓰지 않았다. 확정 후 따로 채점한다.\n"
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
