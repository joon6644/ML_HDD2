"""비시퀀스 모델 4종의 하이퍼파라미터 탐색 — GRU와 같은 절차로.

    python scripts/run_optuna_tabular.py --model xgboost --trials 50
    python scripts/run_optuna_tabular.py --model randomforest --trials 50
    python scripts/run_optuna_tabular.py --report --model xgboost

run_optuna_rnn.py 를 그대로 본뜬다. 바뀌는 것은 탐색 공간뿐이고, 목적함수
(검증 월별 pAUC@FAR<=5% 의 평균), 분할, 시드, 샘플러(TPE seed 0), 시행 수는
같다. 논문 4.1 의 "동일한 절차" 가 말 그대로 동일해야 비교가 성립한다.

━━ 왜 필요한가 ━━

표 1 의 비시퀀스 4종은 기본 하이퍼파라미터다. 그 상태로 GRU(탐색 후)와
나란히 놓으면 "탐색을 한 쪽이 이겼다" 는 당연한 말밖에 안 된다. 같은 예산
(50시행)을 네 모델에도 주고 나서야 저오탐률 구간 목적함수가 모델 계열과
무관하게 작동하는지, 그리고 GRU 의 우위가 탐색 유무 때문인지가 갈린다.

━━ 가지치기 ━━

트리 계열은 에폭 개념이 없어 중간 보고를 할 수 없으므로 가지치기하지 않는다.
MLP 만 RNN 판과 같은 PercentilePruner 를 쓴다. 시행 수가 같으므로 예산의
공정성은 유지된다.
"""

from __future__ import annotations

import argparse
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

from hddpred import config as cfg_mod  # noqa: E402
from hddpred import paths  # noqa: E402
from hddpred.experiments.runner import Pipeline, prepare_drive, _load_part  # noqa: E402
from hddpred.models.mlp import MLPModel  # noqa: E402
from hddpred.models.trees import (  # noqa: E402
    LightGBMModel, RandomForestModel, XGBoostModel,
)
from export_results import disk_rank, load_part, month_windows  # noqa: E402

DRIVE = "TOSHIBA_20MG07ACA14TA"
BASE_EXPERIMENT = "tos_select_pauc"   # 10-2-6, 표 1 과 같은 분할·피처
SEED = 42
PAUC_MAX_FPR = 0.05
STUDY_NAME_FMT = "{model}_tos_win10_pauc5_mon{tag}"
STUDY_DB = ROOT / "runs" / "optuna" / "tabular.db"
RESULT_DIR = ROOT / "results"

MODELS = {
    "randomforest": RandomForestModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "mlp": MLPModel,
}

# 탐색 대상 밖의 고정값. 불균형 처리를 하지 않는다는 확정 사항이 여기 박혀 있다.
FIXED = {
    "randomforest": ({"n_jobs": 8, "verbose": 0}, {"auto_class_weight": False}),
    "xgboost": (
        {"objective": "binary:logistic", "eval_metric": "aucpr",
         "tree_method": "hist", "n_jobs": 8},
        {"early_stopping_rounds": 100, "auto_scale_pos_weight": False},
    ),
    "lightgbm": (
        {"objective": "binary", "metric": "average_precision",
         "n_jobs": 8, "verbose": -1},
        {"early_stopping_rounds": 100, "auto_scale_pos_weight": False},
    ),
    "mlp": (
        {"batch_norm": True},
        {"epochs": 30, "loss": "bce", "auto_pos_weight": False,
         "early_stopping_patience": 5, "early_stopping_metric": "val_pauc",
         "scaling": {"method": "standard", "clip_quantile": [0.001, 0.999]}},
    ),
}


def suggest(trial, name: str) -> tuple[dict, dict]:
    """탐색 공간. 축 수를 GRU(6개)와 비슷하게 맞춘다."""
    params, training = ({**FIXED[name][0]}, {**FIXED[name][1]})
    if name == "randomforest":
        params.update(
            n_estimators=trial.suggest_categorical("n_estimators", [200, 400, 800]),
            max_depth=trial.suggest_categorical("max_depth", [None, 8, 12, 20]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 40, log=True),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            max_samples=trial.suggest_float("max_samples", 0.5, 1.0),
        )
        params["bootstrap"] = True          # max_samples 는 bootstrap 일 때만 쓰인다
    elif name == "xgboost":
        params.update(
            n_estimators=2000,              # 조기종료가 실제 개수를 정한다
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 50, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    elif name == "lightgbm":
        params.update(
            n_estimators=3000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        params["subsample_freq"] = 1        # subsample 은 freq>0 일 때만 걸린다
    else:                                   # mlp
        # CSV 를 거쳐 재생될 때 float 로 돌아오므로 명시적으로 정수화한다.
        width = int(trial.suggest_categorical("width", [64, 128, 256, 512]))
        depth = trial.suggest_int("depth", 2, 4)
        # 층마다 절반씩 줄이는 피라미드. GRU 의 hidden_size x num_layers 와 대응한다.
        params["hidden_dims"] = [max(16, width >> i) for i in range(depth)]
        params["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
        training.update(
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [1024, 2048, 4096]),
        )
    return params, training


def val_pauc(model, parts, aggregate: str = "mean") -> tuple[float, float]:
    """검증 구간의 부분 AUC와 FAR 1% 재현율. 창마다 산출한 뒤 평균한다.

    run_optuna_rnn.val_pauc 와 같은 함수다. 풀링하지 않는다 (3.4 집계 규칙).
    """
    paucs, recalls, ranks, flags = [], [], [], []
    for part in parts:
        rank, has_window = disk_rank(model, part, "in_horizon")
        score = rank.to_numpy()
        actual = has_window.to_numpy().astype(int)
        n_pos = int(actual.sum())
        if n_pos == 0 or n_pos == len(actual):
            continue
        if aggregate == "pool":
            ranks.append(score)
            flags.append(actual)
            continue
        paucs.append(float(roc_auc_score(actual, score, max_fpr=PAUC_MAX_FPR)))
        thr = float(np.quantile(score[actual == 0], 0.99))
        recalls.append(float(((score >= thr) & (actual == 1)).sum() / n_pos))
    if aggregate == "pool":
        if not ranks:
            return float("nan"), float("nan")
        score = np.concatenate(ranks)
        actual = np.concatenate(flags)
        pauc = float(roc_auc_score(actual, score, max_fpr=PAUC_MAX_FPR))
        thr = float(np.quantile(score[actual == 0], 0.99))
        rec = float(((score >= thr) & (actual == 1)).sum() / int(actual.sum()))
        return pauc, rec
    if not paucs:
        return float("nan"), float("nan")
    return float(np.mean(paucs)), float(np.mean(recalls))


def write_config(name: str, frame, tag: str) -> Path:
    """1위 시행을 모델 yaml 로 굳힌다. run_experiment.py 가 바로 읽는다."""
    best = frame.iloc[0]
    trial_params = {k: best[k] for k in frame.columns
                    if k not in ("trial", "pauc", "recall01")and pd.notna(best[k])}

    class _T:                              # suggest 를 재생하기 위한 최소 stub
        def __init__(self, p): self.p = p
        def suggest_categorical(self, k, _): return self._get(k)
        def suggest_int(self, k, *a, **kw): return int(self._get(k))
        def suggest_float(self, k, *a, **kw): return float(self._get(k))

        def _get(self, k):
            """numpy 스칼라를 파이썬 기본형으로 되돌린다.

            값이 DataFrame 을 거쳐 오므로 np.int64/np.float64 가 섞인다.
            yaml.safe_dump 는 그것을 직렬화하지 못한다.
            """
            v = self.p[k]
            return v.item() if hasattr(v, "item") else v

    params, training = suggest(_T(trial_params), name)
    cls = MODELS[name]
    out = {
        "name": f"{name}_pauc_tuned{tag}",
        "family": "tabular",
        "class": f"{cls.__module__}.{cls.__name__}",
        "params": params,
        "training": training,
    }
    path = ROOT / "configs" / "models" / f"{name}_pauc_tuned{tag}.yaml"
    path.write_text(
        f"# {name} 탐색 1위 (val pAUC@FAR<=5% {best.pauc:.4f}, trial {int(best.trial)}).\n"
        "# scripts/run_optuna_tabular.py 가 생성한다. 손으로 고치지 말 것.\n\n"
        + yaml.safe_dump(out, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="비시퀀스 모델 하이퍼파라미터 탐색")
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--base-experiment", default=BASE_EXPERIMENT)
    ap.add_argument("--tag", default="")
    ap.add_argument("--val-aggregate", default="mean", choices=["mean", "pool"],
                    help="검증 목적함수의 집계 방식 (보고 지표와 무관)")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    name = args.model
    study_name = STUDY_NAME_FMT.format(model=name, tag=tag)

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    STUDY_DB.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{STUDY_DB.as_posix()}",
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0),
        # 트리 계열은 중간 보고가 없어 가지치기가 걸리지 않는다.
        pruner=optuna.pruners.PercentilePruner(25.0, n_startup_trials=5,
                                               n_warmup_steps=5),
    )

    if not args.report:
        cfg = cfg_mod.load_yaml(
            paths.CONFIG_DIR / "experiments" / f"{args.base_experiment}.yaml")
        pipeline = Pipeline.from_experiment(cfg)
        prepared = prepare_drive(DRIVE, pipeline)
        fold = sorted(prepared.folds, key=lambda f: f.fold)[0]
        threads = int(pipeline.preprocessing.get("duckdb", {}).get("threads", 8))
        horizon = int(pipeline.labeling["horizon_days"])

        print("[optuna] 데이터 적재 (한 번만)", flush=True)
        started = time.time()
        train = _load_part(prepared, fold, "train", "tabular",
                           pipeline.features, pipeline.labeling, threads)
        val_full = _load_part(prepared, fold, "val", "tabular",
                              pipeline.features, pipeline.labeling, threads)
        # 트리·MLP 모두 스케일러를 모델이 직접 들고 있으므로 외부 변환이 없다.
        val_months = [
            load_part(prepared, pipeline, "tabular", a, b, horizon, threads)
            for a, b in month_windows(*fold.window("val"))
        ]
        print(f"  train {len(train):,} | val {len(val_full):,} "
              f"| val 월별 {len(val_months)}창 ({time.time() - started:.1f}s)",
              flush=True)

        def objective(trial):
            params, training = suggest(trial, name)
            model = MODELS[name](params, training, seed=SEED)
            if name == "mlp":
                def on_epoch(epoch, best):
                    trial.report(best, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                model.epoch_callback = on_epoch
            model.fit(train, val_full)
            if name == "mlp":
                model.epoch_callback = None
            score, recall01 = val_pauc(model, val_months, args.val_aggregate)
            trial.set_user_attr("pauc", score)
            trial.set_user_attr("recall01", recall01)
            return score

        done = len([t for t in study.trials if t.state.is_finished()])
        remaining = max(0, args.trials - done)
        print(f"[optuna] {name}: 완료 {done}개, 남은 시도 {remaining}개 "
              f"(seed {SEED})", flush=True)
        for n in range(remaining):
            t0 = time.time()
            study.optimize(objective, n_trials=1, catch=(Exception,))
            last = study.trials[-1]
            mark = ("pruned" if str(last.state) == "TrialState.PRUNED"
                    else f"pAUC={last.value:.4f}" if last.value is not None
                    else str(last.state))
            best = study.best_value if any(
                t.value is not None for t in study.trials) else float("nan")
            print(f"  [{done + n + 1:>3}/{args.trials}] {mark}  "
                  f"({time.time() - t0:.0f}s)  best={best:.4f}", flush=True)

    rows = [
        {"trial": t.number, "pauc": t.value,
         "recall01": t.user_attrs.get("recall01"), **t.params}
        for t in study.trials
        if str(t.state) == "TrialState.COMPLETE" and t.value is not None
    ]
    if not rows:
        print("완료된 시행이 없다.")
        return 1
    frame = pd.DataFrame(rows).sort_values("pauc", ascending=False)
    csv = RESULT_DIR / f"optuna_{name}{tag}_trials.csv"
    frame.to_csv(csv, index=False, encoding="utf-8-sig")
    cfg_path = write_config(name, frame, tag)
    print(f"\n=== {name} 탐색 결과 (완료 {len(frame)}시행) ===")
    print(frame.head(10).to_string(index=False,
                                   float_format=lambda v: f"{v:.4f}"))
    print(f"\n[저장] {csv}\n[저장] {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
