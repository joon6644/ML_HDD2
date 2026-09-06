"""LSTM/GRU 하이퍼파라미터 탐색 (--cell 로 고른다).

    python scripts/run_optuna_rnn.py --trials 40 --cell lstm
    python scripts/run_optuna_rnn.py --report --cell lstm   # 재탐색 없이 집계만

이어 돌리기: study 가 SQLite 에 남으므로 --trials 를 늘리면 같은 study 를
그대로 이어간다. 40회를 돌고 결과가 무의미하면

    python scripts/run_optuna_rnn.py --cell lstm --trials 80

로 40회를 더 붙인다. TPE 는 앞선 40회를 전부 활용한다. 매 판의 결과와
"더 돌릴 가치가 있는가" 판정은 results/tuning_log.md 에 쌓인다.

순환 계열(LSTM/GRU)을 대상으로 한다. 분할·피처·정규화·불균형 처리는
전부 고정하고 신경망 하이퍼파라미터만 바꾼다.

  분할      도시바 10-2-6 (test 2025-10 ~ 2026-03)
  피처      원본 SMART 16개, lookback 14일 (민감도 스윕에서 확정)
  정규화    min-max (train 구간에서만 적합)
  불균형    처리 없음, 순수 BCE (12종 비교에서 무처리가 최선)
  시드      42 하나

━━ 목적함수: val 의 pAUC @ FAR <= 5% ━━

한 점의 Recall 을 최대화하면 이산값이라 동점이 쏟아진다. 부분 AUC 는 디스크
점수의 순서가 한 쌍만 바뀌어도 값이 움직이는 연속값이라 그 문제가 없다.
FAR 5% 를 상한으로 두는 것은 논문이 보고하는 운영점 격자(0.1/0.5/1/5%)를
모두 덮는 구간이기 때문이다.

val 점수는 **월별 창으로 쪼개서** 접는다. 디스크 점수가 창 안의 최댓값이라
창 길이가 다르면 점수 분포가 달라지는데, test 가 한 달씩 채점되므로 val 도
같은 단위여야 한다. export_results.month_windows 를 그대로 쓴다.

━━ 가지치기 ━━

GRU 한 번 학습이 실측 643초(batch 512)다. 끝까지 다 돌리면 몇 시간이 넘는다.
그래서 매 에폭의 val PR-AUC 를 optuna 에 보고해 가망 없는 조합을 일찍 끊는다
(TorchSequenceModel.epoch_callback). 조기 종료가 이미 걸려 있으므로 나쁜
조합은 원래도 일찍 끝나지만, 가지치기는 "다른 시도들에 비해" 나쁜 것까지
끊는다는 점이 다르다.

주의: 보고하는 값은 조기 종료 지표인 val PR-AUC 이고 목적함수는 pAUC 다.
둘은 다른 지표지만 가지치기는 순위만 쓰므로 상관이 있으면 충분하다. 최종
선택은 언제나 pAUC 로 한다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hddpred import config as cfg_mod  # noqa: E402
from hddpred import paths  # noqa: E402
from hddpred.experiments.runner import Pipeline, prepare_drive, _load_part  # noqa: E402
from hddpred.features import fold as fold_mod  # noqa: E402
from hddpred.models.sequence import RNNModel  # noqa: E402
from export_results import disk_rank, load_part, month_windows, rescale  # noqa: E402

DRIVE = "TOSHIBA_20MG07ACA14TA"
BASE_EXPERIMENT = "toslb_14_pauc"     # lookback 14 확정판 (pAUC 감시)
SEED = 42
PAUC_MAX_FPR = 0.05
STUDY_NAME_FMT = "{cell}_tos_win10_pauc5_mon"
STUDY_DB = ROOT / "runs" / "optuna" / "rnn.db"
RESULT_DIR = ROOT / "results"

# 고정값. 불균형 처리를 하지 않는다는 확정 사항이 여기 박혀 있다.
FIXED_TRAINING = {
    "epochs": 30,
    "optimizer": "adamw",
    "loss": "bce",
    "auto_pos_weight": False,
    "early_stopping_patience": 5,
    # 조기종료 감시값도 목적함수와 같은 pAUC 로 맞춘다. 이 값이 그대로
    # trial.report 로 나가므로 가지치기 기준도 함께 정합해진다.
    "early_stopping_metric": "val_pauc",
    "grad_clip": 1.0,
    "num_workers": 0,
    "amp": True,
}


def suggest(trial, cell: str):
    """탐색 공간.

    현행 값(hidden 128, layers 2, dropout 0.2, lr 1e-3, wd 1e-4, batch 512)을
    전부 안쪽에 포함한다. 탐색이 현행보다 나쁜 곳만 뒤지는 일이 없게 한다.

    bidirectional 은 넣지 않는다. 창 안의 모든 시점이 예측 시점보다 과거라
    누출은 아니지만, 파라미터가 두 배가 되어 "길이/용량" 축이 섞인다.
    """
    params = {
        "cell": cell,
        "bidirectional": False,
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
    }
    # 층이 하나면 nn.GRU 가 dropout 을 무시한다. 탐색 공간에서도 빼서
    # 의미 없는 차원이 TPE 를 헷갈리게 하지 않도록 한다.
    params["dropout"] = (
        trial.suggest_float("dropout", 0.0, 0.5) if params["num_layers"] > 1 else 0.0
    )
    training = {
        **FIXED_TRAINING,
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        # 256 은 뺐다. 실측에서 batch 256 조합이 1,729초로 512(643초)의 2.7배였다.
        # 스텝 수가 두 배가 되는데 이 문제에서 작은 배치의 이점은 확인된 바 없다.
        "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
    }
    return params, training


def record_run(cell: str, study_name: str, frame, n_pruned: int, n_attempted: int) -> str:
    """탐색 한 판을 results/tuning_log.md 에 append 한다.

    같은 study 를 --trials 를 늘려 이어 돌릴 수 있으므로, 매 판이 어디까지
    갔고 무엇이 나왔는지 남겨 둬야 "더 돌릴 가치가 있나" 를 판단할 수 있다.
    """
    from datetime import datetime

    best = frame.iloc[0]
    spread = float(frame.pauc.max() - frame.pauc.min()) if len(frame) > 1 else 0.0
    top5 = frame.head(5).pauc
    top_spread = float(top5.max() - top5.min()) if len(top5) > 1 else 0.0
    # 상위권이 서로 구분되지 않으면 더 돌려도 같은 자리를 맴돌 확률이 높다.
    verdict = ("상위권이 구분되지 않는다 (상위 5개 폭 %.4f). 더 돌려도 같은 자리일 "
               "가능성이 높다." % top_spread) if top_spread < 0.005 else \
              ("상위권이 갈린다 (상위 5개 폭 %.4f). 더 돌릴 가치가 있다." % top_spread)

    line = (
        f"\n## {datetime.now():%Y-%m-%d %H:%M} — {cell.upper()} / `{study_name}`\n\n"
        f"- 시도 {n_attempted}회 → 완료 {len(frame)} / 가지치기 {n_pruned}\n"
        f"- 최고 val pAUC@FAR<=5% **{best.pauc:.4f}** (trial {int(best.trial)}): "
        f"hidden {int(best.hidden_size)}, layers {int(best.num_layers)}, "
        f"lr {best.learning_rate:.2e}, wd {best.weight_decay:.2e}, "
        f"batch {int(best.batch_size)}\n"
        f"- 완료 시도 전체 폭 {spread:.4f}, 상위 5개 폭 {top_spread:.4f}\n"
        f"- 판정: {verdict}\n"
        f"- 이어 돌리려면: `python scripts/run_optuna_rnn.py --cell {cell} "
        f"--trials {n_attempted + 40}` (같은 study 에 40회 추가)\n"
    )
    path = RESULT_DIR / "tuning_log.md"
    if not path.exists():
        path.write_text(
            "# 하이퍼파라미터 탐색 기록\n\n"
            "`scripts/run_optuna_rnn.py` 가 append 한다.\n\n"
            "study 는 SQLite 에 남아 있어 `--trials` 를 늘리면 같은 study 를\n"
            "이어서 탐색한다 (TPE 가 앞선 결과를 그대로 활용한다).\n",
            encoding="utf-8")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
    return verdict


def val_pauc(model, parts) -> tuple[float, float]:
    """검증 구간의 부분 AUC와 FAR 1% 재현율.

    목적함수는 pAUC 하나지만 재현율도 같이 돌려준다. pAUC@FAR<=5% 는
    0~5% 전체 면적이라 5% 근처만 좋아져도 값이 오른다 — 논문이 헤드라인으로
    쓰는 FAR 1% 운영점과 어긋날 수 있어서, 시행마다 둘을 같이 남겨
    나중에 어긋남 자체를 관찰할 수 있게 한다. 선정에는 쓰지 않는다.
    """
    ranks, flags = [], []
    for part in parts:
        rank, has_window = disk_rank(model, part, "in_horizon")
        ranks.append(rank.to_numpy())
        flags.append(has_window.to_numpy())
    score = np.concatenate(ranks)
    actual = np.concatenate(flags).astype(int)
    pauc = float(roc_auc_score(actual, score, max_fpr=PAUC_MAX_FPR))
    thr = float(np.quantile(score[actual == 0], 0.99))
    recall = float(((score >= thr) & (actual == 1)).sum() / max((actual == 1).sum(), 1))
    return pauc, recall


def main() -> int:
    ap = argparse.ArgumentParser(description="GRU 하이퍼파라미터 탐색")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--cell", default="gru", choices=["gru", "lstm"])
    args = ap.parse_args()
    study_name = STUDY_NAME_FMT.format(cell=args.cell)

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    STUDY_DB.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{STUDY_DB.as_posix()}",
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0),
        # 중앙값 기준 2에폭은 너무 이르다 — 실측에서 40시도 중 33개가 잘려
        # TPE 가 제안한 조합이 하나도 끝까지 못 갔다. 하위 25%만, 그것도
        # 5에폭 이후에 자른다. 조기 종료(patience 5)가 이미 있으므로
        # 가지치기는 '다른 시도 대비 확실히 나쁜' 것만 걸러내면 된다.
        pruner=optuna.pruners.PercentilePruner(
            25.0, n_startup_trials=5, n_warmup_steps=5
        ),
    )

    if not args.report:
        cfg = cfg_mod.load_yaml(paths.CONFIG_DIR / "experiments" / f"{BASE_EXPERIMENT}.yaml")
        pipeline = Pipeline.from_experiment(cfg)
        prepared = prepare_drive(DRIVE, pipeline)
        fold = sorted(prepared.folds, key=lambda f: f.fold)[0]
        threads = int(pipeline.preprocessing.get("duckdb", {}).get("threads", 8))
        horizon = int(pipeline.labeling["horizon_days"])

        print("[optuna] 데이터 적재 (한 번만)", flush=True)
        started = time.time()
        train = _load_part(prepared, fold, "train",
                           "sequence", pipeline.features, pipeline.labeling, threads)
        val_full = _load_part(prepared, fold, "val",
                              "sequence", pipeline.features, pipeline.labeling, threads)
        # 스케일러는 하이퍼파라미터와 무관하므로 한 번만 적합한다.
        scaling = pipeline.features.get("scaling", {})
        scaler = fold_mod.Scaler(
            scaling.get("method", "standard"), scaling.get("clip_quantile")
        ).fit(train.matrix)
        train.matrix = scaler.transform(train.matrix, fill_nan=True)
        val_full.matrix = scaler.transform(val_full.matrix, fill_nan=True)
        # 목적함수용 val 은 월별 창으로 따로 읽는다 (test 와 같은 단위).
        val_months = [
            rescale(load_part(prepared, pipeline, "sequence", a, b, horizon, threads), scaler)
            for a, b in month_windows(*fold.window("val"))
        ]
        print(f"  train {len(train):,} | val {len(val_full):,} "
              f"| val 월별 {len(val_months)}창 ({time.time() - started:.1f}s)", flush=True)

        def objective(trial):
            params, training = suggest(trial, args.cell)
            model = RNNModel(params, training, seed=SEED)

            def on_epoch(epoch, best):
                trial.report(best, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            model.epoch_callback = on_epoch
            model.fit(train, val_full)
            model.epoch_callback = None
            score, recall01 = val_pauc(model, val_months)
            trial.set_user_attr("pauc", score)
            trial.set_user_attr("recall01", recall01)
            trial.set_user_attr("epochs", model.fit_info.get("epochs_run"))
            return score

        done = len([t for t in study.trials if t.state.is_finished()])
        remaining = max(0, args.trials - done)
        print(f"[optuna] 완료 {done}개, 남은 시도 {remaining}개 (seed {SEED})", flush=True)
        for n in range(remaining):
            t0 = time.time()
            study.optimize(objective, n_trials=1, catch=(Exception,))
            last = study.trials[-1]
            mark = ("pruned" if str(last.state) == "TrialState.PRUNED"
                    else f"pAUC={last.value:.4f}")
            best = study.best_value if any(
                t.value is not None for t in study.trials) else float("nan")
            print(f"  [{done + n + 1:>3}/{args.trials}] {mark}  "
                  f"({time.time() - t0:.0f}s)  best={best:.4f}", flush=True)

    # 가지치기된 시도는 value 에 마지막 중간 보고값(val PR-AUC)이 남는다.
    # 목적함수(pAUC)와 척도가 달라 섞이면 순위표가 망가진다. COMPLETE 만 쓴다.
    rows = [
        {"trial": t.number, "pauc": t.value, "epochs": t.user_attrs.get("epochs"), **t.params}
        for t in study.trials
        if str(t.state) == "TrialState.COMPLETE" and t.value is not None
    ]
    n_pruned = sum(1 for t in study.trials if str(t.state) == "TrialState.PRUNED")
    print(f"\n완료 {len(rows)}개 / 가지치기 {n_pruned}개")
    if not rows:
        print("완료된 시도가 없다.")
        return 1
    frame = pd.DataFrame(rows).sort_values("pauc", ascending=False)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULT_DIR / f"optuna_{args.cell}_trials.csv", index=False, encoding="utf-8-sig")

    chosen = frame.iloc[0]
    print(f"\n--- val pAUC@FAR<={PAUC_MAX_FPR:.0%} 상위 8개 (seed {SEED}) ---")
    print(frame.head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n선택: trial {int(chosen['trial'])}  pAUC {chosen['pauc']:.4f}")

    import yaml

    layers = int(chosen["num_layers"])
    out = ROOT / "configs" / "models" / f"{args.cell}_tuned.yaml"
    cfg_out = {
        "name": f"{args.cell}_tuned",
        "family": "sequence",
        "class": "hddpred.models.sequence.RNNModel",
        "params": {
            "cell": args.cell,
            "hidden_size": int(chosen["hidden_size"]),
            "num_layers": layers,
            "dropout": float(chosen["dropout"]) if layers > 1 else 0.0,
            "bidirectional": False,
        },
        "training": {
            **FIXED_TRAINING,
            "learning_rate": float(chosen["learning_rate"]),
            "weight_decay": float(chosen["weight_decay"]),
            "batch_size": int(chosen["batch_size"]),
        },
    }
    header = (
        "# Optuna 로 고른 GRU 하이퍼파라미터.\n"
        f"#   탐색: 시드 {SEED} 의 val 부분 AUC(FAR <= {PAUC_MAX_FPR:.0%}) 최대화\n"
        "#   val 점수는 월별 창으로 접는다 (test 와 같은 단위).\n"
        "#   test 는 탐색에 쓰지 않았다. 확정 후 따로 채점한다.\n"
        "#   scripts/run_optuna_rnn.py 가 생성한다. 직접 고치지 마라.\n\n"
    )
    out.write_text(header + yaml.safe_dump(cfg_out, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    print(f"\n[저장] {out}")
    print(f"[저장] {RESULT_DIR / f'optuna_{args.cell}_trials.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
