"""TCN 하이퍼파라미터 탐색.

    python scripts/run_optuna_tcn.py --trials 50
    python scripts/run_optuna_tcn.py --report          # 재탐색 없이 집계만

run_optuna_rnn.py 와 같은 절차다. 분할·피처·정규화·불균형 처리·시드·목적함수·
가지치기 설정을 전부 그쪽과 맞추고, 탐색 공간만 TCN 구조에 맞게 바꾼다.

  분할      도시바 10-2-6 (test 2025-10 ~ 2026-03)
  피처      원본 SMART 16개, lookback 14일
  정규화    min-max (train 구간에서만 적합)
  불균형    처리 없음, 순수 BCE
  시드      42 하나
  목적함수  val 의 pAUC @ FAR <= 5%, 월별 창으로 접어서 산출

━━ 탐색 공간을 RNN 쪽과 다르게 잡는 이유 ━━

TCN 은 은닉 상태 크기와 층 수가 아니라 채널 폭·레벨 수·커널 크기로 용량이
정해진다. 그리고 레벨 수와 커널 크기는 자유롭게 둘 수 없다 — 수용 영역이
입력 길이를 덮어야 한다 (Bai, Kolter & Koltun 2018).

    RF = 1 + 2(k-1)(2^n - 1)        블록당 conv 2개, dilation 2^0 .. 2^(n-1)

lookback 이 14 이므로 RF >= 14 를 만족해야 한다.

    k=2: n=3 -> 15   n=4 -> 31
    k=3: n=3 -> 29   n=4 -> 61
    k=5: n=3 -> 57   n=4 -> 121

n=2 는 k=5 라면 RF=25 로 덮지만 k=2(7)·k=3(13) 에서 모자란다. n=3 이 세 커널
크기 모두에서 유효한 최소 레벨 수라, 공간을 직사각형으로 두어 모든 표본이
유효하도록 n in {3,4} 로 잡는다. 조건부로 n=2 를 허용하면 무효 조합을
가지치기로 버려야 해서 같은 시행 수로 덜 탐색하게 된다.

레벨 수를 고정하지 않는 것은 RNN 쪽이 num_layers 를 1~3 으로 탐색하는 것과
대등하게 두기 위해서다. 커널 크기는 TCN 에만 있는 축이라 탐색 축이 하나 더
많다 — 구조가 다른 모델에 같은 축을 강요하는 대신 각자의 용량 파라미터를
같은 시행 수로 탐색하게 한다.

bidirectional 을 빼는 RNN 쪽과 같은 취지로 causal padding 은 고정한다.
미래 시점을 보지 않는다.
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
from hddpred.models.sequence import TCNModel  # noqa: E402
from export_results import disk_rank, load_part, month_windows, rescale  # noqa: E402

DRIVE = "TOSHIBA_20MG07ACA14TA"
BASE_EXPERIMENT = "toslb_14_pauc"     # lookback 14 확정판 (pAUC 감시)
SEED = 42
PAUC_MAX_FPR = 0.05
LOOKBACK = 14
STUDY_NAME = "tcn_tos_win10_pauc5_mon"
STUDY_DB = ROOT / "runs" / "optuna" / "tcn.db"
RESULT_DIR = ROOT / "results"

# run_optuna_rnn.FIXED_TRAINING 과 동일해야 비교가 성립한다.
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


def receptive_field(kernel_size: int, num_levels: int) -> int:
    """Bai et al. 2018 의 수용 영역. 블록당 conv 2개, dilation 2^0..2^(n-1)."""
    return 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1)


def suggest(trial):
    """탐색 공간.

    현행 tcn_pauc.yaml (channels 64x3, k=3, dropout 0.2, lr 1e-3, wd 1e-4,
    batch 512) 을 전부 안쪽에 포함한다.
    """
    width = trial.suggest_categorical("width", [32, 64, 128, 256])
    num_levels = trial.suggest_int("num_levels", 3, 4)
    kernel_size = trial.suggest_categorical("kernel_size", [2, 3, 5])
    rf = receptive_field(kernel_size, num_levels)
    if rf < LOOKBACK:                       # 설계상 도달하지 않는다. 방어용.
        raise ValueError(f"수용 영역 {rf} < lookback {LOOKBACK}")
    params = {
        "channels": [width] * num_levels,
        "kernel_size": kernel_size,
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "causal": True,
    }
    training = {
        **FIXED_TRAINING,
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
    }
    trial.set_user_attr("receptive_field", rf)
    return params, training


def val_pauc(model, parts) -> tuple[float, float]:
    """검증 구간의 부분 AUC와 FAR 1% 재현율. run_optuna_rnn.val_pauc 와 동일.

    창마다 산출한 뒤 평균한다. 풀링하지 않는다 — 논문이 보고하는 지표가
    전부 월별 산출 후 평균이므로(3.4) 목적함수도 같은 방식이어야 한다.
    """
    paucs, recalls = [], []
    for part in parts:
        rank, has_window = disk_rank(model, part, "in_horizon")
        score = rank.to_numpy()
        actual = has_window.to_numpy().astype(int)
        n_pos = int(actual.sum())
        if n_pos == 0 or n_pos == len(actual):
            continue                    # 한쪽 클래스뿐이면 AUC 가 정의되지 않는다
        paucs.append(float(roc_auc_score(actual, score, max_fpr=PAUC_MAX_FPR)))
        thr = float(np.quantile(score[actual == 0], 0.99))
        recalls.append(float(((score >= thr) & (actual == 1)).sum() / n_pos))
    if not paucs:
        return float("nan"), float("nan")
    return float(np.mean(paucs)), float(np.mean(recalls))


def record_run(frame, n_pruned: int, n_attempted: int) -> str:
    """탐색 한 판을 results/tuning_log.md 에 append 한다."""
    from datetime import datetime

    best = frame.iloc[0]
    spread = float(frame.pauc.max() - frame.pauc.min()) if len(frame) > 1 else 0.0
    top5 = frame.head(5).pauc
    top_spread = float(top5.max() - top5.min()) if len(top5) > 1 else 0.0
    verdict = ("상위권이 구분되지 않는다 (상위 5개 폭 %.4f). 더 돌려도 같은 자리일 "
               "가능성이 높다." % top_spread) if top_spread < 0.005 else \
              ("상위권이 갈린다 (상위 5개 폭 %.4f). 더 돌릴 가치가 있다." % top_spread)
    line = (
        f"\n## {datetime.now():%Y-%m-%d %H:%M} — TCN / `{STUDY_NAME}`\n\n"
        f"- 시도 {n_attempted}회 → 완료 {len(frame)} / 가지치기 {n_pruned}\n"
        f"- 최고 val pAUC@FAR<=5% **{best.pauc:.4f}** (trial {int(best.trial)}): "
        f"width {int(best.width)}, levels {int(best.num_levels)}, "
        f"k {int(best.kernel_size)}, dropout {best.dropout:.3f}, "
        f"lr {best.learning_rate:.2e}, wd {best.weight_decay:.2e}, "
        f"batch {int(best.batch_size)}\n"
        f"- 완료 시도 전체 폭 {spread:.4f}, 상위 5개 폭 {top_spread:.4f}\n"
        f"- 판정: {verdict}\n"
        f"- 이어 돌리려면: `python scripts/run_optuna_tcn.py "
        f"--trials {n_attempted + 40}`\n"
    )
    path = RESULT_DIR / "tuning_log.md"
    if not path.exists():
        path.write_text("# 하이퍼파라미터 탐색 기록\n\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser(description="TCN 하이퍼파라미터 탐색")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--report", action="store_true")
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
        scaling = pipeline.features.get("scaling", {})
        scaler = fold_mod.Scaler(
            scaling.get("method", "standard"), scaling.get("clip_quantile")
        ).fit(train.matrix)
        train.matrix = scaler.transform(train.matrix, fill_nan=True)
        val_full.matrix = scaler.transform(val_full.matrix, fill_nan=True)
        val_months = [
            rescale(load_part(prepared, pipeline, "sequence", a, b, horizon, threads), scaler)
            for a, b in month_windows(*fold.window("val"))
        ]
        print(f"  train {len(train):,} | val {len(val_full):,} "
              f"| val 월별 {len(val_months)}창 ({time.time() - started:.1f}s)", flush=True)

        def objective(trial):
            params, training = suggest(trial)
            model = TCNModel(params, training, seed=SEED)

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

    rows = [
        {"trial": t.number, "pauc": t.value, "epochs": t.user_attrs.get("epochs"),
         "receptive_field": t.user_attrs.get("receptive_field"), **t.params}
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
    frame.to_csv(RESULT_DIR / "optuna_tcn_trials.csv", index=False, encoding="utf-8-sig")

    chosen = frame.iloc[0]
    print(f"\n--- val pAUC@FAR<={PAUC_MAX_FPR:.0%} 상위 8개 (seed {SEED}) ---")
    print(frame.head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n선택: trial {int(chosen['trial'])}  pAUC {chosen['pauc']:.4f}")
    print(record_run(frame, n_pruned, len(study.trials)))

    import yaml

    width = int(chosen["width"])
    num_levels = int(chosen["num_levels"])
    kernel_size = int(chosen["kernel_size"])
    cfg_out = {
        "name": "tcn_tuned",
        "family": "sequence",
        "class": "hddpred.models.sequence.TCNModel",
        "params": {
            "channels": [width] * num_levels,
            "kernel_size": kernel_size,
            "dropout": float(chosen["dropout"]),
            "causal": True,
        },
        "training": {
            **FIXED_TRAINING,
            "learning_rate": float(chosen["learning_rate"]),
            "weight_decay": float(chosen["weight_decay"]),
            "batch_size": int(chosen["batch_size"]),
        },
    }
    out = ROOT / "configs" / "models" / "tcn_tuned.yaml"
    header = (
        "# Optuna 로 고른 TCN 하이퍼파라미터.\n"
        f"#   탐색: 시드 {SEED} 의 val 부분 AUC(FAR <= {PAUC_MAX_FPR:.0%}) 최대화\n"
        "#   val 점수는 월별 창으로 접는다 (test 와 같은 단위).\n"
        "#   test 는 탐색에 쓰지 않았다. 확정 후 따로 채점한다.\n"
        f"#   수용 영역 {receptive_field(kernel_size, num_levels)} "
        f">= lookback {LOOKBACK} (Bai et al. 2018 규칙).\n"
        "#   scripts/run_optuna_tcn.py 가 생성한다. 직접 고치지 마라.\n\n"
    )
    out.write_text(header + yaml.safe_dump(cfg_out, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    print(f"\n[저장] {out}")
    print(f"[저장] {RESULT_DIR / 'optuna_tcn_trials.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
