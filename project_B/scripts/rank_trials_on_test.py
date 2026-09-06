"""탐색된 상위 조합들을 순위대로 test 에 채점한다.

    python scripts/rank_trials_on_test.py --cell lstm --top 8

val pAUC 상위 K개를 각각 학습해 test 지표를 낸다. 결과는
results/<cell>_trial_ranking.csv 에 남는다.

━━ 이 표를 어떻게 읽어야 하는가 ━━

첫 줄(val 1위)만이 held-out 측정이다. 나머지는 test 를 여러 번 들여다본
결과이므로, 그중 최댓값은 "이 탐색 공간에서 도달 가능한 상한" 이지 held-out
성능이 아니다.

test 고장 162대 기준 Recall 의 이항 표준오차가 0.035 다. 후보 8개를 훑으면
아무 신호가 없어도 최댓값이 val 1위보다 0.03~0.05 높게 나오는 것이 정상이다.
그러므로 이 표의 최댓값을 "튜닝으로 얻은 개선" 이라고 부르면 안 된다.

논문에 쓴다면 두 줄을 같이 적는 형태가 정직하다.

    Proposed (val 선정)                 0.7xx
    참고: 탐색 공간 내 test 최고        0.7yy   <- 상한, 선정에 쓰지 않음
"""

from __future__ import annotations

import argparse
import json
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
from export_results import (  # noqa: E402
    disk_rank, load_part, month_windows, rescale,
)
from run_optuna_rnn import FIXED_TRAINING, PAUC_MAX_FPR  # noqa: E402

DRIVE = "TOSHIBA_20MG07ACA14TA"
BASE_EXPERIMENT = "toslb_14"
SEED = 42
FAR_TARGETS = [0.001, 0.005, 0.01, 0.05]


def build(row, cell: str):
    layers = int(row["num_layers"])
    params = {
        "cell": cell,
        "hidden_size": int(row["hidden_size"]),
        "num_layers": layers,
        "dropout": float(row["dropout"]) if layers > 1 and row.get("dropout") == row.get("dropout") else 0.0,
        "bidirectional": False,
    }
    training = {
        **FIXED_TRAINING,
        "learning_rate": float(row["learning_rate"]),
        "weight_decay": float(row["weight_decay"]),
        "batch_size": int(row["batch_size"]),
    }
    return params, training


def score(model, parts, thresholds=None):
    """월별 창의 디스크 점수를 모아 지표를 낸다."""
    ranks, flags = [], []
    for part in parts:
        rank, has_window = disk_rank(model, part, "in_horizon")
        ranks.append(rank.to_numpy())
        flags.append(has_window.to_numpy())
    return ranks, flags


def main() -> int:
    ap = argparse.ArgumentParser(description="상위 조합을 순위대로 test 채점")
    ap.add_argument("--cell", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    trials = pd.read_csv(ROOT / "results" / f"optuna_{args.cell}_trials.csv",
                         encoding="utf-8-sig").sort_values("pauc", ascending=False)
    top = trials.head(args.top)

    cfg = cfg_mod.load_yaml(paths.CONFIG_DIR / "experiments" / f"{BASE_EXPERIMENT}.yaml")
    pipeline = Pipeline.from_experiment(cfg)
    prepared = prepare_drive(DRIVE, pipeline)
    folds = sorted(prepared.folds, key=lambda f: f.fold)
    threads = int(pipeline.preprocessing.get("duckdb", {}).get("threads", 8))
    horizon = int(pipeline.labeling["horizon_days"])
    fold = folds[0]

    print("[rank] 데이터 적재", flush=True)
    train = _load_part(prepared, fold, "train", "sequence",
                       pipeline.features, pipeline.labeling, threads)
    val_full = _load_part(prepared, fold, "val", "sequence",
                          pipeline.features, pipeline.labeling, threads)
    scaling = pipeline.features.get("scaling", {})
    scaler = fold_mod.Scaler(scaling.get("method", "standard"),
                             scaling.get("clip_quantile")).fit(train.matrix)
    train.matrix = scaler.transform(train.matrix, fill_nan=True)
    val_full.matrix = scaler.transform(val_full.matrix, fill_nan=True)
    val_parts = [rescale(load_part(prepared, pipeline, "sequence", a, b, horizon, threads), scaler)
                 for a, b in month_windows(*fold.window("val"))]
    test_parts = [(str(f.window("test")[0])[:7],
                   rescale(load_part(prepared, pipeline, "sequence",
                                     *f.window("test"), horizon, threads), scaler))
                  for f in folds]

    rows = []
    for rank_i, (_, row) in enumerate(top.iterrows(), start=1):
        params, training = build(row, args.cell)
        t0 = time.time()
        model = RNNModel(params, training, seed=SEED)
        model.fit(train, val_full)

        vr, vf = score(model, val_parts)
        vs, vy = np.concatenate(vr), np.concatenate(vf).astype(int)
        thr = {t: float(np.quantile(vs[vy == 0], 1.0 - t)) for t in FAR_TARGETS}
        val_pauc = roc_auc_score(vy, vs, max_fpr=PAUC_MAX_FPR)

        monthly = {t: [] for t in FAR_TARGETS}
        far_m, auc_m, pauc_m = [], [], []
        for _, part in test_parts:
            r, hw = disk_rank(model, part, "in_horizon")
            s, y = r.to_numpy(), hw.to_numpy().astype(bool)
            for t in FAR_TARGETS:
                al = s >= thr[t]
                monthly[t].append((al & y).sum() / y.sum())
            al = s >= thr[0.01]
            far_m.append((al & ~y).sum() / (~y).sum())
            auc_m.append(roc_auc_score(y.astype(int), s))
            pauc_m.append(roc_auc_score(y.astype(int), s, max_fpr=PAUC_MAX_FPR))

        rec = {f"r{int(t*1000):03d}": float(np.mean(monthly[t])) for t in FAR_TARGETS}
        rows.append(dict(val_rank=rank_i, trial=int(row["trial"]),
                         val_pauc=float(row["pauc"]), val_pauc_recomputed=val_pauc,
                         **rec, r01_month_sd=float(np.std(monthly[0.01], ddof=1)),
                         test_pauc=float(np.mean(pauc_m)), roc_auc=float(np.mean(auc_m)),
                         far=float(np.mean(far_m)), secs=time.time() - t0,
                         hidden=params["hidden_size"], layers=params["num_layers"],
                         lr=training["learning_rate"], batch=training["batch_size"]))
        x = rows[-1]
        print(f"  val {rank_i:>2}위 (trial {x['trial']:>2})  val pAUC {x['val_pauc']:.4f}  "
              f"-> test R@1% {x['r010']:.3f}  pAUC {x['test_pauc']:.3f}  "
              f"({x['secs']:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    path = ROOT / "results" / f"{args.cell}_trial_ranking.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"\n=== val 순위별 test 성능 ({args.cell}) ===")
    show = ["val_rank", "trial", "val_pauc", "r001", "r005", "r010", "r050",
            "test_pauc", "roc_auc"]
    print(out[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    head = out.iloc[0]
    best = out.loc[out.r010.idxmax()]
    print(f"\nval 1위 조합의 test R@1% = {head.r010:.4f}   <- held-out 측정")
    print(f"탐색 공간 내 test 최고    = {best.r010:.4f}  (val {int(best.val_rank)}위)"
          f"   <- 상한, 선정 근거로 쓸 수 없다")
    print(f"[저장] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
