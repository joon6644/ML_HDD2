"""Where row-level early stopping halts, and where the operational optimum is.

The paper stops LightGBM on validation average precision, which in practice fires
after a handful of trees. This traces both curves on the same run: the row-level
metric that decides when to stop, and the HDD-level verdict at the threshold that
meets the same 1% false-alarm constraint. If they peak at different depths, the
stopping rule is a third decision the row-level view gets wrong.

Usage
  python experiments/trace_early_stopping.py
  python experiments/trace_early_stopping.py --dataset ST12000NM0007 --seeds 42 43
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EXP_DIR)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from data_loader import load_dataset, create_binary_target
from explore_operational_training import judge, pick_tau_op

CHECKPOINTS = [5, 10, 20, 40, 80, 150, 300]


def run(dataset, seed, rounds):
    import lightgbm as lgb

    splitted = os.path.join(PROJECT_ROOT, "data", "splitted", dataset)
    train_df, val_df, test_df, features = load_dataset(splitted_dir=splitted, model="lgbm")
    for df in (val_df, test_df):
        df.sort_values(["serial_number", "date"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    y_tr = create_binary_target(train_df)
    y_va = create_binary_target(val_df)

    # Same hyperparameters as the paper, minus the early-stopping callback.
    booster = lgb.train(
        {"objective": "binary", "metric": "average_precision", "learning_rate": 0.05,
         "num_leaves": 31, "min_data_in_leaf": 100, "bagging_fraction": 0.9,
         "bagging_freq": 1, "feature_fraction": 0.9, "seed": seed,
         "num_threads": -1, "verbose": -1},
        lgb.Dataset(train_df[features], label=y_tr), num_boost_round=rounds,
    )

    rows = []
    for k in [c for c in CHECKPOINTS if c <= rounds]:
        t0 = time.time()
        p_va = booster.predict(val_df[features], num_iteration=k)
        p_te = booster.predict(test_df[features], num_iteration=k)
        ap = average_precision_score(y_va, p_va)
        tau = pick_tau_op(val_df, p_va)
        m = judge(test_df, p_te, tau)
        m.update(dataset=dataset, seed=seed, trees=k, val_ap=ap, tau=tau,
                 secs=round(time.time() - t0, 1))
        rows.append(m)
        print(f"  trees={k:<4} val AP={ap:.4f}  tau={tau:.3f}  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} FAR={m['far']:.2%} "
              f"medLT={m['median_lt']:.0f}  (OT {m['OT']} / E {m['E']} / M {m['M']} "
              f"/ CE {m['CE']})  {m['secs']}s", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HGST_20HUH721212ALN604")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--rounds", type=int, default=300)
    args = ap.parse_args()

    frames = []
    for seed in args.seeds:
        print(f"\n=== {args.dataset}  seed {seed} ===", flush=True)
        frames.append(run(args.dataset, seed, args.rounds))
    out = pd.concat(frames, ignore_index=True)

    out_dir = os.path.join(PROJECT_ROOT, "results", "operational_training")
    os.makedirs(out_dir, exist_ok=True)
    csv = os.path.join(out_dir, f"early_stopping_trace_{args.dataset}.csv")
    out.to_csv(csv, index=False, encoding="utf-8-sig")
    print(f"\ncsv: {csv}")


if __name__ == "__main__":
    main()
