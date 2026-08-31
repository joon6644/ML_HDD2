"""Row-level loss plus a lead-time penalty (exploratory, not in the paper).

The max-pooled bag objective fails because its gradient is sparse: on a censored
disk with 2,300 rows the softmax hands everything to one row, and pushing that
row down just promotes the next. What actually has to fall is the *level* of
every row outside the actionable window, since the first one to cross decides the
disk. So keep the row-level objective and add a dense penalty on exactly those
rows:

    L = BCE(p, y)  +  lambda * mean_{out} p

"out" is every row that could only ever produce a non-On-time alarm: rows on a
failure-observed disk with RUL > H, and every row on a censored disk. Unlike a
sample weight, this term keeps pushing after a row is already on the right side
of the boundary, which is what moving a maximum requires.

    d/dz [ BCE ]        = p - y
    d/dz [ lambda p ]   = lambda * p (1 - p)

Usage
  python experiments/train_leadtime_penalty.py                  # sweep lambda
  python experiments/train_leadtime_penalty.py --lams 0 2 8
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EXP_DIR)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

import config
from data_loader import load_dataset, create_binary_target
from explore_operational_training import judge, pick_tau_op
from train_operational_loss import sigmoid

H = config.TARGET_LEAD_TIME


def make_objective(y, out_mask, lam):
    y = y.astype(np.float64)
    w_out = out_mask.astype(np.float64) * lam

    def obj(z, _dataset):
        p = sigmoid(z)
        grad = (p - y) + w_out * p * (1.0 - p)
        hess = p * (1.0 - p) * (1.0 + w_out * (1.0 - 2.0 * p))
        return grad, np.maximum(hess, 1e-6)

    return obj


def run(dataset, seed, lams, rounds, lr):
    import lightgbm as lgb

    splitted = os.path.join(PROJECT_ROOT, "data", "splitted", dataset)
    train_df, val_df, test_df, features = load_dataset(splitted_dir=splitted, model="lgbm")
    for df in (val_df, test_df):
        df.sort_values(["serial_number", "date"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    y_tr = create_binary_target(train_df).astype(np.float64)
    rul = train_df["RUL"].to_numpy()
    cens = train_df["censored"].to_numpy() == 1
    out_mask = cens | (rul > H)
    print(f"  out-of-window rows: {out_mask.sum():,} / {len(train_df):,}")

    X = train_df[features]
    rows = []
    for lam in lams:
        t0 = time.time()
        ds = lgb.Dataset(X, label=y_tr, free_raw_data=False)
        booster = lgb.train(
            {"objective": make_objective(y_tr, out_mask, lam), "learning_rate": lr,
             "num_leaves": 31, "min_child_samples": 100, "bagging_fraction": 0.9,
             "bagging_freq": 1, "feature_fraction": 0.9, "seed": seed,
             "num_threads": -1, "verbose": -1},
            ds, num_boost_round=rounds,
        )
        p_va = sigmoid(booster.predict(val_df[features], raw_score=True))
        p_te = sigmoid(booster.predict(test_df[features], raw_score=True))
        tau = pick_tau_op(val_df, p_va)
        m = judge(test_df, p_te, tau)
        m.update(dataset=dataset, seed=seed, lam=lam, tau=tau, rounds=rounds,
                 lr=lr, secs=round(time.time() - t0, 1))
        rows.append(m)
        print(f"  [lam={lam:<5}] tau={tau:.3f}  P={m['precision']:.3f} R={m['recall']:.3f} "
              f"FAR={m['far']:.2%} medLT={m['median_lt']:.0f}  "
              f"(OT {m['OT']} / E {m['E']} / M {m['M']} / CE {m['CE']} / CN {m['CN']})  "
              f"{m['secs']}s", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HGST_20HUH721212ALN604")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 1.0, 4.0, 16.0])
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    frames = []
    for seed in args.seeds:
        print(f"\n=== {args.dataset}  seed {seed} ===", flush=True)
        frames.append(run(args.dataset, seed, args.lams, args.rounds, args.lr))
    out = pd.concat(frames, ignore_index=True)

    out_dir = os.path.join(PROJECT_ROOT, "results", "operational_training")
    os.makedirs(out_dir, exist_ok=True)
    csv = os.path.join(out_dir, f"leadtime_penalty_{args.dataset}.csv")
    out.to_csv(csv, index=False, encoding="utf-8-sig")
    print(f"\ncsv: {csv}")


if __name__ == "__main__":
    main()
