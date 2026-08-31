"""Train against the operational verdict itself (exploratory, not in the paper).

The verdict on a disk is decided by its first crossing, so what the model has to
get right is not each row but two numbers per disk:

    the highest score inside the actionable window   -> should clear tau
    the highest score anywhere outside it            -> should not

Both maxima are replaced by a temperature-softened maximum, which is
differentiable and decomposes per row, so it can be handed to LightGBM as a
custom objective. Rows outside the window on a failure-observed disk and every
row on a censored disk land in the same "should stay quiet" bag; that is exactly
the asymmetry row-level cross-entropy cannot see, since it charges the same price
for one crossing on a disk and five hundred.

    softmax weights   w_i = exp(z_i / T) / sum_j exp(z_j / T)
    soft maximum      s   = sum_i w_i z_i
    d s / d z_i       = w_i (1 + (z_i - s) / T)
    loss              BCE(sigmoid(s_in), 1) + POS_W * BCE(sigmoid(s_out), 0)

Usage
  python experiments/train_operational_loss.py                     # HGST, seed 42
  python experiments/train_operational_loss.py --rounds 300 --temp 2.0
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

H = config.TARGET_LEAD_TIME


def sigmoid(x):
    """Overflow-free logistic. Raw scores wander far during boosting."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def bag_index(serial: np.ndarray, mask: np.ndarray):
    """Row positions of one bag family, plus the start offset of each bag.

    The frame is sorted by serial, so a bag's rows stay contiguous after the
    mask is applied and reduceat can walk them without a groupby.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return idx, np.empty(0, dtype=np.int64)
    s = serial[idx]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    return idx, starts


def soft_max_and_weights(z: np.ndarray, starts: np.ndarray, temp: float):
    """Temperature-softened maximum per bag, and the softmax weight of each row."""
    gmax = np.maximum.reduceat(z, starts)
    n = np.diff(np.r_[starts, z.size])
    e = np.exp((z - np.repeat(gmax, n)) / temp)
    denom = np.add.reduceat(e, starts)
    w = e / np.repeat(denom, n)
    s = np.add.reduceat(w * z, starts)
    return s, w, n


def make_objective(serial, in_mask, out_mask, temp, pos_w, hess_floor):
    in_idx, in_starts = bag_index(serial, in_mask)
    out_idx, out_starts = bag_index(serial, out_mask)
    n_rows = serial.size

    def obj(z_all, _dataset):
        grad = np.zeros(n_rows, dtype=np.float64)
        hess = np.full(n_rows, hess_floor, dtype=np.float64)

        for idx, starts, target, scale in ((in_idx, in_starts, 1.0, 1.0),
                                           (out_idx, out_starts, 0.0, pos_w)):
            if idx.size == 0:
                continue
            z = z_all[idx]
            s, w, n = soft_max_and_weights(z, starts, temp)
            p = sigmoid(s)
            dl_ds = scale * (p - target)                 # BCE on the bag score
            d2l_ds = scale * p * (1.0 - p)
            ds_dz = w * (1.0 + (z - np.repeat(s, n)) / temp)
            grad[idx] += np.repeat(dl_ds, n) * ds_dz
            hess[idx] += np.repeat(d2l_ds, n) * ds_dz ** 2
        return grad, hess

    return obj


def run(dataset: str, seed: int, rounds: int, temp: float, pos_w: float, lr: float):
    import lightgbm as lgb

    splitted = os.path.join(PROJECT_ROOT, "data", "splitted", dataset)
    train_df, val_df, test_df, features = load_dataset(splitted_dir=splitted, model="lgbm")
    sort_cols = ["serial_number", "date"]
    train_df = train_df.sort_values(sort_cols).reset_index(drop=True)
    val_df = val_df.sort_values(sort_cols).reset_index(drop=True)
    test_df = test_df.sort_values(sort_cols).reset_index(drop=True)

    serial = pd.factorize(train_df["serial_number"], sort=False)[0].astype(np.int64)
    rul = train_df["RUL"].to_numpy()
    cens = train_df["censored"].to_numpy() == 1
    in_mask = (~cens) & (rul <= H)
    out_mask = ~in_mask
    print(f"  bags: in={np.unique(serial[in_mask]).size}  out={np.unique(serial[out_mask]).size}"
          f"  rows in/out = {in_mask.sum():,}/{out_mask.sum():,}")

    # Start from the row-level model. Max-pooled bag losses give almost every row
    # zero gradient, so learning the whole tree ensemble from scratch under this
    # objective does not work; what it can do is move an already-trained model.
    y_tr = create_binary_target(train_df)
    base = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              min_child_samples=100, subsample=0.9, subsample_freq=1,
                              colsample_bytree=0.9, random_state=seed, n_jobs=-1,
                              verbose=-1)
    base.fit(train_df[features], y_tr,
             eval_set=[(val_df[features], create_binary_target(val_df))],
             eval_metric="average_precision",
             callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])
    init_tr = base.predict(train_df[features], raw_score=True)
    init_va = base.predict(val_df[features], raw_score=True)
    init_te = base.predict(test_df[features], raw_score=True)
    print(f"  base model: {base.best_iteration_ or 300} trees")

    obj = make_objective(serial, in_mask, out_mask, temp, pos_w, hess_floor=1e-3)
    ds = lgb.Dataset(train_df[features], label=y_tr, init_score=init_tr,
                     free_raw_data=False)

    t0 = time.time()
    # LightGBM >= 4 takes the custom objective through params, not fobj.
    booster = lgb.train(
        {"objective": obj, "learning_rate": lr, "num_leaves": 31,
         "min_child_samples": 100, "bagging_fraction": 0.9, "bagging_freq": 1,
         "feature_fraction": 0.9, "lambda_l2": 10.0,
         "seed": seed, "num_threads": -1, "verbose": -1},
        ds, num_boost_round=rounds,
    )
    secs = round(time.time() - t0, 1)

    p_va = sigmoid(init_va + booster.predict(val_df[features], raw_score=True))
    p_te = sigmoid(init_te + booster.predict(test_df[features], raw_score=True))
    tau = pick_tau_op(val_df, p_va)
    m = judge(test_df, p_te, tau)
    print(f"  [op-loss] tau={tau:.3f}  P={m['precision']:.3f} R={m['recall']:.3f} "
          f"FAR={m['far']:.2%} medLT={m['median_lt']:.0f}  "
          f"(OT {m['OT']} / E {m['E']} / M {m['M']} / CE {m['CE']} / CN {m['CN']})  {secs}s")
    m.update(dataset=dataset, seed=seed, variant="op-loss", tau=tau,
             temp=temp, pos_w=pos_w, rounds=rounds, lr=lr, secs=secs)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HGST_20HUH721212ALN604")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--pos-w", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.1)
    args = ap.parse_args()

    rows = []
    for seed in args.seeds:
        print(f"\n=== {args.dataset}  seed {seed}  (T={args.temp}, pos_w={args.pos_w}) ===",
              flush=True)
        rows.append(run(args.dataset, seed, args.rounds, args.temp, args.pos_w, args.lr))

    out_dir = os.path.join(PROJECT_ROOT, "results", "operational_training")
    os.makedirs(out_dir, exist_ok=True)
    csv = os.path.join(out_dir, f"op_loss_{args.dataset}.csv")
    pd.DataFrame(rows).to_csv(csv, index=False, encoding="utf-8-sig")
    print(f"\ncsv: {csv}")


if __name__ == "__main__":
    main()
