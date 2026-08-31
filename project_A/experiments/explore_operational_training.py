"""Can training be aligned with the operational criterion? (exploratory, not in the paper)

The operational verdict hinges on t_alarm = min{t : p_{d,t} >= tau}. That min is
neither differentiable nor decomposable per row, so there is no direct objective
for a tree model. What *is* reachable is sample weighting, and one asymmetry the
min operator creates is worth targeting:

  On a failure-observed disk, a single crossing far before failure destroys the
  disk's On-time chance outright -- the later, better-timed scores never get
  looked at. Row-level BCE cannot see this. It charges the same price for a
  false positive at RUL=400 and one at RUL=35.

So the hypothesis is that the low HDD-level Recall in the paper is partly a
training artifact, fixable without touching the model or the metric.

Three weighting variants, all expressible as LightGBM sample_weight:

  base     uniform (what the paper trains)
  perdisk  w = 1/n_d, so every HDD contributes equal mass -- matches the
           population the HDD-level metrics actually aggregate over. Without it
           a 2,500-day disk outvotes a 900-day disk nearly 3:1 in the loss while
           counting once in the evaluation.
  early    on failure-observed disks, negatives with RUL > H are upweighted by
           ALPHA. These are exactly the rows whose false positives become Early.
  both     perdisk x early

Judgement is computed here rather than imported: with RUL present, the first
alarm row's RUL *is* LT_d, and load_dataset() has already trimmed censored rows
with RUL < H, so any surviving censored alarm is Censored Early by construction.

Usage
  python experiments/explore_operational_training.py                    # seed 42, HGST
  python experiments/explore_operational_training.py --seeds 42 43 44
  python experiments/explore_operational_training.py --dataset ST12000NM0007
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

H = config.TARGET_LEAD_TIME          # 30 d, same horizon the paper judges with
MAX_DISK_FAR = config.MAX_DISK_FAR   # 1%
ALPHA = 5.0                          # 'early' upweight; swept below if you want
VARIANTS = ["base", "perdisk", "early", "both"]


# ---------------------------------------------------------------- weighting ---
def build_weights(df: pd.DataFrame, y: np.ndarray, variant: str) -> np.ndarray:
    """Per-row training weight. Mean-normalised so the four variants see the
    same total loss mass and the learning rate stays comparable."""
    w = np.ones(len(df), dtype=np.float64)

    if variant in ("perdisk", "both"):
        # Rows per disk, aligned back to row order. groupby.transform keeps the
        # index so this is safe against the frame not being sorted by serial.
        n_d = df.groupby("serial_number")["RUL"].transform("size").to_numpy()
        w *= 1.0 / n_d

    if variant in ("early", "both"):
        # Negatives on failure-observed disks that sit outside the On-time
        # window. A crossing here is what turns a detectable disk into Early.
        pre_window = (df["censored"].to_numpy() == 0) & (df["RUL"].to_numpy() > H) & (y == 0)
        w[pre_window] *= ALPHA

    return w / w.mean()


# ---------------------------------------------------------------- judgement ---
def judge(df: pd.DataFrame, p: np.ndarray, tau: float) -> dict:
    """Five operational outcomes for one (model, threshold), computed per HDD.

    A disk's verdict comes from its *first* row above tau in time order, so the
    frame must be sorted by (serial_number, date) before calling this.
    """
    alarm = p >= tau
    serial = df["serial_number"].to_numpy()
    rul = df["RUL"].to_numpy()
    censored = df["censored"].to_numpy()

    # First alarm per disk: take the first True in each contiguous serial block.
    first = pd.DataFrame({"s": serial, "a": alarm, "rul": rul, "c": censored})
    alarmed = first[first["a"]].groupby("s", sort=False).first()
    disk_cens = first.groupby("s", sort=False)["c"].first()

    OT = E = M = CE = CN = 0
    lead_times = []
    for s, c in disk_cens.items():
        if s in alarmed.index:
            lt = float(alarmed.at[s, "rul"])
            if c == 1:
                CE += 1                      # guaranteed LT > H (tail already trimmed)
            elif lt <= H:
                OT += 1
                lead_times.append(lt)
            else:
                E += 1
                lead_times.append(lt)
        else:
            CN += 1 if c == 1 else 0
            M += 1 if c == 0 else 0

    denom_p, denom_r, denom_f = OT + E + CE, OT + E + M, CE + CN
    return {
        "OT": OT, "E": E, "M": M, "CE": CE, "CN": CN,
        "precision": OT / denom_p if denom_p else np.nan,
        "recall": OT / denom_r if denom_r else np.nan,
        "far": CE / denom_f if denom_f else np.nan,
        "median_lt": float(np.median(lead_times)) if lead_times else np.nan,
    }


def pick_tau_op(df: pd.DataFrame, p: np.ndarray) -> float:
    """Smallest threshold on the validation split whose HDD-level FAR <= 1%.

    Same 0.001 grid and same fallback as the paper: if nothing satisfies the
    constraint, keep the threshold that violates it least rather than dropping
    the run.
    """
    grid = np.arange(0.001, 1.000, 0.001)
    best_tau, best_far = grid[-1], np.inf
    for tau in grid:
        far = judge(df, p, tau)["far"]
        if np.isnan(far):
            continue
        if far <= MAX_DISK_FAR:
            return float(tau)
        if far < best_far:
            best_far, best_tau = far, float(tau)
    return best_tau


# ------------------------------------------------------------------- driver ---
def run_seed(dataset: str, seed: int, rounds: int) -> pd.DataFrame:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    splitted = os.path.join(PROJECT_ROOT, "data", "splitted", dataset)
    train_df, val_df, test_df, features = load_dataset(splitted_dir=splitted, model="lgbm")

    # judge() walks disks in time order; sort once and reuse.
    sort_cols = ["serial_number", "date"]
    val_df = val_df.sort_values(sort_cols).reset_index(drop=True)
    test_df = test_df.sort_values(sort_cols).reset_index(drop=True)

    y_tr = create_binary_target(train_df)
    y_va = create_binary_target(val_df)

    rows = []
    for variant in VARIANTS:
        t0 = time.time()
        w = build_weights(train_df, y_tr, variant)

        # Paper's 4.1 hyperparameters, unchanged -- the only thing that moves
        # between variants is sample_weight.
        model = LGBMClassifier(
            n_estimators=rounds, learning_rate=0.05, num_leaves=31,
            min_child_samples=100, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.9, random_state=seed, n_jobs=-1, verbose=-1,
        )
        model.fit(
            train_df[features], y_tr, sample_weight=w,
            eval_set=[(val_df[features], y_va)], eval_metric="average_precision",
            callbacks=[early_stopping(20, verbose=False), log_evaluation(0)],
        )

        p_va = model.predict_proba(val_df[features])[:, 1]
        p_te = model.predict_proba(test_df[features])[:, 1]
        tau = pick_tau_op(val_df, p_va)
        m = judge(test_df, p_te, tau)
        m.update(dataset=dataset, seed=seed, variant=variant, tau=tau,
                 trees=model.best_iteration_ or rounds, secs=round(time.time() - t0, 1))
        rows.append(m)
        print(f"  [{variant:<7}] tau={tau:.3f}  P={m['precision']:.3f} R={m['recall']:.3f} "
              f"FAR={m['far']:.2%} medLT={m['median_lt']:.0f}  "
              f"(OT {m['OT']} / E {m['E']} / M {m['M']} / CE {m['CE']} / CN {m['CN']})  "
              f"{m['secs']}s", flush=True)
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
        frames.append(run_seed(args.dataset, seed, args.rounds))
    out = pd.concat(frames, ignore_index=True)

    out_dir = os.path.join(PROJECT_ROOT, "results", "operational_training")
    os.makedirs(out_dir, exist_ok=True)
    csv = os.path.join(out_dir, f"weight_variants_{args.dataset}.csv")
    out.to_csv(csv, index=False, encoding="utf-8-sig")

    cols = ["precision", "recall", "far", "median_lt", "OT", "E", "M", "CE"]
    print("\n" + "=" * 78)
    print(f"median over {len(args.seeds)} seed(s), tau_op, {args.dataset}")
    print(out.groupby("variant", sort=False)[cols].median().to_string())
    print("=" * 78)
    print(f"csv: {csv}")


if __name__ == "__main__":
    main()
