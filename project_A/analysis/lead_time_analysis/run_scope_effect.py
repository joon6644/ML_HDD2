"""What a restricted test population reports, and what that same setting costs.

Eight of the eleven studies in Table 1 evaluate on something other than the full
observation history: a balanced sample, or a partial one. Recall and FAR are rates
within a population and survive that untouched, but precision, F1 and accuracy do
not -- they move with prevalence. So the operating point that looks best on a
balanced test set is not the operating point that looks best in production, and
the gap is the prevalence ratio, which here is roughly 560:1.

The demonstration needs no reimplementation. Recall is the axis, because recall is
the one headline number every scope agrees on: fix it, and the disagreement is
entirely in what else the evaluation reports. At each recall level this reads ONE
set of predictions four ways -- precision on a balanced population, on a partial
one, on the full one, and the operational verdict.

Fixing recall rather than optimising a threshold also keeps the result off the
cached grid's floor. Balanced F1 peaks below 0.001 for most runs, so "the point a
balanced evaluation would choose" is not resolvable here, while "the point at
which it reports recall 0.7" is.

Precision on a population holding r negatives per positive is TPR / (TPR + r*FPR),
so each scope is one closed-form number: no resampling, no sampling noise, and no
dependence on which negatives a draw happened to pick.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import prediction_cache as pc  # noqa: E402

from run_burden_sweep import FULL, SHORT, MODELS, SEEDS, disk_arrays, H  # noqa: E402

# Negative-to-positive ratio of the evaluated population. `None` means the ratio
# the data actually has.
SCOPES = [("Balanced", 1.0), ("Partial 10:1", 10.0), ("Partial 100:1", 100.0),
          ("Full", None)]
TARGET_RECALL = [0.3, 0.5, 0.7]
OUT = os.path.join(ROOT, "results", "lead_time_analysis", "scope_effect.csv")


def operational_at(fail_steps, cens_vmax, tau):
    """Five-category verdict at one threshold."""
    ot = early = 0
    lts = []
    for v, lt in fail_steps:
        i = np.searchsorted(v, tau, side="left")
        if i >= v.size:
            continue
        lts.append(lt[i])
        if lt[i] <= H:
            ot += 1
        else:
            early += 1
    ce = int((cens_vmax >= tau).sum())
    n_fail, n_cens = len(fail_steps), cens_vmax.size
    return {
        "op_recall": ot / n_fail,
        "op_precision": ot / max(ot + early + ce, 1),
        "burden": ce / n_cens,
        "op_median_lt": float(np.median(lts)) if lts else np.nan,
        "early_share": early / n_fail,
    }


rows = []
for ds in SHORT:
    for model in MODELS:
        for seed in SEEDS:
            rc = pc.load_row_curve(FULL[ds], model, seed, "test")
            tp, fp = rc.tp.to_numpy(float), rc.fp.to_numpy(float)
            fn, tn = rc.fn.to_numpy(float), rc.tn.to_numpy(float)
            thr = rc.threshold.to_numpy(float)
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            r_true = (fp + tn)[0] / (tp + fn)[0]

            fail_steps, cens_vmax = disk_arrays(ds, model, seed)

            for target in TARGET_RECALL:
                # Highest threshold still reaching the target recall. tpr falls as
                # the threshold rises, so this is the last index at or above it.
                hit = np.flatnonzero(tpr >= target)
                if hit.size == 0:
                    continue
                k = int(hit[-1])
                tau = thr[k]
                rec = {"dataset": ds, "model": model, "seed": seed,
                       "target_recall": target, "tau": tau, "row_recall": tpr[k],
                       "row_far": fpr[k], "neg_per_pos": r_true}
                for name, r in SCOPES:
                    rr = r_true if r is None else r
                    rec[f"prec_{name}"] = tpr[k] / (tpr[k] + rr * fpr[k])
                rec.update(operational_at(fail_steps, cens_vmax, tau))
                rows.append(rec)
    print(f"  {ds} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\nsaved {OUT}\n")

names = [n for n, _ in SCOPES]
agg = df.groupby(["dataset", "target_recall"]).median(numeric_only=True)
cov = df.groupby(["dataset", "target_recall"]).size()

print("Row-level recall is held fixed; only the evaluated population changes.")
print("The last four columns are the operational verdict at the SAME threshold.\n")
head = f"{'dataset':9s} {'rowRec':>6s} " + " ".join(f"{n:>13s}" for n in names)
print(head + f" | {'rowFAR':>7s} | {'burden':>7s} {'opRec':>6s} {'opPrec':>7s} {'medLT':>6s} {'runs':>5s}")
for ds in SHORT:
    for t in TARGET_RECALL:
        if (ds, t) not in agg.index:
            continue
        r = agg.loc[(ds, t)]
        cells = " ".join(f"{r[f'prec_{n}']:13.3f}" for n in names)
        print(f"{ds:9s} {t:6.1f} {cells} | {r.row_far*100:6.2f}% | "
              f"{r.burden*100:6.1f}% {r.op_recall:6.3f} {r.op_precision:7.3f} "
              f"{r.op_median_lt:5.0f}d {int(cov[(ds,t)]):5d}")
    print()

print("Negatives per positive row in the data as collected")
for ds in SHORT:
    print(f"  {ds:9s} {df[df.dataset==ds].neg_per_pos.median():8.0f} : 1")
