"""How much of the measured burden a shorter collection period would have hidden.

A first alarm is never withdrawn while the denominator stays fixed at a count of
HDDs, so a right-censored disk that alarms once is charged once no matter how long
it is watched -- but a disk watched for six years has more chances to alarm than
one watched for two. Operational FAR therefore rises with the collection period,
and a dataset that stops earlier reports a lower one for the same model at the
same threshold.

That is a property of the metric, not of any study, and it is what makes values
from datasets of different length incomparable. It is measurable directly: hold
the model and the threshold fixed, truncate every disk's history to its first L
days, and read the burden off the shortened record. Because the cached steps are
running-maximum increases, the maximum inside a disk's first L days is just its
last step at or before that date, so every truncation is exact.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import prediction_cache as pc  # noqa: E402

from run_burden_sweep import FULL, SHORT, MODELS, SEEDS  # noqa: E402

WINDOWS = [180, 365, 730, 1095, 1460, None]   # None = the record as collected
REF_BURDEN = 0.025
OUT = os.path.join(ROOT, "results", "lead_time_analysis", "window_optimism.csv")


def censored_steps(ds, model, seed):
    """Per censored disk: step values and days since that disk's first record."""
    d = pd.read_parquet(pc.disk_steps_path(FULL[ds], model, seed, "test"))
    d = d[~d.has_failed].sort_values(["serial_number", "step_date"], kind="mergesort")
    v = d.step_value.to_numpy(np.float64)
    sn = d.serial_number.to_numpy()
    date = pd.to_datetime(d.step_date).to_numpy()

    start = np.r_[0, np.flatnonzero(sn[1:] != sn[:-1]) + 1]
    end = np.r_[start[1:], v.size]
    out = []
    for s, e in zip(start, end):
        age = (date[s:e] - date[s]).astype("timedelta64[D]").astype(int)
        out.append((v[s:e], age))
    return out


rows = []
for ds in SHORT:
    for model in MODELS:
        for seed in SEEDS:
            disks = censored_steps(ds, model, seed)
            vmax_full = np.array([v[-1] for v, _ in disks])
            n = vmax_full.size

            # Threshold that produces the reference burden over the full record.
            w = np.sort(vmax_full)[::-1]
            tau = w[max(int(np.floor(REF_BURDEN * n)), 1) - 1]

            for L in WINDOWS:
                if L is None:
                    alarmed = int((vmax_full >= tau).sum())
                else:
                    # Running maxima only rise, so the maximum inside the first L
                    # days is the last step recorded at or before day L.
                    alarmed = sum(1 for v, age in disks
                                  if v[np.searchsorted(age, L, "right") - 1] >= tau)
                rows.append({"dataset": ds, "model": model, "seed": seed,
                             "window": -1 if L is None else L, "tau": tau,
                             "alarmed": alarmed, "burden": alarmed / n, "n_cens": n})
    print(f"  {ds} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\nsaved {OUT}\n")

agg = df.groupby(["dataset", "window"]).burden.median().unstack()
cnt = df.groupby(["dataset", "window"]).alarmed.median().unstack()
cols = [w for w in WINDOWS if w is not None] + [-1]
lab = [f"{c}d" if c > 0 else "full" for c in cols]

print("Operational FAR measured over a truncated record, at the threshold that")
print(f"produces {REF_BURDEN*100:.1f}% over the full record (median of 120 runs)\n")
print(f"{'dataset':9s} " + " ".join(f"{l:>8s}" for l in lab))
for ds in SHORT:
    print(f"{ds:9s} " + " ".join(f"{agg.loc[ds, c]*100:7.2f}%" for c in cols))

print("\nAlarming censored HDDs behind those rates")
print(f"{'dataset':9s} " + " ".join(f"{l:>8s}" for l in lab) + "     of")
for ds in SHORT:
    print(f"{ds:9s} " + " ".join(f"{cnt.loc[ds, c]:8.0f}" for c in cols)
          + f"  {df[df.dataset==ds].n_cens.median():6.0f}")

print("\nShare of the full-record burden a truncated record would report")
for ds in SHORT:
    full = agg.loc[ds, -1]
    print(f"  {ds:9s} " + "  ".join(f"{l} {agg.loc[ds, c]/full:.2f}"
                                    for c, l in zip(cols, lab) if c > 0))
