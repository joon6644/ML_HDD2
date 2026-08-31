"""What choosing a wider prediction horizon costs, and what it buys.

H is not fitted; it states how far ahead of failure an alarm can still start a
useful response, which is a property of the operation rather than of the model.
Setting it wider admits earlier alarms as successes, so the reported numbers rise
without the predictions having changed. Quantifying that is what lets someone
choose H knowingly instead of choosing it for the number it produces.

The operating point is fixed by the early alarm rate -- at the levels Table 3
reports -- so every horizon is read off the same set of alarms at matched
operational cost. EAR itself is defined under the deployed H, which is what an
operator would have set the operating point with.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import prediction_cache as pc  # noqa: E402

from run_burden_sweep import FULL, SHORT, MODELS, SEEDS, H, disk_arrays  # noqa: E402

EARS = [0.01, 0.05]
# The first five are the interval widths Table 1's studies actually use; the rest
# extend past any of them.
HORIZONS = [5, 7, 10, 15, 30, 60, 90, 180, 365]
OUT = os.path.join(ROOT, "results", "lead_time_analysis", "horizon_sensitivity.csv")

rows = []
for ds in SHORT:
    for model in MODELS:
        for seed in SEEDS:
            fail_steps, cens_vmax = disk_arrays(ds, model, seed)
            n_all = len(fail_steps) + cens_vmax.size
            # EAR is monotone in the threshold, so the level is reached by
            # scanning the thresholds at which any verdict changes.
            grid = np.unique(np.concatenate([cens_vmax] + [v for v, _ in fail_steps]))[::-1]
            ear = np.array([
                (sum(1 for v, lt in fail_steps
                     if (i := np.searchsorted(v, t, "left")) < v.size and lt[i] > H)
                 + int((cens_vmax >= t).sum())) / n_all for t in grid])
            for b in EARS:
                hit = np.flatnonzero(ear <= b)
                if hit.size == 0:
                    continue
                tau = grid[hit[-1]]
                lts = []
                for v, lt in fail_steps:
                    i = np.searchsorted(v, tau, side="left")
                    if i < v.size:
                        lts.append(lt[i])
                if not lts:
                    continue
                lts = np.asarray(lts)
                rec = {"dataset": ds, "model": model, "seed": seed, "burden": b,
                       "n_alarmed": lts.size, "n_fail": len(fail_steps),
                       "ear": float(ear[hit[-1]])}
                for hz in HORIZONS:
                    # Share of ALARMED failed disks inside the horizon, and share
                    # of ALL failed disks -- the second is operational recall at
                    # that horizon.
                    rec[f"within_{hz}"] = float((lts <= hz).mean())
                    rec[f"recall_{hz}"] = float((lts <= hz).sum() / len(fail_steps))
                rows.append(rec)
    print(f"  {ds} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\nsaved {OUT}\n")

agg = df.groupby(["dataset", "burden"]).median(numeric_only=True)
print("Share of ALARMED failure-observed HDDs whose first alarm falls inside H")
print(f"{'dataset':9s} {'burden':>7s} " + " ".join(f"H={h:<4d}" for h in HORIZONS))
for (ds, b), r in agg.iterrows():
    print(f"{ds:9s} {b*100:6.1f}% " + " ".join(f"{r[f'within_{h}']:6.2f}" for h in HORIZONS))

print("\nOperational recall if H were widened (denominator: all failure-observed HDDs)")
print(f"{'dataset':9s} {'burden':>7s} " + " ".join(f"H={h:<4d}" for h in HORIZONS))
for (ds, b), r in agg.iterrows():
    print(f"{ds:9s} {b*100:6.1f}% " + " ".join(f"{r[f'recall_{h}']:6.3f}" for h in HORIZONS))
