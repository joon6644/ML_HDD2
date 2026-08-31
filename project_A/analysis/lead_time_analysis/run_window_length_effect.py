"""How long an evaluation window has to be before operational recall turns over.

Early means a lead time above H, so a window of W days can only produce it when
W > H: inside a 30-day window every alarm is by construction within 30 days of the
failure it precedes. The turnover that makes recall non-monotone comes entirely
from disks leaving On-time for Early, so a short window cannot show it however the
verdict is defined -- the five categories collapse to three, and recall goes back
to being monotone in the threshold.

This measures where the crossover actually is: for each window length, apply the
same first-alarm verdict to only the last W days of each failure-observed disk and
ask whether recall still has an interior maximum.

Windows need the maximum inside a window rather than from the start of the record,
which the step cache cannot give, so this runs on the full-prediction cache --
seed 42 only.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import prediction_cache as pc  # noqa: E402

FULL = {"HGST": "HGST_20HUH721212ALN604", "Seagate": "ST12000NM0007",
        "Toshiba": "TOSHIBA_20MG07ACA14TA"}
SHORT = ["HGST", "Seagate", "Toshiba"]
MODELS = ["LGBM", "XGB", "LSTM", "GRU"]
SEED = 42
H = 30
WINDOWS = [30, 45, 60, 90, 180, 365, 730, None]      # None = the whole record
TAU = np.geomspace(1e-4, 1.0, 500)
OUT = os.path.join(ROOT, "results", "lead_time_analysis", "window_length_effect.csv")


def recall_curve(disks, W):
    """Operational recall over thresholds, seeing only the last W days of each disk."""
    rec = np.zeros(TAU.size)
    early = np.zeros(TAU.size)
    for lt, p in disks:
        if W is not None:
            keep = lt <= W
            lt, p = lt[keep], p[keep]
            if lt.size == 0:
                continue
        # The first alarm is the earliest crossing, i.e. the largest lead time
        # among rows above the threshold. Sorting by prediction and taking a
        # running maximum of lead time gives that for every threshold at once.
        o = np.argsort(p)[::-1]
        cm = np.maximum.accumulate(lt[o])
        k = np.searchsorted(-p[o], -TAU, "right")      # rows with p >= tau
        hit = k > 0
        first_lt = np.where(hit, cm[np.clip(k - 1, 0, cm.size - 1)], np.nan)
        rec += hit & (first_lt <= H)
        early += hit & (first_lt > H)
    n = len(disks)
    return rec / n, early / n


rows = []
for ds in SHORT:
    for m in MODELS:
        df = pd.read_parquet(pc.full_preds_path(FULL[ds], m, SEED, "test"))
        df = df[df.has_failed]
        df["date"] = pd.to_datetime(df.date)
        df["failure_date"] = pd.to_datetime(df.failure_date)
        df["lt"] = (df.failure_date - df.date).dt.days

        # `g.lt` would resolve to DataFrame.lt, the comparison method.
        disks = [(g["lt"].to_numpy(), g["pred"].to_numpy(float))
                 for _, g in df.groupby("serial_number", sort=False)]

        for W in WINDOWS:
            rec, early = recall_curve(disks, W)
            k = int(np.argmax(rec))
            # TAU ascends, so index 0 is the lowest threshold -- the far end of the
            # sweep, where every disk that can alarm has. An interior maximum means
            # recall has fallen back by the time it gets there.
            drop = rec[k] - rec[0] if rec[k] > 0 else 0.0
            rows.append(dict(dataset=ds, model=m, window=(-1 if W is None else W),
                             peak=rec[k], tau_peak=TAU[k], rec_at_min_tau=rec[0],
                             drop=drop, rel_drop=drop / rec[k] if rec[k] > 0 else 0.0,
                             early_at_min_tau=early[0]))
    print(f"  {ds} done", flush=True)

d = pd.DataFrame(rows)
d.to_csv(OUT, index=False)
print(f"\nsaved {OUT}\n")

cols = [w if w is not None else -1 for w in WINDOWS]
lab = [f"{w}d" if w is not None else "full" for w in WINDOWS]
agg = d.groupby(["dataset", "window"]).median(numeric_only=True)

print("Relative fall of recall from its maximum to the lowest threshold")
print("(0 means recall is still monotone; larger means a clearer turnover)\n")
print(f"{'dataset':9s} " + " ".join(f"{l:>7s}" for l in lab))
for ds in SHORT:
    print(f"{ds:9s} " + " ".join(f"{agg.loc[(ds, c)].rel_drop:6.1%} " for c in cols))

print("\nEarly share among failure-observed HDDs at the lowest threshold")
print(f"{'dataset':9s} " + " ".join(f"{l:>7s}" for l in lab))
for ds in SHORT:
    print(f"{ds:9s} " + " ".join(f"{agg.loc[(ds, c)].early_at_min_tau:6.2f} " for c in cols))
