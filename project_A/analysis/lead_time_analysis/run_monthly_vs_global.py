"""Does the threshold a monthly disk-level evaluation picks match the global peak?

A monthly evaluation asks a bounded question -- of the disks alive this month, which
alarm and which fail -- and the alarm state resets every month. The full-history
evaluation asks about one first alarm over a disk's whole life. The two need not
prefer the same operating point, and whether they do is worth knowing before
claiming that either says anything about the other.

The monthly protocol here follows RODMAN in shape: within a calendar month a disk
alarms if any of its predictions that month crosses the threshold, is positive if
its failure falls in that month, and TPR/FPR are counted over disks. Youden's J
picks the operating point because it needs no cost ratio and is unaffected by the
extreme prevalence.

Rolling windows need the maximum inside a window, which the running-maximum step
cache cannot give, so this runs on the full-prediction cache -- seed 42 only.
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
TAU = np.unique(np.r_[np.geomspace(1e-4, 1.0, 400), 0.5])
OUT = os.path.join(ROOT, "results", "lead_time_analysis", "monthly_vs_global.csv")


def monthly_curve(df):
    """TPR and FPR over disks, pooled across calendar months."""
    df = df.assign(month=df.date.values.astype("datetime64[M]"))
    fail_month = df.groupby("serial_number").failure_date.first()
    fail_month = fail_month.dt.to_period("M").dt.to_timestamp()

    # Per (disk, month): the strongest prediction that month.
    g = df.groupby(["serial_number", "month"], sort=False).pred.max().reset_index()
    g["is_pos"] = g.month.values == g.serial_number.map(fail_month).values

    pos = g.pred[g.is_pos].to_numpy()
    neg = g.pred[~g.is_pos].to_numpy()
    tpr = (pos[:, None] >= TAU[None, :]).sum(0) / max(pos.size, 1)
    fpr = (neg[:, None] >= TAU[None, :]).sum(0) / max(neg.size, 1)
    return tpr, fpr, pos.size, neg.size


def global_curve(df):
    """Operational recall over the whole history, by first alarm."""
    failed = df[df.has_failed].sort_values(["serial_number", "date"])
    rec = np.zeros(TAU.size)
    n = 0
    for _, d in failed.groupby("serial_number", sort=False):
        p = d.pred.to_numpy()
        lt = (d.failure_date.iloc[0] - d.date).dt.days.to_numpy()
        n += 1
        # First alarm is the first crossing; running maximum locates it.
        run = np.maximum.accumulate(p)
        idx = np.searchsorted(run, TAU, "left")
        ok = idx < p.size
        got = np.zeros(TAU.size, bool)
        got[ok] = lt[idx[ok]] <= H
        rec += got
    return rec / max(n, 1)


rows = []
for ds in SHORT:
    for m in MODELS:
        df = pd.read_parquet(pc.full_preds_path(FULL[ds], m, SEED, "test"))
        df["date"] = pd.to_datetime(df.date)
        df["failure_date"] = pd.to_datetime(df.failure_date)

        tpr, fpr, n_pos, n_neg = monthly_curve(df)
        rec = global_curve(df)

        k_month = int(np.argmax(tpr - fpr))
        k_global = int(np.argmax(rec))
        rows.append(dict(
            dataset=ds, model=m, n_pos_month=n_pos, n_neg_month=n_neg,
            tau_month=TAU[k_month], tau_global=TAU[k_global],
            j_month=(tpr - fpr)[k_month], tpr_month=tpr[k_month], fpr_month=fpr[k_month],
            j_at_global=(tpr - fpr)[k_global],
            rec_global=rec[k_global], rec_at_month=rec[k_month]))
        print(f"  {ds:8s} {m:5s} tau month {TAU[k_month]:.4f} / global {TAU[k_global]:.4f}",
              flush=True)

d = pd.DataFrame(rows)
d.to_csv(OUT, index=False)
print(f"\nsaved {OUT}\n")

print("Median over the four models (seed 42)\n")
print(f"{'dataset':9s} {'tau_month':>9s} {'tau_glob':>9s} {'ratio':>6s} | "
      f"{'TPR_mo':>7s} {'FPR_mo':>7s} | {'globRec@mo':>10s} {'globRec@peak':>12s} {'loss':>6s} | "
      f"{'J@peak/J_mo':>11s}")
for ds in SHORT:
    r = d[d.dataset == ds].median(numeric_only=True)
    print(f"{ds:9s} {r.tau_month:9.4f} {r.tau_global:9.4f} {r.tau_global/r.tau_month:6.2f} | "
          f"{r.tpr_month:7.3f} {r.fpr_month:7.4f} | {r.rec_at_month:10.3f} {r.rec_global:12.3f} "
          f"{1 - r.rec_at_month / r.rec_global:6.1%} | {r.j_at_global / r.j_month:11.2f}")

print("\nPer-model tau ratio (global peak / monthly optimum)")
for ds in SHORT:
    x = d[d.dataset == ds]
    print(f"  {ds:9s} " + "  ".join(f"{a.model} {a.tau_global/a.tau_month:.2f}"
                                    for a in x.itertuples()))
