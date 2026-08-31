"""Operating-characteristic sweep over the inspection-burden axis.

The verdict of an HDD is fixed by a single first alarm, so the whole five-category
verdict can be recomputed from the cached step curves at any threshold -- no
inference, no retraining. That makes the four constraint levels of the old table
four points on a curve that is defined everywhere.

The x axis is the Early Alarm Rate: the share of ALL evaluated HDDs whose first
alarm could not have started a useful response, being Early plus Censored Early
over the whole fleet. It is deliberately not a false-alarm rate over a negative
class -- the verdict scheme keeps right-censored HDDs as their own category
precisely because censoring does not mean healthy, so using them as the negative
population would contradict it. Every HDD is one HDD to an operator, and EAR asks
only what share of them was disturbed for nothing.

Thresholds are not comparable across models (each model puts its probabilities on
its own scale), so EAR is what puts four models on one axis. It is monotone in the
threshold -- Early is absorbing as the threshold falls, and Censored Early only
grows -- so it orders the sweep. Runs are pooled by taking the median across the
4 models x 30 seeds, with the interquartile range as the band.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import prediction_cache as pc  # noqa: E402

FULL = {
    "HGST": "HGST_20HUH721212ALN604",
    "Seagate": "ST12000NM0007",
    "Toshiba": "TOSHIBA_20MG07ACA14TA",
}
SHORT = ["HGST", "Seagate", "Toshiba"]
MODELS = ["LGBM", "XGB", "LSTM", "GRU"]
SEEDS = range(42, 72)
H = 30

# Common EAR grid the per-run curves are interpolated onto. Its lower end is an
# order of magnitude under the tightest of the old four constraint levels; the
# upper end is past where all three datasets have already turned over.
EAR_GRID = np.geomspace(0.0005, 0.35, 120)
PCTL = [10, 25, 50, 75, 90]

OUT_DIR = os.path.join(ROOT, "results", "lead_time_analysis")
OUT_CURVES = os.path.join(OUT_DIR, "burden_sweep_curves.parquet")
OUT_RUNS = os.path.join(OUT_DIR, "burden_sweep_runs.parquet")


def disk_arrays(ds, model, seed):
    """Per-disk step values and lead times, split by failure observation."""
    d = pd.read_parquet(pc.disk_steps_path(FULL[ds], model, seed, "test"))
    d = d.sort_values(["serial_number", "step_date"], kind="mergesort")

    v = d.step_value.to_numpy(np.float64)
    sn = d.serial_number.to_numpy()
    failed = d.has_failed.to_numpy(bool)
    lt = (pd.to_datetime(d.failure_date) - pd.to_datetime(d.step_date)).dt.days.to_numpy(np.float64)

    start = np.r_[0, np.flatnonzero(sn[1:] != sn[:-1]) + 1]
    end = np.r_[start[1:], v.size]
    is_failed = failed[start]

    # Steps are running-maximum increases, so v is strictly increasing within a
    # disk and lt is non-increasing: raising the threshold moves the first alarm
    # later, never earlier.
    fail_steps = [(v[s:e], lt[s:e]) for s, e, f in zip(start, end, is_failed) if f]
    cens_vmax = np.array([v[e - 1] for s, e, f in zip(start, end, is_failed) if not f])
    return fail_steps, cens_vmax


def sweep_thresholds(fail_steps, cens_vmax):
    """Every threshold at which some disk's verdict changes.

    A verdict only moves where a running maximum sits, so the distinct step values
    across the whole fleet are the complete set of interesting thresholds and
    anything between two of them is the same evaluation twice. Sweeping the values
    themselves rather than a grid of target rates keeps the low end honest: with a
    few hundred disks, quantising a target onto counts collapses a decade of
    targets onto the same one or two alarms and invents a floor the data does not
    have.
    """
    vals = np.unique(np.concatenate([cens_vmax] + [v for v, _ in fail_steps]))
    if vals.size > 600:
        idx = np.unique(np.linspace(0, vals.size - 1, 600).astype(int))
        vals = vals[idx]
    return np.r_[vals, np.nextafter(vals[-1], np.inf)][::-1]


def evaluate(fail_steps, cens_vmax, tau):
    """Five-category verdict and its metrics at every threshold in `tau`."""
    nt = tau.size
    n_fail = len(fail_steps)
    n_cens = cens_vmax.size

    ot = np.zeros(nt, np.int64)
    early = np.zeros(nt, np.int64)
    lt_sum = [[] for _ in range(nt)]

    for v, lt in fail_steps:
        idx = np.searchsorted(v, tau, side="left")
        alarms = idx < v.size
        if not alarms.any():
            continue
        lt_at = np.full(nt, np.nan)
        lt_at[alarms] = lt[idx[alarms]]
        on = alarms & (lt_at <= H)
        ot += on
        early += alarms & ~on
        for j in np.flatnonzero(alarms):
            lt_sum[j].append(lt_at[j])

    missed = n_fail - ot - early
    ce = (cens_vmax[:, None] >= tau[None, :]).sum(0)
    cn = n_cens - ce

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = ot / np.maximum(n_fail, 1)
        precision = ot / np.maximum(ot + early + ce, 1)
        far = ce / np.maximum(n_cens, 1)

    pct = np.full((nt, len(PCTL)), np.nan)
    for j in range(nt):
        if lt_sum[j]:
            pct[j] = np.percentile(lt_sum[j], PCTL)

    n_all = n_fail + n_cens
    out = {
        "ot": ot / n_fail, "early": early / n_fail, "missed": missed / n_fail,
        "alarm_share": (ot + early + ce) / n_all,
        "ear": (early + ce) / n_all,
        "recall": recall, "precision": precision, "cens_far": far,
    }
    for k, p in enumerate(PCTL):
        out[f"lt_p{p}"] = pct[:, k]
    return out


VALUE_COLS = (["recall", "precision", "ot", "early", "missed", "alarm_share",
               "cens_far"] + [f"lt_p{p}" for p in PCTL])

rows, zeros = [], []
for ds in SHORT:
    for model in MODELS:
        for seed in SEEDS:
            path = pc.disk_steps_path(FULL[ds], model, seed, "test")
            if not os.path.exists(path):
                raise FileNotFoundError(f"missing step cache: {path}")
            fail_steps, cens_vmax = disk_arrays(ds, model, seed)
            res = evaluate(fail_steps, cens_vmax, sweep_thresholds(fail_steps, cens_vmax))

            # Ties make the same EAR appear twice; keep the last, which is the
            # lowest threshold reaching it.
            x = res["ear"]
            keep = np.r_[np.flatnonzero(np.diff(x) > 0), x.size - 1]
            rec = {"dataset": ds, "model": model, "seed": seed,
                   "ear_grid": EAR_GRID, "n_cens": cens_vmax.size,
                   "n_fail": len(fail_steps)}
            for c in VALUE_COLS:
                rec[c] = np.interp(EAR_GRID, x[keep], res[c][keep],
                                   left=np.nan, right=np.nan)
            rows.append(pd.DataFrame(rec))
            # The tightest point of all: no censored disk alarms.
            zeros.append({"dataset": ds, "model": model, "seed": seed,
                          **{c: res[c][0] for c in VALUE_COLS}})
        print(f"  {ds:8s} {model:5s} done", flush=True)

runs = pd.concat(rows, ignore_index=True).dropna(subset=VALUE_COLS)
os.makedirs(OUT_DIR, exist_ok=True)
runs.to_parquet(OUT_RUNS, index=False)

g = runs.groupby(["dataset", "ear_grid"])
curves = g[VALUE_COLS].median()
curves.columns = [f"{c}_med" for c in curves.columns]
for c in ["recall", "precision", "lt_p50"]:
    curves[f"{c}_q1"] = g[c].quantile(0.25)
    curves[f"{c}_q3"] = g[c].quantile(0.75)
curves["n_runs"] = g.size()
curves = curves[curves.n_runs >= 60].reset_index()
curves.to_parquet(OUT_CURVES, index=False)

print(f"\nsaved {OUT_RUNS}")
print(f"saved {OUT_CURVES}")
print(f"{len(runs)} run-points, {curves.n_runs.min()}-{curves.n_runs.max()} runs per grid point")

for ds in SHORT:
    c = curves[curves.dataset == ds].sort_values("ear_grid")
    peak = c.loc[c.recall_med.idxmax()]
    under = c[c.lt_p50_med <= H]
    print(f"\n{ds}   ({int(c.n_runs.min())}-{int(c.n_runs.max())} runs/point)")
    print(f"  EAR range             {c.ear_grid.min()*100:.3f}% .. {c.ear_grid.max()*100:.1f}%")
    print(f"  operational recall    peak {peak.recall_med:.3f} at EAR {peak.ear_grid*100:.2f}%"
          f"   (median LT there {peak.lt_p50_med:.0f} d)")
    print(f"  precision             {c.precision_med.max():.3f} .. {c.precision_med.min():.3f}")
    if under.empty:
        lo = c.loc[c.lt_p50_med.idxmin()]
        print(f"  median LT <= H        never on the grid "
              f"(min {lo.lt_p50_med:.0f} d at EAR {lo.ear_grid*100:.3f}%)")
    else:
        top = under.loc[under.ear_grid.idxmax()]
        print(f"  median LT <= H        EAR <= {top.ear_grid*100:.2f}%  "
              f"(recall there {top.recall_med:.3f})")
    # What doubling the wasted inspections buys.
    for lo_t, hi_t in [(0.025, 0.05), (0.05, 0.10)]:
        a = c.loc[(c.ear_grid - lo_t).abs().idxmin()]
        b = c.loc[(c.ear_grid - hi_t).abs().idxmin()]
        print(f"  EAR {a.ear_grid*100:5.2f}% -> {b.ear_grid*100:5.2f}%   "
              f"recall {a.recall_med:.3f} -> {b.recall_med:.3f}   "
              f"median LT {a.lt_p50_med:.0f} -> {b.lt_p50_med:.0f} d")
    print(f"  median LT             {c.lt_p50_med.min():.0f} .. {c.lt_p50_med.max():.0f} d"
          f"   |  p10 {c.lt_p10_med.min():.0f} .. {c.lt_p10_med.max():.0f} d")
