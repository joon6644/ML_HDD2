"""Figure 3 (continuous): the outcome mix over the whole constraint axis.

The four constraint levels of Table 3 are four points on a curve that is defined
everywhere, because the cached first-alarm steps let the verdict be recomputed at
any threshold. On a failure-observed disk the running maximum only rises, so as
the threshold rises the first alarm moves later and the disk passes through at
most three regimes -- Early, then On-time, then Missed. Each disk therefore
contributes two cut points, and the whole sweep is interval accumulation.

The x axis is the row-level FAR that the threshold produces, so the reader sees
the same constraint the paper selects on.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figure_style as fs

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import prediction_cache as pc

OUT = os.path.join(ROOT, 'results', 'lead_time_analysis', 'constraint_continuous.png')
FULL = {'HGST': 'HGST_20HUH721212ALN604', 'Seagate': 'ST12000NM0007',
        'Toshiba': 'TOSHIBA_20MG07ACA14TA'}
SH = ['HGST', 'Seagate', 'Toshiba']
MD = ['LGBM', 'XGB', 'LSTM', 'GRU']
SEEDS = range(42, 72)
H = 30
SEG = [('OT', 'On-time', '#24506e'), ('E', 'Early', '#8ba9c0'),
       ('M', 'Missed', '#e4e9ee')]
MAIN = 'Seagate'


def outcome_curves(ds, model, seed, T):
    """Share of failed HDDs in each outcome, for every threshold in T."""
    d = pd.read_parquet(pc.disk_steps_path(FULL[ds], model, seed, 'test'))
    d = d[d.failure_date.notna()].sort_values(['serial_number', 'step_date'])
    if d.empty:
        return None
    lt = (pd.to_datetime(d.failure_date) - pd.to_datetime(d.step_date)).dt.days.to_numpy()
    v = d.step_value.to_numpy(float)
    sn = d.serial_number.to_numpy()
    start = np.r_[0, np.flatnonzero(sn[1:] != sn[:-1]) + 1]
    end = np.r_[start[1:], v.size]

    n = start.size
    cut_on = np.empty(n)      # threshold at which the disk becomes On-time
    cut_off = np.empty(n)     # threshold above which it alarms no more
    for i, (s, e) in enumerate(zip(start, end)):
        vi, li = v[s:e], lt[s:e]
        cut_off[i] = vi[-1]
        ok = np.flatnonzero(li <= H)
        # The first step whose lead time is inside the horizon; every threshold
        # above the preceding step's value lands on it or later.
        cut_on[i] = vi[ok[0] - 1] if ok.size and ok[0] > 0 else (
            -np.inf if ok.size else np.inf)

    ot = ((T[None, :] > cut_on[:, None]) & (T[None, :] <= cut_off[:, None])).sum(0)
    al = (T[None, :] <= cut_off[:, None]).sum(0)
    return np.vstack([ot, al - ot, n - al]) / n


res = {}
for ds in SH:
    T = pc.load_row_curve(FULL[ds], 'XGB', 42, 'test')['threshold'].to_numpy()
    acc, far = [], []
    for m in MD:
        for s in SEEDS:
            c = outcome_curves(ds, m, s, T)
            if c is None:
                continue
            acc.append(c)
            rc = pc.load_row_curve(FULL[ds], m, s, 'test')
            far.append((rc.fp / (rc.fp + rc.tn)).to_numpy())
    share = np.mean(acc, axis=0)
    xfar = np.mean(far, axis=0) * 100
    o = np.argsort(xfar)
    res[ds] = (xfar[o], share[:, o])
    print(f'{ds:8s} {len(acc):3d} runs   row FAR {xfar.min():.3f}% .. {xfar.max():.1f}%   '
          f'On-time peak {share[0].max():.1%} at {xfar[share[0].argmax()]:.2f}%')

fs.apply()
fig, a = plt.subplots(figsize=(fs.COL_W, 2.6))
xf, sh = res[MAIN]
a.stackplot(xf, sh[0], sh[1], sh[2], colors=[c for _, _, c in SEG],
            edgecolor='white', linewidth=0.7, zorder=2)
for lv in (0.05, 0.1, 0.5, 1.0, 5.0):
    a.axvline(lv, color='0.45', lw=0.5, ls=(0, (2, 2.5)), alpha=0.6, zorder=5)
a.set_xscale('log')
a.set_xlim(0.02, 5)
a.set_xticks([0.05, 0.1, 0.5, 1, 5])
a.set_xticklabels(['0.05', '0.1', '0.5', '1', '5'], fontsize=fs.FS_TICK)
a.minorticks_off()
a.set_ylim(0, 1)
a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
a.set_yticklabels(['0', '25', '50', '75', '100'], fontsize=fs.FS_TICK)
a.set_ylabel('Share of failed HDDs (%)', fontsize=fs.FS_LABEL)
a.set_xlabel('Row-level FAR at the threshold (%)', fontsize=fs.FS_LABEL)
fs.style_axes(a)
a.grid(False)
a.tick_params(axis='both', length=3.0, width=0.5, color=fs.SPINE)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.965))
h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SEG]
fig.legend(h, [l for _, l, _ in SEG], fontsize=fs.FS_NOTE, frameon=False,
           loc='upper center', bbox_to_anchor=(0.5, 1.005), ncol=3,
           handlelength=1.1, handleheight=0.9, columnspacing=1.5, borderpad=0.0)
fig.savefig(OUT, dpi=fs.DPI, bbox_inches='tight')
print('saved', OUT)
