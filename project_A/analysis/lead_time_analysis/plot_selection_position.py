"""Figure 4: a row-level choice is a coin flip.

One sample is one run of the selection experiment -- a single seed in a single
condition (dataset x threshold criterion x row-level selection metric). Its value
is how much the chosen model beats the average of the four candidates, i.e. what
knowing the row-level metric bought over knowing nothing, in the units of the
metric being scored. The upper distribution is the same difference for the best
of the four, which is what was there to be won.

Row-level Precision is recovered from the cached threshold curve, which the sweep
table does not carry; everything else comes from the committed sweep table.
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

SRC = os.path.join(ROOT, 'results', 'lead_time_analysis', 'reports', 'budget_sweep_360runs.csv')
DUMP = os.path.join(ROOT, 'results', 'lead_time_analysis', 'reports', 'selection_per_seed.csv')
OUT = os.path.join(ROOT, 'results', 'lead_time_analysis', 'selection_position.png')
FULL = {'HGST': 'HGST_20HUH721212ALN604', 'Seagate': 'ST12000NM0007',
        'Toshiba': 'TOSHIBA_20MG07ACA14TA'}
SH = ['HGST', 'Seagate', 'Toshiba']
SCORE = [('recall', 'Scored on HDD-level Recall'),
         ('precision', 'Scored on HDD-level Precision')]
C_SEL, C_ORA = '#2b5c8f', '#b0b0b0'

df = pd.read_csv(SRC)
df = df[df.budget == 0.01]


def row_precision(ds, m, seed, thr):
    c = pc.load_row_curve(FULL[ds], m, seed, 'test')
    i = int(np.abs(c['threshold'].to_numpy() - thr).argmin())
    tp, fp = float(c.tp.iloc[i]), float(c.fp.iloc[i])
    return tp / (tp + fp) if tp + fp else 0.0


rows = []
for ds in SH:
    base = df[(df.ds == ds) & (df.crit == 'row')].set_index(['seed', 'model'])
    cache = {(s, m): row_precision(ds, m, s, r.thr) for (s, m), r in base.iterrows()}
    for crit in ['row', 'op']:
        sc = df[(df.ds == ds) & (df.crit == crit)]
        for metric, _ in SCORE:
            for seed in sorted(sc.seed.unique()):
                x = sc[sc.seed == seed].set_index('model')
                if len(x) < 4 or seed not in base.index.get_level_values(0):
                    continue
                b = base.loc[seed]
                if len(b) < 4:
                    continue
                pr = pd.Series({m: cache[(seed, m)] for m in b.index})
                rows.append(dict(ds=ds, crit=crit, metric=metric, seed=seed,
                                 by_rec=x.loc[b.row_rec.idxmax(), metric],
                                 by_prec=x.loc[pr.idxmax(), metric],
                                 rand=x[metric].mean(), best=x[metric].max()))

R = pd.DataFrame(rows)
R.to_csv(DUMP, index=False)

fs.apply()
fig, ax = plt.subplots(2, 1, figsize=(fs.COL_W * 2.05, 3.5), sharex=True)
for a, (metric, title) in zip(ax, SCORE):
    q = R[R.metric == metric]
    sel = np.r_[q.by_rec - q.rand, q.by_prec - q.rand]
    ora = (q.best - q.rand).to_numpy()
    bins = np.linspace(-0.10, 0.10, 41)
    a.hist(np.clip(sel, bins[0], bins[-1]), bins=bins, color=C_SEL, zorder=3)
    a.axvline(0, color='0.2', lw=fs.LW_RULE, zorder=4)
    mo = float(np.median(ora))
    a.axvline(mo, color='#c0392b', lw=fs.LW_RULE, ls=(0, (3, 2)), zorder=4)
    a.annotate(f'best of the four\n(median +{mo:.3f})', xy=(mo, a.get_ylim()[1] * 0.82),
               xytext=(mo + 0.006, a.get_ylim()[1] * 0.92), fontsize=fs.FS_NOTE,
               color='#c0392b', ha='left', va='top', linespacing=1.35)
    a.text(-0.097, a.get_ylim()[1] * 0.92,
           f'{(sel < 0).mean():.0%} of runs land below zero', fontsize=fs.FS_NOTE,
           color='0.3', ha='left', va='top')
    a.set_title(title, fontsize=fs.FS_TITLE, loc='left')
    a.set_ylabel('Runs', fontsize=fs.FS_LABEL)
    fs.style_axes(a)
    print(f"  {metric:10s} sel median {np.median(sel):+.4f}  below 0 {(sel < 0).mean():.0%}"
          f"   best median {np.median(ora):+.4f}")
ax[1].set_xlabel('Gain over choosing at random, in the units of the scored metric',
                 fontsize=fs.FS_LABEL)
ax[0].set_xlim(-0.105, 0.105)
fig.tight_layout(pad=0.4, h_pad=1.0)
fig.savefig(OUT, dpi=fs.DPI, bbox_inches='tight')
print('saved', OUT)
