"""Figure 4: the operational rank of the model a row-level metric picks.

One run is one execution of the selection procedure -- train four models on one
seed, read a row-level metric, deploy the winner. With four candidates a coin
flip lands on each rank a quarter of the time and averages rank 2.5, so the
dashed marks are what knowing nothing looks like. Ties share their ranks.
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
OUT = os.path.join(ROOT, 'results', 'lead_time_analysis', 'selection_rank.png')
FULL = {'HGST': 'HGST_20HUH721212ALN604', 'Seagate': 'ST12000NM0007',
        'Toshiba': 'TOSHIBA_20MG07ACA14TA'}
SH = ['HGST', 'Seagate', 'Toshiba']
SCORE = [('recall', 'HDD-level Recall'), ('precision', 'HDD-level Precision')]
RCLR = ['#2b5c8f', '#7ba3c9', '#c8d5e2', '#ececec']

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
                rk = x[metric].rank(ascending=False, method='average')
                rows.append(dict(ds=ds, metric=metric,
                                 r1=rk[b.row_rec.idxmax()], r2=rk[pr.idxmax()]))
O = pd.DataFrame(rows)


def shares(v):
    """Rank shares 1..4; a tied rank such as 1.5 splits evenly over 1 and 2."""
    s = np.zeros(4)
    for r in v:
        lo, hi = int(np.floor(r)), int(np.ceil(r))
        s[lo - 1] += 0.5 if hi != lo else 1.0
        if hi != lo:
            s[hi - 1] += 0.5
    return s / len(v)


fs.apply()
fig, a = plt.subplots(figsize=(fs.FULL_W, 2.5))
lab, y = [], 0
for metric, mname in SCORE:
    for ds in SH:
        q = O[(O.metric == metric) & (O.ds == ds)]
        v = np.r_[q.r1, q.r2]
        s = shares(v)
        left = 0.0
        for k in range(4):
            a.barh(y, s[k], 0.62, left=left, color=RCLR[k], lw=0.4,
                   edgecolor='white', zorder=2)
            if s[k] > 0.07:
                a.text(left + s[k] / 2, y, f'{s[k]*100:.0f}', ha='center', va='center',
                       fontsize=fs.FS_NOTE, color='white' if k < 2 else '0.35', zorder=4)
            left += s[k]
        a.text(1.015, y, f'{v.mean():.2f}', va='center', ha='left',
               fontsize=fs.FS_NOTE, color='0.25')
        lab.append(ds)
        print(f"{mname:20s} {ds:8s} n={v.size}  ranks " +
              " ".join(f"{x*100:4.0f}%" for x in s) + f"  mean {v.mean():.2f}")
        y += 1
    y += 0.7

for x in (0.25, 0.5, 0.75):
    a.axvline(x, color='0.35', lw=fs.LW_RULE, ls=(0, (2, 2)), zorder=3)
a.set_yticks(range(len(lab) + 1))
a.set_ylim(-0.7, y - 0.9)
a.set_yticks([0, 1, 2, 3.7, 4.7, 5.7])
a.set_yticklabels(lab, fontsize=fs.FS_TICK)
a.invert_yaxis()
a.set_xlim(0, 1)
a.set_xticks([0, 0.25, 0.5, 0.75, 1])
a.set_xticklabels(['0', '25', '50', '75', '100%'], fontsize=fs.FS_TICK)
a.set_xlabel('Share of runs, by the operational rank of the model that was picked',
             fontsize=fs.FS_LABEL)
a.text(-0.075, 1.0, 'Scored on\n' + SCORE[0][1], transform=a.get_yaxis_transform(),
       fontsize=fs.FS_NOTE, ha='right', va='center', color='0.25', linespacing=1.35)
a.text(-0.075, 4.7, 'Scored on\n' + SCORE[1][1], transform=a.get_yaxis_transform(),
       fontsize=fs.FS_NOTE, ha='right', va='center', color='0.25', linespacing=1.35)
a.text(1.015, -0.62, 'mean\nrank', va='center', ha='left', fontsize=fs.FS_NOTE,
       color='0.25', linespacing=1.3)
h = [plt.Rectangle((0, 0), 1, 1, color=RCLR[k]) for k in range(4)]
a.legend(h, ['best of the four', '2nd', '3rd', 'worst'], fontsize=fs.FS_NOTE,
         frameon=False, ncol=4, loc='lower left', bbox_to_anchor=(0.0, 1.0),
         handlelength=1.1, handleheight=0.85, columnspacing=1.2, borderpad=0.0)
fs.style_axes(a)
a.tick_params(axis='y', length=0)
fig.tight_layout(pad=0.4)
fig.savefig(OUT, dpi=fs.DPI, bbox_inches='tight')
print('saved', OUT)
