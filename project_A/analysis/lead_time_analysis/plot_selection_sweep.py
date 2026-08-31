"""Figure 4: who can pick the right model, at every operating point.

For each false-alarm constraint the models are ranked by the operational metric,
and two selection rules are scored against that ranking: one that reads the
row-level metric of the same run, and one that reads the operational metric of
the other 29 seeds. Rank 1 is the best of the four; a pick made at random
averages 2.5, which is exact arithmetic rather than an estimate.

Reads the sweep table committed under results/, plus the cached threshold curves
for row-level Precision, which the table does not carry.
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
OUT = os.path.join(ROOT, 'results', 'lead_time_analysis', 'selection_sweep.png')
FULL = {'HGST': 'HGST_20HUH721212ALN604', 'Seagate': 'ST12000NM0007',
        'Toshiba': 'TOSHIBA_20MG07ACA14TA'}
SH = ['HGST', 'Seagate', 'Toshiba']
SCORE = [('recall', 'Operational Recall'), ('precision', 'Operational Precision')]
L = [0.001, 0.005, 0.01, 0.05]
XT = ['0.1', '0.5', '1', '5']
C_ROW, C_OP = '#d95f02', '#2b5c8f'

df = pd.read_csv(SRC)
df = df[df.crit == 'row']


def row_precision(ds, m, seed, thr):
    c = pc.load_row_curve(FULL[ds], m, seed, 'test')
    i = int(np.abs(c['threshold'].to_numpy() - thr).argmin())
    tp, fp = float(c.tp.iloc[i]), float(c.fp.iloc[i])
    return tp / (tp + fp) if tp + fp else 0.0


B = 2000
rng = np.random.default_rng(0)
res = []
for ds in SH:
    for metric, _ in SCORE:
        for lv in L:
            g = df[(df.ds == ds) & (df.budget == lv)]
            seeds = [s for s in sorted(g.seed.unique()) if len(g[g.seed == s]) == 4]
            val = {s: g[g.seed == s].set_index('model')[metric] for s in seeds}
            rk = {s: val[s].rank(ascending=False, method='average') for s in seeds}
            row_hits, op_hits = [], []
            for s in seeds:
                x = g[g.seed == s].set_index('model')
                # Pair like with like: the row-level counterpart of the metric
                # this panel scores on.
                if metric == 'recall':
                    pick = x.row_rec.idxmax()
                else:
                    pick = pd.Series({m: row_precision(ds, m, s, x.loc[m].thr)
                                      for m in x.index}).idxmax()
                row_hits.append(rk[s][pick])
                other = [o for o in seeds if o != s]
                op_hits.append(rk[s][pd.concat([val[o] for o in other], axis=1)
                                     .mean(axis=1).idxmax()])
            rowg = {s: v for s, v in zip(seeds, row_hits)}
            opg = {s: v for s, v in zip(seeds, op_hits)}
            pool = {s: rk[s].to_numpy() for s in seeds}
            rb, ob, nb = np.empty(B), np.empty(B), np.empty(B)
            for b in range(B):
                pick = rng.choice(seeds, size=len(seeds), replace=True)
                rb[b] = np.mean([rowg[s] for s in pick])
                ob[b] = np.mean([opg[s] for s in pick])
                nb[b] = np.mean([rng.choice(pool[s]) for s in pick])
            res.append(dict(ds=ds, metric=metric, lv=lv,
                            row=float(np.mean(row_hits)), op=float(np.mean(op_hits)),
                            rlo=np.percentile(rb, 2.5), rhi=np.percentile(rb, 97.5),
                            olo=np.percentile(ob, 2.5), ohi=np.percentile(ob, 97.5),
                            nlo=np.percentile(nb, 2.5), nhi=np.percentile(nb, 97.5)))
D = pd.DataFrame(res)
print(D.pivot_table(index=['metric', 'ds'], columns='lv', values=['row', 'op'])
       .round(2).to_string())

fs.apply()
fig, ax = plt.subplots(2, 3, figsize=(fs.FULL_W, 3.6), sharex=True, sharey=True)
x = np.arange(len(L))
for i, (metric, mname) in enumerate(SCORE):
    for j, ds in enumerate(SH):
        a = ax[i, j]
        q = D[(D.metric == metric) & (D.ds == ds)].sort_values('lv')
        a.fill_between(x, q.nlo, q.nhi, color='#ededed', lw=0, zorder=1)
        a.axhline(2.5, color='0.45', lw=fs.LW_RULE, ls=(0, (2.5, 2)), zorder=2)
        for col, lo, hi, c in ((q.row, q.rlo, q.rhi, C_ROW),
                               (q.op, q.olo, q.ohi, C_OP)):
            a.fill_between(x, lo, hi, color=c, alpha=0.18, lw=0, zorder=3)
            a.plot(x, col, color=c, lw=fs.LW_DATA, marker='o', ms=2.6, zorder=4)
        a.set_xticks(x)
        a.set_xticklabels(XT, fontsize=fs.FS_TICK)
        a.set_xlim(-0.35, len(L) - 0.65)
        fs.style_axes(a)
        if i == 0:
            a.set_title(ds, fontsize=fs.FS_TITLE)
        if j == 0:
            a.set_ylabel(f'Mean rank\n(scored on {mname.split()[-1]})',
                         fontsize=fs.FS_LABEL,
                         linespacing=1.4)
        if i == 1:
            a.set_xlabel('Row-level FAR constraint (%)', fontsize=fs.FS_LABEL)

ax[0, 0].set_ylim(1.0, 4.0)
ax[0, 0].invert_yaxis()
ax[0, 0].set_yticks([1, 2, 3, 4])
fig.tight_layout(pad=0.4, h_pad=0.7, w_pad=0.8, rect=(0, 0, 1, 0.945))
h = [plt.Line2D([], [], color=c, lw=fs.LW_DATA, marker='o', ms=2.8) for c in (C_ROW, C_OP)]
h.append(plt.Rectangle((0, 0), 1, 1, facecolor='#ededed', lw=0))
fig.legend(h, ['chosen by row-level metric', 'chosen by operational metric',
               'random selection'], fontsize=fs.FS_NOTE, frameon=False,
           loc='upper center', bbox_to_anchor=(0.5, 1.005), ncol=3,
           handlelength=1.6, handleheight=0.9, columnspacing=1.5, borderpad=0.0)
fig.savefig(OUT, dpi=fs.DPI, bbox_inches='tight')
print('saved', OUT)
