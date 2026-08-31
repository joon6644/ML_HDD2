"""Figure 4: is a row-level metric better than knowing nothing?

One run is one execution of the selection procedure -- train four models on one
seed, read a row-level metric, deploy the winner -- scored by the operational
rank of the model it picked. With four candidates a pick made at random averages
rank 2.5, so that line is what carrying no information looks like; it is exact
arithmetic, not an estimate. Intervals resample seeds as clusters, since the two
threshold criteria and the two row-level metrics of one seed move together.
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
OUT = os.path.join(ROOT, 'results', 'lead_time_analysis', 'selection_rank_ci.png')
FULL = {'HGST': 'HGST_20HUH721212ALN604', 'Seagate': 'ST12000NM0007',
        'Toshiba': 'TOSHIBA_20MG07ACA14TA'}
SH = ['HGST', 'Seagate', 'Toshiba']
SCORE = [('recall', 'Operational Recall'), ('precision', 'Operational Precision')]
B = 4000

df = pd.read_csv(SRC)
df = df[df.budget == 0.01]


H = 30
MODELS = ['LGBM', 'XGB', 'LSTM', 'GRU']


def row_precision(ds, m, seed, thr):
    c = pc.load_row_curve(FULL[ds], m, seed, 'test')
    i = int(np.abs(c['threshold'].to_numpy() - thr).argmin())
    tp, fp = float(c.tp.iloc[i]), float(c.fp.iloc[i])
    return tp / (tp + fp) if tp + fp else 0.0


def disk_metrics(ds, model, seed, thr):
    """HDD-level Recall and Precision on test at one threshold."""
    d = pd.read_parquet(pc.disk_steps_path(FULL[ds], model, seed, 'test'))
    d = d.sort_values(['serial_number', 'step_date'])
    v = d.step_value.to_numpy(float)
    sn = d.serial_number.to_numpy()
    fd, sd = pd.to_datetime(d.failure_date), pd.to_datetime(d.step_date)
    failed = fd.notna().to_numpy()
    dtf = np.where(failed, (fd - sd).dt.days.to_numpy(), -1).astype(float)
    first = np.r_[True, sn[1:] != sn[:-1]]
    prev = np.where(first, -1.0, np.r_[0.0, v[:-1]])
    fire = (v >= thr) & (prev < thr)
    out = {}
    for s, f, dd, fl in zip(sn, fire, dtf, failed):
        c = out.setdefault(s, [fl, False, np.nan])
        if f and not c[1]:
            c[1], c[2] = True, dd
    fl = np.array([out[s][0] for s in out])
    al = np.array([out[s][1] for s in out])
    dd = np.array([out[s][2] for s in out])
    OT = int((fl & al & (dd <= H)).sum())
    E = int((fl & al & (dd > H)).sum())
    CE = int(((~fl) & al).sum())
    return OT / fl.sum(), (OT / (OT + E + CE) if OT + E + CE else 0.0)


rows = []
POOL = {}          # (ds, metric, seed) -> the four models' ranks in that run
for ds in SH:
    g = df[(df.ds == ds) & (df.crit == 'row')]
    for metric, _ in SCORE:
        for seed in sorted(g.seed.unique()):
            x = g[g.seed == seed].set_index('model')
            if len(x) < 4:
                continue
            pr = pd.Series({m: row_precision(ds, m, seed, x.loc[m].thr) for m in x.index})
            rk = x[metric].rank(ascending=False, method='average')
            rows.append(dict(ds=ds, metric=metric, seed=seed,
                             r1=rk[x.row_rec.idxmax()], r2=rk[pr.idxmax()]))
            POOL.setdefault((ds, metric, seed), []).append(rk.to_numpy())
O = pd.DataFrame(rows)

rng = np.random.default_rng(0)
res = []
for metric, mname in SCORE:
    for ds in SH:
        q = O[(O.metric == metric) & (O.ds == ds)]
        g = {sd: np.r_[h.r1, h.r2] for sd, h in q.groupby('seed')}
        keys = np.array(list(g))
        obs = float(np.mean(np.concatenate([g[k] for k in keys])))
        bs = np.empty(B)
        for b in range(B):
            pick = rng.choice(keys, size=keys.size, replace=True)
            bs[b] = np.mean(np.concatenate([g[k] for k in pick]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        # Null band: the same design, but every slot picks one of the four at
        # random. 2.5 is the expectation; with this many runs the achievable
        # range around it is what the observed intervals must clear.
        pools = {k: POOL[(ds, metric, k)] for k in keys}
        nb = np.empty(B)
        dif = np.empty(B)
        for b in range(B):
            pick = rng.choice(keys, size=keys.size, replace=True)
            v = np.concatenate([rng.choice(rk, size=2, replace=True)
                                for k in pick for rk in pools[k]])
            nb[b] = v.mean()
            dif[b] = np.mean(np.concatenate([g[k] for k in pick])) - nb[b]
        nlo, nhi = np.percentile(nb, [2.5, 97.5])
        dlo, dhi = np.percentile(dif, [2.5, 97.5])
        res.append(dict(metric=metric, ds=ds, obs=obs, lo=lo, hi=hi,
                        nlo=nlo, nhi=nhi, dlo=dlo, dhi=dhi,
                        n=int(sum(len(g[k]) for k in keys))))
        print(f"{mname:20s} {ds:8s} n={res[-1]['n']:4d}  mean {obs:.3f}"
              f"   null [{nlo:.3f}, {nhi:.3f}]"
              f"   paired diff {np.median(dif):+.3f} [{dlo:+.3f}, {dhi:+.3f}]")
D = pd.DataFrame(res)

fs.apply()
CL = ['#2b5c8f', '#d95f02']
DODGE = 0.13
fig, a = plt.subplots(figsize=(fs.COL_W * 0.76, 2.45))
YLO, YHI = 1.85, 2.95
for i, ds in enumerate(SH):
    q = D[D.ds == ds]
    nlo, nhi = float(q.nlo.mean()), float(q.nhi.mean())
    a.add_patch(plt.Rectangle((i - 0.33, nlo), 0.66, nhi - nlo,
                              facecolor='#ededed', lw=0, zorder=0))
    a.add_patch(plt.Rectangle((i - 0.33, YLO), 0.66, YHI - YLO, fill=False,
                              ec=fs.SPINE, lw=fs.LW_SPINE, zorder=1,
                              clip_on=False))
for k, (metric, _) in enumerate(SCORE):
    for i, ds in enumerate(SH):
        r = D[(D.metric == metric) & (D.ds == ds)].iloc[0]
        x = i + (k - 0.5) * 2 * DODGE
        a.plot([x, x], [r.lo, r.hi], color=CL[k], lw=0.7, zorder=3)
        for e in (r.lo, r.hi):
            a.plot([x - 0.06, x + 0.06], [e, e], color=CL[k], lw=0.7, zorder=3)
        a.plot([x], [r.obs], 'o', ms=3.4, color=CL[k], mew=0, zorder=4)

a.axhline(2.5, color='0.45', lw=fs.LW_RULE, ls=(0, (2.5, 2)), zorder=2)
a.set_xticks(range(len(SH)))
a.set_xticklabels(SH, fontsize=fs.FS_TICK)
for i, ds in enumerate(SH):
    a.text(i, -0.105, f'(n = {int(D[D.ds == ds].n.iloc[0])})',
           transform=a.get_xaxis_transform(), fontsize=fs.FS_NOTE,
           color=fs.INK_TICK, ha='center', va='top')
a.set_xlim(-0.44, len(SH) - 0.56)
a.set_ylim(YLO, YHI)
a.invert_yaxis()
a.set_yticks([2.0, 2.25, 2.5, 2.75])
a.set_yticklabels(['2.00', '2.25', '2.50', '2.75'], fontsize=fs.FS_TICK)
a.set_ylabel('Mean rank of the chosen model', fontsize=fs.FS_LABEL,
             labelpad=fs.LABELPAD)
h = [plt.Line2D([], [], ls='', marker='o', ms=3.4, color=c) for c in CL]
h.append(plt.Rectangle((0, 0), 1, 1, facecolor='#ededed', lw=0))
lg = a.legend(h, ['Recall', 'Precision', 'random selection'],
              title='Ranked by Operational',
              fontsize=fs.FS_NOTE, frameon=False, loc='lower center',
              bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.35,
              columnspacing=1.0, borderpad=0.0, borderaxespad=0.9)
lg.get_title().set_fontsize(fs.FS_NOTE)
lg.get_title().set_color('0.4')
fs.style_axes(a)
a.spines['left'].set_visible(False)
a.spines['bottom'].set_visible(False)
a.tick_params(axis='both', length=0, pad=3)
fig.tight_layout(pad=0.3)
fig.savefig(OUT, dpi=fs.DPI, bbox_inches='tight')
print('saved', OUT)
