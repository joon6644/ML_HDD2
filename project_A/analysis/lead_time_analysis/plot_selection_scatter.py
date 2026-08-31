"""Figure 4: does a row-level metric predict operational performance?

Each point is one trained model in one seed. Both axes are deviations from that
seed's own four-model average, so the question is the one selection actually
asks: this model looked better than its siblings on the row-level metric -- was
it better in operation? If it were, points would lie along the diagonal. The
shaded quadrants are where the row-level metric pointed the wrong way.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figure_style as fs

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import prediction_cache as pc

SRC = os.path.join(ROOT, 'results', 'lead_time_analysis', 'reports', 'budget_sweep_360runs.csv')
OUT = os.path.join(ROOT, 'results', 'lead_time_analysis', 'selection_scatter.png')
FULL = {'HGST': 'HGST_20HUH721212ALN604', 'Seagate': 'ST12000NM0007',
        'Toshiba': 'TOSHIBA_20MG07ACA14TA'}
SH = ['HGST', 'Seagate', 'Toshiba']
MD = ['LGBM', 'XGB', 'LSTM', 'GRU']
LBL = {'LGBM': 'LightGBM', 'XGB': 'XGBoost', 'LSTM': 'LSTM', 'GRU': 'GRU'}
CLR = {'LGBM': '#2b5c8f', 'XGB': '#d95f02', 'LSTM': '#1b9e77', 'GRU': '#7570b3'}
PAIR = [('row_rec', 'recall', 'Recall'), ('row_prec', 'precision', 'Precision')]

df = pd.read_csv(SRC)
df = df[(df.budget == 0.01) & (df.crit == 'row')].copy()


def row_precision(ds, m, seed, thr):
    c = pc.load_row_curve(FULL[ds], m, seed, 'test')
    i = int(np.abs(c['threshold'].to_numpy() - thr).argmin())
    tp, fp = float(c.tp.iloc[i]), float(c.fp.iloc[i])
    return tp / (tp + fp) if tp + fp else 0.0


df['row_prec'] = [row_precision(r.ds, r.model, r.seed, r.thr) for r in df.itertuples()]
for col in ['row_rec', 'row_prec', 'recall', 'precision']:
    df['d_' + col] = df[col] - df.groupby(['ds', 'seed'])[col].transform('mean')

fs.apply()
fig, ax = plt.subplots(2, 3, figsize=(fs.FULL_W, 4.5))
for i, (xc, yc, name) in enumerate(PAIR):
    for j, sh in enumerate(SH):
        a = ax[i, j]
        g = df[df.ds == sh]
        x, y = g['d_' + xc].to_numpy(), g['d_' + yc].to_numpy()
        lim = max(np.abs(x).max(), np.abs(y).max()) * 1.12
        a.axhspan(-lim, 0, xmin=0.5, xmax=1.0, color='#f5f0e8', zorder=0)
        a.axhspan(0, lim, xmin=0.0, xmax=0.5, color='#f5f0e8', zorder=0)
        a.axhline(0, color='0.45', lw=fs.LW_RULE, zorder=2)
        a.axvline(0, color='0.45', lw=fs.LW_RULE, zorder=2)
        for m in MD:
            k = g.model == m
            a.plot(g.loc[k, 'd_' + xc], g.loc[k, 'd_' + yc], 'o', ms=3.0,
                   color=CLR[m], alpha=0.8, mew=0, zorder=3)
        taus, hit, ns = [], 0, 0
        for _, h in g.groupby('seed'):
            if len(h) < 4:
                continue
            taus.append(kendalltau(h[xc], h[yc]).statistic)
            hit += int(h[xc].idxmax() == h[yc].idxmax())
            ns += 1
        tau = float(np.mean(taus))
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.text(0.03, 0.97, f'rank agreement {tau:+.2f}\nsame pick {hit}/{ns} seeds',
               transform=a.transAxes, fontsize=fs.FS_NOTE, va='top', ha='left',
               linespacing=1.4, color='0.25')
        fs.style_axes(a)
        a.tick_params(labelsize=fs.FS_TICK)
        if i == 0:
            a.set_title(sh, fontsize=fs.FS_TITLE)
        if j == 0:
            a.set_ylabel(f'HDD-level {name}\n(deviation from seed mean)', fontsize=fs.FS_LABEL)
        a.set_xlabel(f'Row-level {name}  (deviation)', fontsize=fs.FS_LABEL)
        print(f"{name:10s} {sh:8s} tau {tau:+.3f}  same pick {hit}/{ns}  n={len(g)}")

h = [plt.Line2D([], [], ls='', marker='o', ms=3.6, color=CLR[m]) for m in MD]
ax[0, 0].legend(h, [LBL[m] for m in MD], fontsize=fs.FS_NOTE, frameon=False,
                loc='lower right', handletextpad=0.3, labelspacing=0.22, borderpad=0.1)
fig.tight_layout(pad=0.4, h_pad=1.0, w_pad=1.0)
fig.savefig(OUT, dpi=fs.DPI, bbox_inches='tight')
print('saved', OUT)
