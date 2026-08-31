"""Check: what a single run looks like before averaging.

Figure 3 averages 120 runs, which makes the outcome mix look continuous. Within
one run it is not -- each failed HDD flips category at one threshold, so the
curve is a staircase whose steps are 1 / (failed HDDs) tall. This draws one run
next to the average so the smoothing can be seen for what it is.
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

OUT = os.path.join(ROOT, 'results', 'lead_time_analysis', 'constraint_steps.png')
FULL = {'Seagate': 'ST12000NM0007'}
DS = 'Seagate'
MD = ['LGBM', 'XGB', 'LSTM', 'GRU']
SEEDS = range(42, 72)
ONE = ('LGBM', 42)
H = 30
SEG = [('OT', 'On-time', '#24506e'), ('E', 'Early', '#8ba9c0'),
       ('M', 'Missed', '#e4e9ee')]


def outcome_curves(model, seed, T):
    d = pd.read_parquet(pc.disk_steps_path(FULL[DS], model, seed, 'test'))
    d = d[d.failure_date.notna()].sort_values(['serial_number', 'step_date'])
    lt = (pd.to_datetime(d.failure_date) - pd.to_datetime(d.step_date)).dt.days.to_numpy()
    v = d.step_value.to_numpy(float)
    sn = d.serial_number.to_numpy()
    start = np.r_[0, np.flatnonzero(sn[1:] != sn[:-1]) + 1]
    end = np.r_[start[1:], v.size]
    n = start.size
    cut_on = np.empty(n)
    cut_off = np.empty(n)
    for i, (s, e) in enumerate(zip(start, end)):
        vi, li = v[s:e], lt[s:e]
        cut_off[i] = vi[-1]
        ok = np.flatnonzero(li <= H)
        cut_on[i] = vi[ok[0] - 1] if ok.size and ok[0] > 0 else (
            -np.inf if ok.size else np.inf)
    ot = ((T[None, :] > cut_on[:, None]) & (T[None, :] <= cut_off[:, None])).sum(0)
    al = (T[None, :] <= cut_off[:, None]).sum(0)
    return np.vstack([ot, al - ot, n - al]) / n, n


T = pc.load_row_curve(FULL[DS], 'XGB', 42, 'test')['threshold'].to_numpy()
acc, far = [], []
for m in MD:
    for s in SEEDS:
        c, n = outcome_curves(m, s, T)
        acc.append(c)
        rc = pc.load_row_curve(FULL[DS], m, s, 'test')
        far.append((rc.fp / (rc.fp + rc.tn)).to_numpy())
mean_share = np.mean(acc, axis=0)
mean_far = np.mean(far, axis=0) * 100

one_share, n_fail = outcome_curves(*ONE, T)
rc = pc.load_row_curve(FULL[DS], ONE[0], ONE[1], 'test')
one_far = (rc.fp / (rc.fp + rc.tn)).to_numpy() * 100
print(f'{DS} {ONE[0]} seed {ONE[1]}: {n_fail} failed HDDs, one step = {1/n_fail:.2%}')

fs.apply()
fig, ax = plt.subplots(2, 1, figsize=(fs.COL_W, 3.8), sharex=True, sharey=True)
for a, (xf, sh, ttl) in zip(ax, [
        (one_far, one_share, f'one run  ({ONE[0]}, seed {ONE[1]})'),
        (mean_far, mean_share, 'mean of 120 runs')]):
    o = np.argsort(xf)
    a.stackplot(xf[o], sh[0][o], sh[1][o], sh[2][o],
                colors=[c for _, _, c in SEG], step='post',
                edgecolor='white', linewidth=0.5, zorder=2)
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
    a.set_title(ttl, fontsize=fs.FS_LABEL, loc='left', pad=fs.TITLE_PAD)
    a.grid(False)
    fs.style_axes(a)
    a.tick_params(axis='both', length=3.0, width=0.5, color=fs.SPINE)

ax[0].set_ylabel('Share of failed HDDs (%)', fontsize=fs.FS_LABEL)
ax[1].set_ylabel('Share of failed HDDs (%)', fontsize=fs.FS_LABEL)
ax[1].set_xlabel('Row-level FAR at the threshold (%)', fontsize=fs.FS_LABEL)
fig.tight_layout(pad=0.3, h_pad=0.8, rect=(0, 0, 1, 0.955))
h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SEG]
fig.legend(h, [l for _, l, _ in SEG], fontsize=fs.FS_NOTE, frameon=False,
           loc='upper center', bbox_to_anchor=(0.5, 1.005), ncol=3,
           handlelength=1.1, handleheight=0.9, columnspacing=1.5, borderpad=0.0)
fig.savefig(OUT, dpi=fs.DPI, bbox_inches='tight')
print('saved', OUT)
