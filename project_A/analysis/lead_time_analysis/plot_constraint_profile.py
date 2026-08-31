"""Figure 3: where the extra detections actually go.

Both panels are shares of the failed HDDs in the test set, averaged over the 120
runs (4 models x 30 seeds). They are drawn on separate scales because the two
quantities live in different ranges, and that is the point: loosening the
false-alarm constraint lifts Early steeply while On-time stays flat or falls.
The On-time share is the operational Recall of Eq. (2).

Reads the sweep table committed under results/, so it runs from a clean repo.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figure_style as fs

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, 'results', 'lead_time_analysis', 'reports', 'budget_sweep_360runs.csv')
OUT = os.path.join(ROOT, 'results', 'lead_time_analysis', 'constraint_profile.png')

SH = ['HGST', 'Seagate', 'Toshiba']
CLR = {'HGST': '#2b5c8f', 'Seagate': '#d95f02', 'Toshiba': '#1b9e77'}
C = [0.001, 0.005, 0.01, 0.05]
XT = ['0.1', '0.5', '1', '5']
SEG = [('OT', 'On-time', '#2b5c8f'), ('E', 'Early', '#e8a33d'),
       ('M', 'Missed', '#d9d9d9')]
MAIN = 'Seagate'

df = pd.read_csv(SRC)
r = df[(df.crit == 'row') & (df.budget.isin(C))]

share = {}
for sh in SH:
    g = r[r.ds == sh]
    tot = np.array([g[g.budget == c][['OT', 'E', 'M']].to_numpy().sum(axis=1).mean()
                    for c in C])
    for k in ('OT', 'E', 'M'):
        share[(sh, k)] = np.array([g[g.budget == c][k].mean() for c in C]) / tot

fs.apply()
fig, a = plt.subplots(figsize=(fs.COL_W, 3.0))
x = np.arange(len(C))

bot = np.zeros(len(C))
for k, lab, col in SEG:
    a.bar(x, share[(MAIN, k)], 0.66, bottom=bot, color=col, lw=0.4,
          edgecolor='white', zorder=2, label=lab)
    bot += share[(MAIN, k)]
a.set_xticks(x)
a.set_xticklabels(XT, fontsize=fs.FS_TICK)
a.set_xlim(-0.6, len(C) - 0.4)
a.set_ylim(0, 1.0)
a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
a.set_yticklabels(['0', '25', '50', '75', '100'], fontsize=fs.FS_TICK)
a.text(-0.6, 1.02, '(%)', fontsize=fs.FS_NOTE, color=fs.INK_TICK, ha='right', va='bottom')
a.set_ylabel('Share of failed HDDs', fontsize=fs.FS_LABEL)
a.set_xlabel('Row-level FAR constraint (%)', fontsize=fs.FS_LABEL)
fs.style_axes(a)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.92))
h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SEG]
fig.legend(h, [lab for _, lab, _ in SEG], fontsize=fs.FS_NOTE, frameon=False,
           loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3,
           handlelength=1.1, handleheight=0.9, columnspacing=1.5, borderpad=0.0)
fig.savefig(OUT, dpi=fs.DPI, bbox_inches='tight')
print('saved', OUT)

for sh in SH:
    o, e = share[(sh, 'OT')], share[(sh, 'E')]
    print(f"{sh:8s} alarmed {o[0]+e[0]:5.1%} -> {o[-1]+e[-1]:5.1%}   "
          f"On-time {o[0]:5.1%} -> {o[-1]:5.1%}")
