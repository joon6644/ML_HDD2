"""Fig. 2 -- where the failure-observed population sits as the early-alarm rate rises.

Two forces act in opposite directions as the threshold falls. Disks that raised
no alarm at all start raising one, which moves mass out of Missed; and alarms
that already existed move earlier, which moves mass out of On-time into Early.
Stacking the three shares over a continuous EAR axis shows which force wins
where, which a table of four constraint levels cannot.

The marker sits at the top of the On-time band, which is operational recall at its
largest. It is a dot rather than a rule because the band is broad and flat near
its top and the vertex moves by up to a factor of five between models: the height
is worth reading, the abscissa is not.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figure_style as fs

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CURVES = os.path.join(ROOT, "results", "lead_time_analysis", "burden_sweep_curves.parquet")
OUT = os.path.join(ROOT, "results", "lead_time_analysis", "burden_outcome_composition.png")

SHORT = ["HGST", "Seagate", "Toshiba"]
SEG = [("ot_med", "On-time", "#24506e"),
       ("early_med", "Early", "#8ba9c0"),
       ("missed_med", "Missed", "#e4e9ee")]
H = 30
XLIM = (0.1, 30.0)
XTICKS = [0.1, 0.5, 1, 5, 10, 30]

c = pd.read_parquet(CURVES)

fs.apply()
fig, axes = plt.subplots(1, 3, figsize=(fs.FULL_W, 2.35), sharey=True)

for ax, ds in zip(axes, SHORT):
    d = c[c.dataset == ds].sort_values("ear_grid")
    x = d.ear_grid.to_numpy() * 100
    # The three medians are taken independently and so miss unity by a fraction
    # of a percent; renormalising keeps the stack's top edge from wobbling.
    bands = np.vstack([d[k].to_numpy() for k, _, _ in SEG])
    bands /= bands.sum(0)
    ax.stackplot(x, *bands, colors=[col for _, _, col in SEG],
                 edgecolor="white", linewidth=0.6, zorder=2)

    # Same rule and same label form as Fig. 3, here at the top of the On-time
    # band. The label is the unnormalised operational recall, so figure and table
    # agree digit for digit.
    recall = d.recall_med.to_numpy()
    k = int(np.argmax(recall))
    ax.axvline(x[k], color="0.35", lw=0.5, ls=(0, (1.5, 2)), zorder=5)
    # Two lines, so the label stays inside the panel without leaving the rule.
    ax.annotate(f"max recall\n{recall[k]:.3f}", xy=(x[k], 0.97), xytext=(3, -5),
                textcoords="offset points", ha="left", va="top",
                fontsize=fs.FS_NOTE, color="0.3", zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([f"{t:g}" for t in XTICKS], fontsize=fs.FS_TICK)
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_title(ds, fontsize=fs.FS_TITLE, pad=fs.TITLE_PAD)
    ax.set_xlabel("Early alarm rate (%)", fontsize=fs.FS_LABEL,
                  labelpad=fs.LABELPAD)
    fs.style_axes(ax)
    ax.grid(False)
    ax.tick_params(axis="both", length=3.0, width=0.5, color=fs.SPINE)

axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axes[0].set_yticklabels(["0", "25", "50", "75", "100"], fontsize=fs.FS_TICK)
axes[0].set_ylabel("Share of failure-observed HDDs (%)",
                   fontsize=fs.FS_LABEL, labelpad=fs.LABELPAD)

handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in SEG]
fig.legend(handles, [lab for _, lab, _ in SEG], fontsize=fs.FS_NOTE,
           frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.005),
           ncol=3, handlelength=1.1, handleheight=0.9, columnspacing=1.6,
           borderpad=0.0)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
fig.savefig(OUT, dpi=fs.DPI, bbox_inches="tight")
print("saved", OUT)

for ds in SHORT:
    d = c[c.dataset == ds].sort_values("ear_grid")
    a = d.iloc[(d.ear_grid - 0.025).abs().argmin()]
    b = d.iloc[(d.ear_grid - 0.10).abs().argmin()]
    print(f"{ds:9s} EAR {a.ear_grid*100:.2f}% -> {b.ear_grid*100:.2f}%   "
          f"recall {a.recall_med:.3f} -> {b.recall_med:.3f}   "
          f"Early {a.early_med:.2f} -> {b.early_med:.2f}")
