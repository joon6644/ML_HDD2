"""Fig. 3 -- the lead-time distribution of the first alarm, over the EAR axis.

The verdict splits this distribution at H, so plotting the distribution itself
says what the split cannot: how far past H the mass sits, and how wide it is. The
band spans two orders of magnitude at every burden, which is why the answer to
"is H too short" is not a yes or a no -- moving the line moves the verdict.
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
OUT = os.path.join(ROOT, "results", "lead_time_analysis", "burden_leadtime_band.png")

SHORT = ["HGST", "Seagate", "Toshiba"]
H = 30
XLIM = (0.1, 30.0)
XTICKS = [0.1, 0.5, 1, 5, 10, 30]
YTICKS = [1, 10, 30, 100, 365, 1000]
OUTER, INNER, MID = "#c3d0da", "#8ba9c0", "#24506e"

c = pd.read_parquet(CURVES)

fs.apply()
fig, axes = plt.subplots(1, 3, figsize=(fs.FULL_W, 2.35), sharey=True)

for ax, ds in zip(axes, SHORT):
    d = c[c.dataset == ds].sort_values("ear_grid")
    x = d.ear_grid.to_numpy() * 100

    ax.fill_between(x, d.lt_p10_med, d.lt_p90_med, color=OUTER,
                    linewidth=0, zorder=2)
    ax.fill_between(x, d.lt_p25_med, d.lt_p75_med, color=INNER,
                    linewidth=0, zorder=3)
    ax.plot(x, d.lt_p50_med, color=MID, lw=fs.LW_DATA, zorder=4)

    ax.axhline(H, color=fs.ACCENT, lw=fs.LW_RULE, ls=(0, (3, 2)), zorder=5)

    under = d[d.lt_p50_med <= H]
    if not under.empty:
        b = under.ear_grid.max() * 100
        ax.axvline(b, color="0.35", lw=0.5, ls=(0, (1.5, 2)), zorder=5)
        ax.annotate(f"EAR {b:.2f}%", xy=(b, 1400), xytext=(3, -5),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=fs.FS_NOTE, color="0.3", zorder=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(1, 1600)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([f"{t:g}" for t in XTICKS], fontsize=fs.FS_TICK)
    ax.minorticks_off()
    ax.set_title(ds, fontsize=fs.FS_TITLE, pad=fs.TITLE_PAD)
    ax.set_xlabel("Early alarm rate (%)", fontsize=fs.FS_LABEL,
                  labelpad=fs.LABELPAD)
    fs.style_axes(ax)
    ax.tick_params(axis="both", length=3.0, width=0.5, color=fs.SPINE)

axes[0].set_yticks(YTICKS)
axes[0].set_yticklabels([str(t) for t in YTICKS], fontsize=fs.FS_TICK)
axes[0].set_ylabel("Lead time of first alarm (d)", fontsize=fs.FS_LABEL,
                   labelpad=fs.LABELPAD)
axes[0].annotate("$H$ = 30 d", xy=(XLIM[0] * 1.15, H), xytext=(0, 3),
                 textcoords="offset points", ha="left", va="bottom",
                 fontsize=fs.FS_NOTE, color=fs.ACCENT, zorder=6)

handles = [plt.Rectangle((0, 0), 1, 1, color=INNER),
           plt.Rectangle((0, 0), 1, 1, color=OUTER),
           plt.Line2D([], [], color=MID, lw=fs.LW_DATA)]
fig.legend(handles, ["25–75 pct", "10–90 pct", "median"], fontsize=fs.FS_NOTE,
           frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.005),
           ncol=3, handlelength=1.1, handleheight=0.9, columnspacing=1.6,
           borderpad=0.0)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
fig.savefig(OUT, dpi=fs.DPI, bbox_inches="tight")
print("saved", OUT)
