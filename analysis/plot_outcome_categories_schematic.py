"""Schematic of the five outcome categories defined in Section 3.2.

Not derived from data: it draws the decision rule itself. Each row is one
HDD's observation history, and the segments of that history carry the category
a first alarm falling there would receive.

The two rows live in one coordinate system so the observation-end line is
literally shared. That line is also what separates the two groups: a failed
disk's record ends before it, a censored disk's record runs to it.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "method_figures")

# Schematic scale. The H stretch is drawn wide enough to label; a real history
# spans years while H is 30 days, so no drawing scale would be faithful. The
# quoted value is therefore kept separate from the drawn width.
T_START = 0
T_OBS_END = 300
H_DRAW = 70
# The failure sits well inside the window. Placing it just short of the end
# would make the remaining gap look like a quantity of its own; landing it
# mid-way through the censored row's excluded stretch reads as incidental.
T_FAIL = T_OBS_END - H_DRAW // 2
H_DAYS = 30
T_H_FAILED = T_FAIL - H_DRAW        # On-time window opens here
T_H_CENSORED = T_OBS_END - H_DRAW   # last evaluable instant
# Right-hand column for the two outcomes that are not a stretch of time but the
# absence of any alarm. They are set in the same style as the other three names
# so that all five categories read as one set. The centre is placed so that the
# longest name ("Censored No Alarm") clears the observation-end line by roughly
# the width of one word and no more.
X_COL_C = 352
# Both limits sit one pad width outside the outermost text, so the trimmed PNG
# carries the same white margin on the left and the right.
X_LIM_L, X_LIM_R = T_START - 95, 397

Y_FAILED, Y_CENSORED = 1.0, 0.0

COL_LINE = "#8a8a8a"
COL_ONTIME = "#2b5c8f"
COL_FAIL = "#111111"
COL_END = "#5a5a5a"
COL_BOUND = "#2b5c8f"
# Faded rather than dashed: a dash pattern gets truncated mid-stroke at the
# segment end, which reads as a drawing error rather than as an excluded range.
COL_EXCLUDE = "#d2d2d2"


def _segment(ax, x0, x1, y, color, lw=5.0, **kw):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt",
            zorder=3, **kw)


def _span(ax, x0, x1, y, label, color="#333333"):
    """Dimension-style bracket naming the category of a stretch of history."""
    yb = y + 0.20
    ax.plot([x0, x1], [yb, yb], color=color, lw=0.9, zorder=3)
    for x in (x0, x1):
        ax.plot([x, x], [yb - 0.045, yb], color=color, lw=0.9, zorder=3)
    ax.text((x0 + x1) / 2, yb + 0.035, label, ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=color)


def _cond(ax, x0, x1, y, text, color="#5a5a5a"):
    ax.text((x0 + x1) / 2, y - 0.13, text, ha="center", va="top",
            fontsize=9.5, color=color)


def _tick(ax, x, y, color, ls, lw, half=0.11):
    ax.plot([x, x], [y - half, y + half], color=color, ls=ls, lw=lw, zorder=5)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Korean labels match the terms the manuscript uses in prose; the outcome
    # names stay in English because that is how the manuscript defines them.
    plt.rcParams["font.sans-serif"] = ["Malgun Gothic", "Noto Sans KR",
                                       "Arial", "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    # Computer Modern for the math only, so the symbols match how they are set
    # in the manuscript. Full text.usetex would route the Korean labels through
    # LaTeX as well, which the default engine cannot typeset.
    plt.rcParams["mathtext.fontset"] = "cm"

    # Narrow enough to drop into a manuscript column without further scaling;
    # the point sizes are fixed, so a narrower figure also sets the labels
    # larger relative to the drawing.
    fig, ax = plt.subplots(figsize=(10.2, 4.3), dpi=300)

    # ---- shared reference: the end of the observation window -----------------
    ax.plot([T_OBS_END, T_OBS_END], [Y_CENSORED - 0.42, Y_FAILED + 0.46],
            color=COL_END, ls="--", lw=1.7, zorder=4)
    ax.text(T_OBS_END, Y_FAILED + 0.50, "관측 종료($t_{\mathrm{end},d}$)", ha="center",
            va="bottom", fontsize=10, fontweight="bold", color=COL_END)

    # ---- row 1: failure observed inside the window --------------------------
    _segment(ax, T_START, T_H_FAILED, Y_FAILED, COL_LINE)
    _segment(ax, T_H_FAILED, T_FAIL, Y_FAILED, COL_ONTIME, lw=7.0)
    _span(ax, T_START, T_H_FAILED, Y_FAILED, "Early")
    _span(ax, T_H_FAILED, T_FAIL, Y_FAILED, "On-time", color=COL_ONTIME)
    _cond(ax, T_START, T_H_FAILED, Y_FAILED, r"$LT_d > H$")
    _cond(ax, T_H_FAILED, T_FAIL, Y_FAILED, r"$LT_d \leq H$", color=COL_ONTIME)
    _tick(ax, T_H_FAILED, Y_FAILED, COL_BOUND, ":", 1.5)
    _tick(ax, T_FAIL, Y_FAILED, COL_FAIL, "-", 2.2, half=0.13)
    ax.text(T_FAIL, Y_FAILED - 0.30, r"$t_{\mathrm{failure},d}$", ha="center",
            va="top", fontsize=9.5, fontweight="bold", color=COL_FAIL)
    ax.text(T_H_FAILED, Y_FAILED - 0.30, r"$t_{\mathrm{failure},d} - H$",
            ha="center", va="top", fontsize=9.5, color=COL_BOUND)

    # ---- row 2: right-censored ---------------------------------------------
    _segment(ax, T_START, T_H_CENSORED, Y_CENSORED, COL_LINE)
    _segment(ax, T_H_CENSORED, T_OBS_END, Y_CENSORED, COL_EXCLUDE, lw=5.0)
    # Only the assignable stretch gets a bracket: the bracket row is reserved
    # for category names, and the excluded stretch is not a category.
    _span(ax, T_START, T_H_CENSORED, Y_CENSORED, "Censored Early")
    _cond(ax, T_START, T_H_CENSORED, Y_CENSORED, r"$LT_d > H$ 보장")
    _cond(ax, T_H_CENSORED, T_OBS_END, Y_CENSORED, "평가 제외", color="#8a8a8a")
    _tick(ax, T_H_CENSORED, Y_CENSORED, COL_BOUND, ":", 1.5)
    ax.text(T_H_CENSORED, Y_CENSORED - 0.30, r"$t_{\mathrm{end},d} - H$",
            ha="center", va="top", fontsize=9.5, color=COL_BOUND)

    for y, row_name, no_alarm in (
        (Y_FAILED, "고장 관측 HDD", "Missed"),
        (Y_CENSORED, "우측 중도절단 HDD", "Censored No Alarm"),
    ):
        ax.text(T_START - 14, y, row_name, ha="right", va="center", fontsize=11,
                fontweight="bold", color="#111111")
        ax.text(X_COL_C, y, no_alarm, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="#333333")

    ax.annotate("", xy=(T_OBS_END + 6, Y_CENSORED - 0.55),
                xytext=(T_START, Y_CENSORED - 0.55),
                arrowprops=dict(arrowstyle="->", color="#9a9a9a", lw=1.3))
    ax.text(T_OBS_END + 6, Y_CENSORED - 0.62, "시간($t$)", ha="right", va="top",
            fontsize=9, color="#9a9a9a")

    handles = [
        Line2D([], [], color=COL_ONTIME, lw=7.0,
               label=f"On-time 구간 (고장 전 $H$일 이내, $H={H_DAYS}$일)"),
        Line2D([], [], color=COL_LINE, lw=5.0, label="평가 대상 관측 이력"),
        Line2D([], [], color=COL_EXCLUDE, lw=5.0, label="평가 제외 구간"),
    ]
    # The legend is centred on the timeline rather than on the axes, which
    # extends further right to hold the outcome column.
    timeline_c = ((T_START + T_OBS_END) / 2 - X_LIM_L) / (X_LIM_R - X_LIM_L)
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(timeline_c, -0.02),
              ncol=3, frameon=False, fontsize=9.5)

    ax.set_xlim(X_LIM_L, X_LIM_R)
    ax.set_ylim(Y_CENSORED - 0.98, Y_FAILED + 0.72)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "outcome_categories_schematic.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved -> {out}")


if __name__ == "__main__":
    main()