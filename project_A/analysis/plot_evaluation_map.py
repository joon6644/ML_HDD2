"""Map of the two evaluation procedures defined in Section 3.

Not derived from data: it draws the structure of the evaluation itself. Both
tracks descend from a single root -- the same trained model, the same online
inference, the same row set -- so every difference below the fork is a
difference of procedure and not of data. That is the claim 4.2 makes in prose,
and the figure is what makes it checkable at a glance.

The two grids at the bottom carry the rest of the argument. The row-level
track folds into a 2x2 confusion matrix; the operational track does not fold,
because a disk's outcome is decided by three axes (censoring x alarm x timing)
and 2x2 has only four cells. The 5x4 grid also carries the coverage claim of
METRIC_DESIGN.md 5: no category row is empty, so no HDD falls outside the
metric set. Early's row is where the asymmetry shows -- it is charged to
Precision and Recall but never to FAR.

No measured quantity appears here except H, so the pending threshold re-search
leaves the figure untouched.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "method_figures")

H_DAYS = 30

# A 0..100 square is easier to reason about than inches; the figure aspect
# below is what actually sets the proportions. The two tracks are not equally
# wide -- the 5x4 grid needs more room than the 2x2 -- so the centre lines are
# placed to make the OUTER edges symmetric rather than the centres.
W_L, W_R = 18.0, 22.0
X_L, X_R = 25.0, 71.0          # left spans 7..43, right spans 49..93
X_FORK = (X_L + X_R) / 2
Y_ROOT = (84.0, 95.5)
Y_BAR = 80.5
Y_HEAD = (71.0, 78.0)
Y_ST1 = (60.0, 68.5)
Y_ST2 = (49.0, 57.5)
Y_GRID_TOP = 46.0              # both bottom blocks start here

# The baseline track is set in neutral grey and the proposed track in the
# accent colour, so which of the two the paper contributes reads without a
# caption. The accent matches the On-time colour of the companion figure.
COL_BASE = "#5a5a5a"
COL_BASE_FILL = "#f2f2f2"
COL_OP = "#2b5c8f"
COL_OP_FILL = "#eaf0f6"
COL_ROOT_FILL = "#e8e8e8"
COL_TEXT = "#111111"
COL_MUTED = "#7a7a7a"
COL_RULE = "#b5b5b5"


def _box(ax, x0, y0, x1, y1, text, edge, fill, fontsize=9.5, weight="normal",
         color=COL_TEXT):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=1.4",
        linewidth=1.3, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=color, zorder=3,
            linespacing=1.55)


def _arrow(ax, x, y0, y1, color):
    ax.annotate("", xy=(x, y1), xytext=(x, y0), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                                shrinkA=0, shrinkB=0))


def _grid(ax, x0, x1, y0, y1, n_rows, n_cols, label_w, color):
    """Draw a table frame and return the cell centre lookups.

    label_w is the share of the width given to the row-label column; the
    remaining width is split evenly among the data columns.
    """
    x_split = x0 + label_w
    col_w = (x1 - x_split) / n_cols
    row_h = (y1 - y0) / n_rows
    ax.plot([x_split, x_split], [y0, y1], color=color, lw=1.0, zorder=3)
    for i in range(1, n_cols):
        x = x_split + i * col_w
        ax.plot([x, x], [y0, y1], color=color, lw=0.6, zorder=3)
    for j in range(1, n_rows):
        y = y0 + j * row_h
        ax.plot([x0, x1], [y, y], color=color, lw=0.6, zorder=3)
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.8",
        linewidth=1.1, edgecolor=color, facecolor="none", zorder=3))
    col_c = [x_split + (i + 0.5) * col_w for i in range(n_cols)]
    row_c = [y1 - (j + 0.5) * row_h for j in range(n_rows)]
    return (x0 + x_split) / 2, col_c, row_c


def _draw_root(ax):
    x0, x1 = X_L - W_L, X_R + W_R
    _box(ax, x0, Y_ROOT[0], x1, Y_ROOT[1], "", COL_MUTED, COL_ROOT_FILL)
    ax.text((x0 + x1) / 2, 91.3,
            "학습된 모델   →   시간 순 온라인 추론   →   HDD별 예측확률 $p_{d,t}$",
            ha="center", va="center", fontsize=11.5, fontweight="bold",
            color=COL_TEXT, zorder=3)
    # Kept inside the root box rather than floating beneath it: the sentence is
    # a property of the shared root, and a caption under the box would have to
    # share that space with the fork.
    ax.text((x0 + x1) / 2, 87.0,
            "두 평가는 동일한 예측 결과와 동일한 평가 대상 행 집합을 사용한다",
            ha="center", va="center", fontsize=9.5, color=COL_MUTED, zorder=3)

    # The fork descends from the root band rather than from either track, so
    # neither procedure reads as the continuation and the other as a branch.
    ax.plot([X_FORK, X_FORK], [Y_ROOT[0], Y_BAR], color=COL_RULE, lw=1.3,
            zorder=1)
    ax.plot([X_L, X_R], [Y_BAR, Y_BAR], color=COL_RULE, lw=1.3, zorder=1)
    _arrow(ax, X_L, Y_BAR, Y_HEAD[1], COL_BASE)
    _arrow(ax, X_R, Y_BAR, Y_HEAD[1], COL_OP)


def _draw_row_track(ax):
    w = W_L
    _box(ax, X_L - w, Y_HEAD[0], X_L + w, Y_HEAD[1],
         "3.1  행 단위 평가 (기준선)", COL_BASE, COL_BASE_FILL,
         fontsize=11.5, weight="bold", color=COL_BASE)
    _arrow(ax, X_L, Y_HEAD[0], Y_ST1[1], COL_BASE)

    _box(ax, X_L - w, Y_ST1[0], X_L + w, Y_ST1[1],
         "각 관측 행을 독립적인 분류 대상으로 취급\n"
         "$p_{d,t} \\geq \\tau$ 인 행을 양성으로 예측",
         COL_BASE, "#ffffff")
    _arrow(ax, X_L, Y_ST1[0], Y_ST2[1], COL_BASE)

    _box(ax, X_L - w, Y_ST2[0], X_L + w, Y_ST2[1],
         "행 레이블 부여\n"
         "양성: 고장 관측 HDD의 $t_{\\mathrm{failure},d} - t \\leq H$\n"
         "음성: 그 외의 모든 행",
         COL_BASE, "#ffffff")
    _arrow(ax, X_L, Y_ST2[0], Y_GRID_TOP, COL_BASE)

    # 2x2. Drawn at the same scale as the operational grid so the reader can
    # compare the two shapes directly -- that comparison is the point.
    gx0, gx1 = X_L - w, X_L + w
    gy0, gy1 = 30.0, Y_GRID_TOP
    lab_c, col_c, row_c = _grid(ax, gx0, gx1, gy0, gy1, 3, 2, 13.0, COL_BASE)
    for x, t in zip(col_c, ("실제 양성 행", "실제 음성 행")):
        ax.text(x, row_c[0], t, ha="center", va="center", fontsize=8.8,
                color=COL_BASE, zorder=4)
    for y, t in zip(row_c[1:], ("양성 예측", "음성 예측")):
        ax.text(lab_c, y, t, ha="center", va="center", fontsize=8.8,
                color=COL_BASE, zorder=4)
    for j, row in enumerate((("TP", "FP"), ("FN", "TN"))):
        for x, t in zip(col_c, row):
            ax.text(x, row_c[j + 1], t, ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color=COL_TEXT, zorder=4)

    for i, formula in enumerate((
        r"$\mathrm{Precision}_{\mathrm{row}} = TP\,/\,(TP+FP)$",
        r"$\mathrm{Recall}_{\mathrm{row}} = TP\,/\,(TP+FN)$",
        r"$\mathrm{FAR}_{\mathrm{row}} = FP\,/\,(FP+TN)$",
    )):
        ax.text(X_L, 24.0 - i * 5.4, formula, ha="center", va="center",
                fontsize=10.5, color=COL_TEXT)


def _draw_op_track(ax):
    w = W_R
    _box(ax, X_R - w, Y_HEAD[0], X_R + w, Y_HEAD[1],
         "3.2  운영 환경 기반 평가 (제안)", COL_OP, COL_OP_FILL,
         fontsize=11.5, weight="bold", color=COL_OP)
    _arrow(ax, X_R, Y_HEAD[0], Y_ST1[1], COL_OP)

    _box(ax, X_R - w, Y_ST1[0], X_R + w, Y_ST1[1],
         "HDD별 반복 Alarm을 하나의 운영 사건으로 통합\n"
         "최초 Alarm 시점 $t_{\\mathrm{alarm},d}$ 를 기준으로 판정",
         COL_OP, "#ffffff")
    _arrow(ax, X_R, Y_ST1[0], Y_ST2[1], COL_OP)

    _box(ax, X_R - w, Y_ST2[0], X_R + w, Y_ST2[1],
         "고장 관측 여부와 $LT_d \\leq H$ 에 따라\n"
         "각 HDD를 5개 결과 범주 중 하나로 분류",
         COL_OP, "#ffffff")
    _arrow(ax, X_R, Y_ST2[0], Y_GRID_TOP, COL_OP)

    gx0, gx1 = X_R - w, X_R + w
    gy0, gy1 = 12.0, Y_GRID_TOP
    lab_c, col_c, row_c = _grid(ax, gx0, gx1, gy0, gy1, 6, 4, 17.0, COL_OP)
    for x, t in zip(col_c, ("Precision", "Recall", "FAR",
                            "Median\nLead Time")):
        ax.text(x, row_c[0], t, ha="center", va="center", fontsize=8.6,
                color=COL_OP, zorder=4, linespacing=1.35)

    # "N" marks a category that enters the numerator, "D" one that enters only
    # the denominator. Median Lead Time is not a ratio, so its column marks
    # sample membership instead; the note under the grid says so.
    rows = (
        ("On-time",           ("N", "N", "-", "N")),
        ("Early",             ("D", "D", "-", "N")),
        ("Missed",            ("-", "D", "-", "-")),
        ("Censored Early",    ("D", "-", "N", "-")),
        ("Censored No Alarm", ("-", "-", "D", "-")),
    )
    for j, (name, marks) in enumerate(rows):
        y = row_c[j + 1]
        ax.text(lab_c, y, name, ha="center", va="center", fontsize=8.8,
                color=COL_TEXT, zorder=4)
        for x, m in zip(col_c, marks):
            if m == "-":
                ax.text(x, y, "·", ha="center", va="center", fontsize=16,
                        color="#a8a8a8", zorder=4)
            else:
                ax.scatter([x], [y], s=64, zorder=4, color=COL_OP,
                           facecolor=COL_OP if m == "N" else "#ffffff",
                           linewidths=1.4, edgecolor=COL_OP)

    ax.text(X_R, gy0 - 3.0,
            "●  분자에 포함     ○  분모에만 포함     ·  사용하지 않음",
            ha="center", va="top", fontsize=8.8, color=COL_TEXT)
    ax.text(X_R, gy0 - 6.6,
            "Median Lead Time은 비율이 아니므로 ● 는 산출 표본에 포함됨을 뜻한다.",
            ha="center", va="top", fontsize=8.2, color=COL_MUTED)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Malgun Gothic", "Noto Sans KR",
                                       "Arial", "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    # Computer Modern for the math only, matching how the symbols are set in
    # the manuscript. Full text.usetex would route the Korean through LaTeX.
    plt.rcParams["mathtext.fontset"] = "cm"

    # Sized for a double-column figure: the 5x4 grid and the three row-level
    # formulas cannot both hold their point sizes inside one column.
    fig, ax = plt.subplots(figsize=(13.0, 8.6), dpi=300)

    _draw_root(ax)
    _draw_row_track(ax)
    _draw_op_track(ax)

    # Placed under the row-level track, which ends higher than the operational
    # grid: it fills the one empty corner instead of adding a fourth line of
    # notes beneath the legend.
    ax.text(X_L, 7.0,
            f"$H$ 는 Prediction Horizon이며\n본 연구에서는 $H={H_DAYS}$일이다.",
            ha="center", va="center", fontsize=9, color=COL_MUTED,
            linespacing=1.6)

    ax.set_xlim(X_L - W_L - 4, X_R + W_R + 4)
    ax.set_ylim(2, 99)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "evaluation_map.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved -> {out}")


if __name__ == "__main__":
    main()