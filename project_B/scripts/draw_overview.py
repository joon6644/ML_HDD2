"""예측 문제와 판정 규칙 개요도 — 심사의견 3-2 에 대응하는 그림.

    python scripts/draw_overview.py

양식은 arXiv:1810.08985 의 Fig.1 을 따른다. 그 그림은 기기 한 대당 가로줄
하나를 긋고, 표식 몇 개와 모서리 범례로 끝낸다. 문장을 쓰지 않는다.

따라서 여기서도 글로 설명하지 않는다. 재현율·오탐률의 정의는 3.4 의 수식이
맡고, 이 그림은 그 수식이 세는 대상이 무엇인지만 보인다.

    고장 디스크 한 대와 정상 디스크 한 대의 한 달
    점수 곡선, 임곗값, 예측 지평, 그리고 그 결과 붙는 판정 하나

담지 않는 것: 시간순 분할, 전처리, 하이퍼파라미터 탐색, 비교 모델, 지표 정의.
본문에 있고 이 그림이 답하는 질문이 아니다.

색은 둘. 주황은 사실(고장·지평), 파랑은 모델(임곗값·경보). 흑백 인쇄를
고려해 선 모양으로도 갈린다. 그림에는 한글을 쓰지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

WARM = "#C0601A"
COOL = "#2A5A9E"
INK = "#141414"
MUTE = "#767676"

DAYS = 30
X0, X1 = 7.0, 84.0
THR = 0.52
FAIL_DAY, HZ_START = 28, 18


def day_x(d: float) -> float:
    return X0 + (X1 - X0) * (d / DAYS)


def score_curve(kind: str) -> np.ndarray:
    """모식도용 점수 곡선. 실제 예측값이 아니라 판정 규칙을 설명한다."""
    rng = np.random.default_rng(7 if kind == "failed" else 11)
    t = np.linspace(0, 1, DAYS)
    base = 0.18 + 0.05 * np.sin(t * 7)
    noise = np.convolve(rng.normal(0, 0.055, DAYS), np.ones(3) / 3, "same")
    if kind == "failed":
        ramp = 0.62 / (1 + np.exp(-(t - 0.72) * 13))
        return np.clip(base + ramp + noise * 0.6, 0.05, 0.97)
    out = np.clip(base + noise * 0.8, 0.04, 0.95)
    out[16] = 0.60
    out[15], out[17] = 0.40, 0.42
    return out


def lane(ax, yb, h, kind, title, verdict, verdict_frac):
    s = score_curve(kind)
    end = FAIL_DAY + 1 if kind == "failed" else DAYS
    xs = np.array([day_x(d) for d in range(end)])
    ys = yb + s[:end] * h

    ax.text(X0, yb + h + 2.4, title, ha="left", va="bottom", fontsize=8.6,
            color=INK, fontweight="bold")

    if kind == "failed":
        ax.add_patch(Rectangle(
            (day_x(HZ_START), yb), day_x(FAIL_DAY) - day_x(HZ_START), h,
            facecolor=WARM, alpha=0.12, edgecolor="none", zorder=1))
        ax.plot([day_x(HZ_START)] * 2, [yb, yb + h], lw=0.9,
                ls=(0, (2.5, 2.0)), color=WARM, zorder=2)
        ax.text(day_x(HZ_START) + 0.6, yb + h - 1.2, "10-day horizon",
                ha="left", va="top", fontsize=7.0, color=WARM, zorder=5)
        ax.scatter([day_x(FAIL_DAY)], [yb + h], marker="X", s=48, color=WARM,
                   zorder=7, edgecolor="white", linewidth=0.6)

    ax.plot([X0, X1], [yb] * 2, lw=0.7, color="#DDDDDD", zorder=1)
    ax.plot([X0, X1], [yb + THR * h] * 2, lw=1.0, ls=(0, (4.5, 2.5)),
            color=COOL, zorder=3)
    ax.plot(xs, ys, lw=1.5, color=INK, zorder=4, solid_capstyle="round")

    over = np.where(s[:end] >= THR)[0]
    if kind == "failed":
        over = over[(over >= HZ_START) & (over < FAIL_DAY)]
    ax.scatter(xs[over], ys[over], s=17, color=COOL, zorder=6,
               edgecolor="white", linewidth=0.5)

    ax.text(X1 + 1.6, yb + h * verdict_frac, verdict, ha="left", va="center",
            fontsize=8.0, color=INK, fontweight="bold")


def main() -> int:
    fig, ax = plt.subplots(figsize=(8.8, 2.9), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    lane(ax, 58.0, 26.0, "failed", "Disk that fails", "detected", 0.16)
    lane(ax, 16.0, 26.0, "healthy", "Disk that does not fail",
         "false positive", 0.82)

    # 점수 하나가 어디서 오는지. 창의 끝이 예측 시점이다.
    wb, we = day_x(10), day_x(23)
    ax.plot([wb, we], [52.0] * 2, lw=2.6, color="#BFBFBF",
            solid_capstyle="butt", zorder=3)
    for x in (wb, we):
        ax.plot([x] * 2, [50.8, 53.2], lw=1.0, color="#9A9A9A", zorder=3)
    ax.text(wb - 0.8, 52.0, "14-day window", ha="right", va="center",
            fontsize=7.2, color=MUTE)
    ax.add_patch(FancyArrowPatch(
        (we, 53.6), (we, 58.0 + 26.0 * 0.45), arrowstyle="-|>",
        mutation_scale=8, linewidth=0.9, color=MUTE, zorder=4))
    ax.text(we + 1.2, 55.4, "GRU", ha="left", va="center", fontsize=7.4,
            color=MUTE)

    ax.plot([X0, X1], [11.0] * 2, lw=0.8, color=MUTE)
    for d in (0, 10, 20, 30):
        ax.plot([day_x(d)] * 2, [11.0, 10.0], lw=0.8, color=MUTE)
        ax.text(day_x(d), 8.8, "day " + str(max(d, 1)), ha="center", va="top",
                fontsize=6.8, color=MUTE)

    handles = [
        Line2D([], [], color=INK, lw=1.5, label="predicted score"),
        Line2D([], [], color=COOL, lw=1.0, ls=(0, (4.5, 2.5)),
               label="threshold"),
        Line2D([], [], color=COOL, marker="o", ls="none", markersize=4.6,
               label="alarm"),
        Patch(facecolor=WARM, alpha=0.12, label="10-day horizon"),
        Line2D([], [], color=WARM, marker="X", ls="none", markersize=6.4,
               label="failure"),
    ]
    ax.legend(handles=handles, loc="lower right",
              bbox_to_anchor=(1.005, -0.045), ncol=5, frameon=True,
              fontsize=7.0, handlelength=1.5, handletextpad=0.5,
              columnspacing=1.5, borderpad=0.55, edgecolor="#CFCFCF")

    fig.subplots_adjust(left=0.002, right=0.998, top=0.995, bottom=0.005)
    out = ROOT / "results" / "overview.png"
    fig.savefig(out)
    print("[저장] " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
