"""방법 파이프라인 개요도 — 심사의견 3-2 에 대응하는 그림.

    python scripts/draw_pipeline.py

심사위원이 "제안하는 예측 모델의 전체적인 구성과 동작 원리"를 물었다. 논문의
장 구성도가 아니라 데이터가 모델을 통과해 판정까지 가는 경로를 그린다.

일반적인 ML 흐름도로는 짐작이 안 되는 넷만 남긴다.
  1. 입력이 14일 x 16속성 창이고, 레이블은 창의 끝(예측 시점)에서 10일 앞을 본다
  2. 분할이 시간순이고, 임곗값은 검증에서 정해 테스트에 고정된다
  3. 탐색 목적함수가 저오탐률 구간 pAUC 다
  4. 판정이 행 단위가 아니라 디스크 단위 월별 집계다

속성 93->16 선택, 결측 보간, 세그먼트 분리, 조기 종료는 본문에 있고 독자가
메울 수 있으므로 넣지 않는다. 스케일링과 비교 모델은 박스 대신 주석 한 줄로
둔다. 그림과 표에는 한글을 쓰지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

EDGE = "0.30"
FACE = "0.965"
YC = 0.56           # 본선 박스의 세로 중심
BH = 0.50           # 본선 박스 높이

# 박스 좌우 경계. 사이 간격이 흐름 화살표 자리다.
SLOTS = [(0.004, 0.258), (0.300, 0.458), (0.500, 0.722), (0.764, 0.906),
         (0.930, 0.998)]


def stage(ax, x0, x1, title, body, *, face=FACE, title_size=9.0,
          body_size=7.4):
    """본선 박스 하나. 제목을 위에 두고 본문을 남은 공간 중앙에 놓는다."""
    ax.add_patch(FancyBboxPatch(
        (x0, YC - BH / 2), x1 - x0, BH,
        boxstyle="round,pad=0.003,rounding_size=0.014",
        linewidth=1.0, edgecolor=EDGE, facecolor=face, zorder=2))
    xm, top, bot = (x0 + x1) / 2, YC + BH / 2, YC - BH / 2
    ax.text(xm, top - 0.075, title, ha="center", va="center",
            fontsize=title_size, zorder=3)
    ax.text(xm, (top - 0.145 + bot) / 2, "\n".join(body), ha="center",
            va="center", fontsize=body_size, color="0.15",
            linespacing=1.6, zorder=3)
    return xm


def flow(ax, x_from, x_to):
    ax.add_patch(FancyArrowPatch(
        (x_from, YC), (x_to, YC), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.1, color="0.25", zorder=4))


def draw_input(ax, x0, x1):
    """입력 창 — 격자로 (14일 x 속성) 을 보이고 예측 지평을 화살표로 붙인다."""
    ax.add_patch(FancyBboxPatch(
        (x0, YC - BH / 2), x1 - x0, BH,
        boxstyle="round,pad=0.003,rounding_size=0.014",
        linewidth=1.0, edgecolor=EDGE, facecolor="white", zorder=2))
    ax.text((x0 + x1) / 2, YC + BH / 2 - 0.075, "Input window",
            ha="center", va="center", fontsize=9.0, zorder=3)

    # 격자. 14열(시점) x 5행. 실제 속성 수 16 은 왼쪽 라벨로 적는다.
    gx0, gx1 = x0 + 0.044, x1 - 0.060
    gy1, rows, cols = YC + BH / 2 - 0.150, 5, 14
    cw, ch = (gx1 - gx0) / cols, 0.032
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle(
                (gx0 + c * cw, gy1 - (r + 1) * ch), cw, ch,
                linewidth=0.45, edgecolor="0.55",
                facecolor="0.70" if c == cols - 1 else "0.93", zorder=3))
    gy0 = gy1 - rows * ch
    ax.text(x0 + 0.026, gy1 - rows * ch / 2, "16\nattr.", ha="center",
            va="center", fontsize=6.0, color="0.25", linespacing=1.3, zorder=3)
    ax.text((gx0 + gx1) / 2, gy0 - 0.040, "14 days", ha="center", va="center",
            fontsize=6.8, color="0.25", zorder=3)
    ax.text(gx1 - cw / 2, gy1 + 0.026, "t", ha="center", va="center",
            fontsize=7.4, zorder=3)

    # 예측 지평: t 에서 앞으로 10일. 창 밖을 향한다는 것이 요점이다.
    hy = gy0 - 0.092
    ax.add_patch(FancyArrowPatch(
        (gx1 - cw / 2, gy0 - 0.008), (gx1 - cw / 2, hy + 0.006),
        arrowstyle="-", linewidth=0.7, color="0.45", zorder=3))
    ax.add_patch(FancyArrowPatch(
        (gx1 - cw / 2, hy), (x1 - 0.012, hy), arrowstyle="-|>",
        mutation_scale=9, linewidth=1.1, color="0.15", zorder=3))
    ax.text((x0 + x1) / 2 - 0.004, hy - 0.062,
            "failure within 10 days after t  →  positive",
            ha="center", va="center", fontsize=6.9, color="0.15", zorder=3)


def main() -> int:
    fig, ax = plt.subplots(figsize=(9.2, 2.35), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 0.97)
    ax.axis("off")

    draw_input(ax, *SLOTS[0])
    stage(ax, *SLOTS[1], "Chronological split",
          ["train 10 months", "validation 2 months", "test 6 months",
           "(2024-10 – 2026-03)"])
    xm_gru = stage(ax, *SLOTS[2], "GRU training",
                   ["Optuna TPE, 50 trials", "objective: validation pAUC",
                    "at FPR ≤ 5%"])
    stage(ax, *SLOTS[3], "Disk-level scoring",
          ["monthly aggregation", "threshold from", "validation quantile,",
           "fixed over test"])
    stage(ax, *SLOTS[4], "Metrics",
          ["Recall", "FPR", "pAUC", "ROC-AUC"], face="white", body_size=7.2)

    for (_, a), (b, _) in zip(SLOTS[:-1], SLOTS[1:]):
        flow(ax, a, b)

    # 탐색 되돌림 — 박스 대신 화살표로 "설정을 바꿔 다시 학습한다" 를 보인다.
    top = YC + BH / 2
    ax.add_patch(FancyArrowPatch(
        (xm_gru + 0.062, top + 0.028), (xm_gru - 0.062, top + 0.028),
        connectionstyle="arc3,rad=0.50", arrowstyle="-|>", mutation_scale=9,
        linewidth=1.0, color="0.30", zorder=4))

    # 100 Scaling note: the gap between boxes is too narrow for this text,
    # so it sits under the split box instead. "fit on train" is the point.
    ax.text((SLOTS[1][0] + SLOTS[1][1]) / 2, YC - BH / 2 - 0.036,
            "inputs scaled by min-max (fit on train)", ha="center",
            va="top", fontsize=6.6, color="0.30")

    # 비교 모델: 레인을 나누지 않고 한 줄로 갈음한다.
    bot = YC - BH / 2
    ax.text(xm_gru, bot - 0.036,
            "same path for LSTM, XGBoost, LightGBM, MLP",
            ha="center", va="top", fontsize=6.6, color="0.30")

    # IG: 학습된 모델에서 갈라지는 사후 분석이므로 점선.
    ax.add_patch(FancyArrowPatch(
        (xm_gru, bot - 0.082), (xm_gru, bot - 0.140), arrowstyle="-|>",
        mutation_scale=9, linewidth=1.0, linestyle=(0, (2.2, 1.6)),
        color="0.30", zorder=4))
    ax.text(xm_gru, bot - 0.162,
            "Integrated Gradients  (failure-imminent positive windows)",
            ha="center", va="top", fontsize=6.9, color="0.15")

    fig.subplots_adjust(left=0.004, right=0.998, top=0.99, bottom=0.01)
    out = ROOT / "results" / "pipeline.png"
    fig.savefig(out)
    print(f"[저장] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
