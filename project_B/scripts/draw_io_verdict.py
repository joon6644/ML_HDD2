"""입력·출력과 디스크 단위 판정 개요도 — 심사의견 #2 에 대응하는 그림.

    python scripts/draw_io_verdict.py            # 점수 캐시가 없으면 재채점 후 그림
    python scripts/draw_io_verdict.py --rescore  # 캐시를 무시하고 다시 채점

실제 사례 하나로 그린다. 논문 모델(tos_proposed / gru_tuned, 시드 42)과 논문 표 1 의
임곗값(검증 월별 분위수, 오탐률 1%)을 쓴다. 저장된 행 단위 예측은 테스트 첫 달
것뿐이라, 사례가 속한 달을 같은 모델로 다시 채점해 results/io_case_scores.parquet
에 둔다. 재채점한 10월 점수는 저장된 예측과 2e-4 이내로 일치했고, 월별 탐지 수도
tos_proposed_long.csv 와 같았다.

    위   아래 점 하나(t = 12월 13일)가 나오는 과정. 같은 날짜 축 위의 한 줄 막대에서
         t 까지 14일은 입력, t 뒤 10일은 예측 지평. 입력은 GRU 를 거쳐 그 점이
         되고, 지평 안의 고장이 레이블 y_t = 1 이 된다
    아래 그 디스크의 일별 점수(로그 척도). 양성 표본의 최댓값을 τ 와 비교

사례는 10J0A057F97G (2025-12-18 고장). 테스트 6개월의 고장 디스크 162대 중 양성
표본 10개가 그 달 안에 다 있고, 탐지되었고, 지평 밖 점수가 전부 τ 아래인 디스크는
14대다. 그중 이 디스크는 12월 13일 SMART 197 이 0 -> 6 으로 늘면서 점수가 1e-4 에서
0.09 로 뛰어 고장 5일 전에 τ 를 넘는다. 판정 규칙과 4.3 의 IG 결과(섹터 결함 속성,
예측 시점 집중)를 함께 보인다. 설명용으로 고른 사례이지 대표 사례가 아니다.
월말에 고장 나면서 지평 밖 점수가 τ 아래인 디스크는 없었다.

가로축은 막대와 같은 범위(t-13 ~ t+10)로 자른다. 달 전체를 그리면 고장 뒤가 비어
그림이 휑해지고, 이 범위면 고장 뒤 빈 칸이 지평 안에만 남는다.

그림 안의 글은 기호와 이름만 둔다. 일련번호, 날짜, 점수와 τ 의 값, SMART 197 의
변화는 캡션이 맡는다. 위아래가 같은 날짜축이라 확대 장치를 쓰지 않는다. t 칸과
점은 점선으로 1:1 을 잇고, 입력은 GRU 를 거쳐 점의 값(score)을, 지평은 점의
레이블(label)을 정하는 두 화살표로 나타낸다.

색은 draw_overview.py 를 따른다. 주황은 사실(고장·양성), 파랑은 모델(입력·임곗값).
그림에는 한글을 쓰지 않는다. 본문 폭(약 16 cm)에 실측 크기로 그린다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LONG = ROOT / "results" / "tos_proposed_long.csv"
CACHE = ROOT / "results" / "io_case_scores.parquet"

DRIVE = "TOSHIBA_20MG07ACA14TA"
EXPERIMENT, MODEL, SEED = "tos_proposed", "gru_tuned", 42
SERIAL = "10J0A057F97G"
TEST_MONTH = "2025-12"
T = date(2025, 12, 13)        # 펼칠 예측 시점

WARM = "#C0601A"
WARM_FILL = "#F6E3D5"
COOL = "#2A5A9E"
COOL_FILL = "#DCE6F3"
INK = "#141414"
MUTE = "#767676"
FAINT = "#BDBDBD"

WIDTH = 6.3                   # 인치, 본문 폭
LOOKBACK = 14
HORIZON = 10
START = T - timedelta(days=LOOKBACK - 1)    # 가로축 첫날 = t-13
DAYS = LOOKBACK + HORIZON                   # t-13 ~ t+10
LOG_LO, LOG_HI = np.log10(4e-5), np.log10(0.2)   # 점수 축 범위
YTICKS = (1e-4, 1e-3, 1e-2, 1e-1)

X0, X1 = 9.0, 97.0            # 날짜 축의 좌우 끝
CW = (X1 - X0) / DAYS         # 하루 칸 폭

BAR_Y, BAR_H = 82.0, 8.5      # 위: 한 줄 막대
LANE_Y, LANE_H = 17.0, 35.0   # 아래: 점수 줄 (바닥이 곧 날짜축)
LEGEND_Y = 3.5

plt.rcParams.update({
    "font.size": 7.5,
    "svg.fonttype": "none",   # SVG 글자를 글자로 남겨 편집할 수 있게
    "pdf.fonttype": 42,
})


def idx(d: date) -> int:
    """가로축 위 칸 번호 (t-13 이 0)."""
    return (d - START).days


def left(d: date) -> float:
    return X0 + CW * idx(d)


def mid(d: date) -> float:
    return left(d) + CW / 2


def ys(score) -> np.ndarray:
    v = (np.log10(score) - LOG_LO) / (LOG_HI - LOG_LO)
    return LANE_Y + np.clip(v, 0, 1) * LANE_H


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.9, color="0.30", zorder=6,
                                 shrinkA=0, shrinkB=0))


def rescore() -> pd.DataFrame:
    """사례가 속한 테스트 달을 논문 모델로 다시 채점한다."""
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from hddpred import config as cfg_mod
    from hddpred import paths
    from hddpred.experiments.runner import Pipeline, prepare_drive
    from hddpred.features import fold as fold_mod
    from hddpred.models import registry
    from export_results import load_part, rescale

    cfg = cfg_mod.load_yaml(paths.CONFIG_DIR / "experiments" / f"{EXPERIMENT}.yaml")
    pipeline = Pipeline.from_experiment(cfg)
    prepared = prepare_drive(DRIVE, pipeline)
    fold = next(f for f in prepared.folds if str(f.test_month)[:7] == TEST_MONTH)
    mcfg = next(m for m in (registry.load_model_config(x) for x in cfg["models"])
                if m["name"] == MODEL)
    run = paths.run_dir(EXPERIMENT, DRIVE, MODEL, SEED, 0)
    model = registry.resolve_class(mcfg["class"]).load(run / "model")
    record = json.loads((run / "fold_metrics.json").read_text(encoding="utf-8"))
    scaler = fold_mod.Scaler.from_state_dict(record["scaler"])
    threads = int(pipeline.preprocessing.get("duckdb", {}).get("threads", 8))

    horizon = int(pipeline.labeling["horizon_days"])

    def score(start, end):
        part = rescale(load_part(prepared, pipeline, "sequence", start, end,
                                 horizon, threads), scaler)
        f = pd.DataFrame({"serial_number": part.serial,
                          "record_date": part.record_date, "y": part.y,
                          "score": model.predict_proba(part),
                          "failure_date": part.failure_date})
        return f[f["serial_number"] == SERIAL]

    # 판정 대상은 사례가 속한 테스트 달 그대로다.
    month = score(*fold.window("test"))
    # 가로축 첫날(t-13)은 앞 달에 속한다. 그 달 채점에서는 월말 검열로 빠지므로
    # 사례 달 끝까지 이어진 창에서 그날 점수만 가져온다. 고장이 18일 뒤라 음성이다.
    lead = score(START, fold.window("test")[1])
    lead = lead[pd.to_datetime(lead["record_date"]).dt.date < fold.window("test")[0]]
    frame = pd.concat([lead, month]).sort_values("record_date")
    frame.to_parquet(CACHE)
    print("[저장] " + str(CACHE))
    return frame


def load_case(force: bool):
    d = rescore() if force or not CACHE.exists() else pd.read_parquet(CACHE)
    d = d.assign(record_date=pd.to_datetime(d["record_date"]).dt.date)
    fail = pd.Timestamp(d["failure_date"].dropna().iloc[0]).date()

    long = pd.read_csv(LONG)
    tau = float(long.query("model == @MODEL and threshold_source == "
                           "'val_quantile' and far_target == 0.01")
                ["threshold"].iloc[0])
    return d, fail, tau


def bar(ax, fail):
    """검은 테두리 막대 두 개(입력 14일, 지평 10일)를 연한 회색 선으로 하루씩 나눈다."""
    tx = left(T + timedelta(days=1))
    y0, h = BAR_Y, BAR_H

    ax.add_patch(Rectangle((X0, y0), tx - X0, h, facecolor=COOL_FILL,
                           edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((tx, y0), X1 - tx, h, facecolor=WARM_FILL,
                           edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((left(T), y0), CW, h, facecolor="#B9CCE6",
                           edgecolor="none", zorder=2))
    for k in range(1, DAYS):
        if k != LOOKBACK:
            x = X0 + k * CW
            ax.plot([x, x], [y0, y0 + h], lw=0.5, color="#C4C4C4", zorder=3)
    for x0, x1 in ((X0, tx), (tx, X1)):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, h, facecolor="none",
                               edgecolor=INK, linewidth=0.7, zorder=4))

    ax.text((X0 + tx) / 2, y0 + h / 2, "input  (14 days × 16 attributes)",
            ha="center", va="center", fontsize=7.0, color=COOL, zorder=6)
    # 고장 선을 피해 t 와 고장 사이에 둔다. 길이는 위 눈금 t+10 이 말한다
    ax.text((tx + mid(fail)) / 2, y0 + h / 2, "horizon",
            ha="center", va="center", fontsize=7.0, color=WARM, zorder=6)

    # 눈금 글자는 그날 칸의 가운데에 둔다. t 는 칸 하나다
    for d, s in ((START, r"$t-13$"), (T, r"$t$"),
                 (T + timedelta(days=HORIZON), r"$t+10$")):
        ax.text(mid(d), y0 + h + 1.6, s, ha="center", va="bottom",
                fontsize=7.0, color=INK if d == T else MUTE)


def failure_line(ax, fail):
    """고장일. 위아래가 같은 날짜축이라 선 하나로 두 줄을 꿰뚫는다."""
    x = mid(fail)
    ax.plot([x, x], [LANE_Y, BAR_Y + BAR_H], lw=1.1, color=WARM, zorder=4)


def lane(ax, d, tau):
    dates = d["record_date"].to_numpy()
    keep = np.array([START <= x < START + timedelta(days=DAYS) for x in dates])
    dates, score = dates[keep], d["score"].to_numpy()[keep]
    positive = d["y"].to_numpy()[keep] == 1
    yb, h = LANE_Y, LANE_H

    # L자 축: 세로축과 날짜축이 원점에서 만난다
    ax.plot([X0, X0, X1], [yb + h, yb, yb], lw=0.7, color=MUTE, zorder=3,
            solid_joinstyle="miter")
    for v in YTICKS:
        y = ys(v)
        ax.plot([X0 - 0.8, X0], [y, y], lw=0.7, color=MUTE)
        ax.text(X0 - 1.2, y, f"{v:g}", ha="right", va="center",
                fontsize=6.4, color=MUTE)
    ax.text(X0 - 7.0, yb + h / 2, "score", rotation=90, ha="center",
            va="center", fontsize=6.8, color=MUTE)
    for dd in (date(2025, 12, 1), date(2025, 12, 10), date(2025, 12, 20)):
        ax.plot([mid(dd)] * 2, [yb, yb - 1.2], lw=0.7, color=MUTE)
        ax.text(mid(dd), yb - 2.0, f"Dec {dd.day}", ha="center", va="top",
                fontsize=6.6, color=MUTE)

    ax.plot([X0, X1], [ys(tau)] * 2, lw=0.9, ls=(0, (4.0, 2.2)), color=COOL,
            zorder=2)

    p0 = min(dates[positive])
    p1 = max(dates[positive]) + timedelta(days=1)
    ax.add_patch(Rectangle((left(p0), yb), left(p1) - left(p0), h,
                           facecolor=WARM_FILL, edgecolor="none", zorder=0))

    xs = np.array([mid(x) for x in dates])
    yv = ys(score)
    ax.plot(xs, yv, lw=0.6, color=FAINT, zorder=2)
    ax.scatter(xs[~positive], yv[~positive], s=9, facecolor="white",
               edgecolor="0.45", linewidth=0.7, zorder=3)
    ax.scatter(xs[positive], yv[positive], s=9, color=INK, linewidth=0,
               zorder=4)

    i = int(np.flatnonzero(dates == T)[0])
    px, py = xs[i], float(yv[i])
    ax.scatter([px], [py], s=26, color=COOL, edgecolor="white",
               linewidth=0.8, zorder=7)
    return px, py


def links(ax, px, py):
    """막대와 점을 잇는다. 같은 날짜축이라 확대가 아니라 두 갈래의 기여다.

        t 칸 -> 점        점선. 이 칸이 이 점이다 (1:1)
        입력 -> GRU -> 점  점의 값(score)
        지평 -> 점        점의 레이블(label). 지평 안에 고장이 있어 양성
    """
    tx = left(T + timedelta(days=1))
    yb = BAR_Y
    ax.plot([px, px], [yb - 0.6, py + 1.6], lw=0.7, color=COOL, alpha=0.55,
            ls=(0, (2.0, 2.0)), zorder=1)

    gw, gh = 8.0, 6.0
    gx, gy = (X0 + tx) / 2 + 2.0, yb - 12.0
    ax.add_patch(FancyBboxPatch((gx - gw / 2, gy - gh / 2), gw, gh,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor="white", edgecolor="0.30",
                                linewidth=0.9, zorder=5))
    ax.text(gx, gy, "GRU", ha="center", va="center", fontsize=8.0,
            color=INK, zorder=6)
    arrow(ax, (gx, yb - 0.6), (gx, gy + gh / 2 + 0.3))
    arrow(ax, (gx + 1.2, gy - gh / 2 - 0.3), (px - 0.5, py + 1.0))
    ax.text((gx + px) / 2 - 1.0, (gy + py) / 2 + 2.6, "score", ha="center",
            va="bottom", fontsize=6.6, color="0.30")

    hx = mid(T + timedelta(days=2))
    ax.add_patch(FancyArrowPatch((hx, yb - 0.6), (px + 0.6, py + 1.2),
                                 arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.9, color=WARM, zorder=6,
                                 shrinkA=0, shrinkB=0))
    ax.text((hx + px) / 2 + 1.4, (yb + py) / 2 + 2.0, "label", ha="left",
            va="center", fontsize=6.6, color=WARM)


def legend(ax):
    """맨 아래 한 줄. 기호 다섯 개."""
    y = LEGEND_Y
    items = [
        ("cell", INK, None, r"prediction time $t$"),
        ("line", WARM, "solid", "failure"),
        ("line", COOL, (0, (4.0, 2.2)), r"threshold $\tau$"),
        ("dot", INK, None, "positive sample"),
        ("hollow", "0.45", None, "negative sample"),
    ]
    widths = [17.0, 10.0, 15.0, 15.0, 15.0]
    x = (100 - sum(widths)) / 2 + 1.0
    for (kind, color, ls, label), w in zip(items, widths):
        if kind == "cell":
            ax.add_patch(Rectangle((x + 0.3, y - 1.6), 1.4, 3.2,
                                   facecolor="#B9CCE6", edgecolor=INK,
                                   linewidth=0.6))
            tx = x + 2.4
        elif kind == "line" and color == WARM:
            ax.plot([x + 1.0, x + 1.0], [y - 1.6, y + 1.6], lw=1.1, color=color)
            tx = x + 2.4
        elif kind == "line":
            ax.plot([x, x + 3.6], [y, y], lw=0.9, ls=ls, color=color)
            tx = x + 4.6
        elif kind == "dot":
            ax.scatter([x + 1.0], [y], s=9, color=color, linewidth=0)
            tx = x + 2.4
        else:
            ax.scatter([x + 1.0], [y], s=9, facecolor="white",
                       edgecolor=color, linewidth=0.7)
            tx = x + 2.4
        ax.text(tx, y, label, ha="left", va="center", fontsize=6.6,
                color=INK)
        x += w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    d, fail, tau = load_case(args.rescore)

    fig, ax = plt.subplots(figsize=(WIDTH, 2.85))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 97)
    ax.axis("off")

    bar(ax, fail)
    px, py = lane(ax, d, tau)
    failure_line(ax, fail)
    links(ax, px, py)
    legend(ax)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for ext, kw in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        out = ROOT / "results" / f"io_overview.{ext}"
        fig.savefig(out, **kw)
        print("[저장] " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
