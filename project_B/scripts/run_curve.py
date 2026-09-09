"""오탐률에 따른 고장 탐지 성능 곡선 — notion.md 4.3 의 그림.

    python scripts/run_curve.py

4.2 표의 세 구성(Baseline / + Feature / Proposed)을 한 그림에 겹친다. 표가
FAR 1% 한 점을 읽는다면, 이 그림은 운영점 전체를 보여준다.

━━ 무엇을 그리는가 ━━

test 세 달(2025-12 / 2026-01 / 2026-02)을 하나의 모집단으로 통합한 뒤
(3.3 의 평가 단위 정의와 같다), 미고장 디스크 점수의 상위 f 지점을 임곗값으로
잡아가며 f 를 훑는다. 즉 x 는 목표 FAR, y 는 그때의 Recall 이다.

표와 달리 여기서는 val 임곗값을 쓰지 않는다. 곡선은 "임곗값을 어디에 두든"
얻을 수 있는 성능의 궤적이므로, 임곗값 선택 자체가 변수가 아니다. 표의
운영점(FAR 1%)이 이 곡선 위 어디쯤인지 세로선으로 같이 표시한다.

━━ 시드 ━━

시드 42 하나다. 논문의 모든 수치와 같은 기준이다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hddpred import config as cfg_mod  # noqa: E402
from hddpred import paths  # noqa: E402
from hddpred.experiments.runner import Pipeline, prepare_drive  # noqa: E402
from hddpred.models import registry  # noqa: E402
from export_results import disk_rank, load_part, rescale  # noqa: E402
from hddpred.features import fold as fold_mod  # noqa: E402

DRIVE = "TOSHIBA_20MG07ACA14TA"
SEED = 42
RULE = "in_horizon"
FAR_MARK = 0.01  # 표가 읽는 운영점
# (범례, experiment, model, 색, 선굵기, 선모양)
#
# 튜닝 전후 두 곡선을 겹친다. 목적함수가 FPR 5% 이하만 최대화하므로,
# 저오탐 구간에서는 제안 모델이 위에 있다가 그 위에서 교차하는 모양이
# 나와야 한다. 그 교차가 곧 "그 위 구간은 미보장" 의 증거다.
#
# 범례는 표 1 의 Model 열 표기를 그대로 쓴다. 그림과 표에서 같은 모델이
# 다른 이름으로 불리면 안 된다.
ARMS = [
    ("GRU",           "toslb_14_pauc", "gru_pauc",  "#8a8a8a", 1.6, "--"),
    ("Optimized GRU", "tos_proposed",  "gru_tuned", "#2e6fb7", 2.0, "-"),
]
# 관심 구간만 본다. pAUC@FAR<=5% 가 적분하는 범위와 같아서, 그림의 곡선
# 아래 면적이 곧 그 지표가 된다.
GRID = np.linspace(2e-4, 0.05, 300)   # 0.02% ~ 5%


def disk_scores(experiment: str, model_name: str):
    """세 달의 디스크 점수를 하나로 잇는다. (점수, 고장여부) 를 준다."""
    cfg = cfg_mod.load_yaml(paths.CONFIG_DIR / "experiments" / f"{experiment}.yaml")
    pipeline = Pipeline.from_experiment(cfg)
    prepared = prepare_drive(DRIVE, pipeline)
    folds = sorted(prepared.folds, key=lambda f: f.fold)
    mcfg = next(
        m for m in (registry.load_model_config(x) for x in cfg["models"])
        if m["name"] == model_name
    )
    horizon = int(pipeline.labeling["horizon_days"])
    # 모델은 한 번만 학습한다(fold00). 세 달은 그 모델로 각각 예측한다.
    model = registry.resolve_class(mcfg["class"]).load(
        paths.run_dir(experiment, DRIVE, model_name, SEED, 0) / "model"
    )
    # 시퀀스 계열은 학습 때 fold train 구간 스케일러를 거쳤다. 채점에도
    # 같은 스케일러를 먹여야 한다 — 빼먹으면 모델이 상수를 뱉는다.
    record = json.loads(
        (paths.run_dir(experiment, DRIVE, model_name, SEED, 0)
         / "fold_metrics.json").read_text(encoding="utf-8"))
    scaler = (fold_mod.Scaler.from_state_dict(record["scaler"])
              if record.get("scaler") else None)
    months = []
    for fold in folds:
        start, end = fold.window("test")
        part = load_part(prepared, pipeline, mcfg["family"], start, end, horizon, 8)
        if mcfg["family"] == "sequence":
            part = rescale(part, scaler)
        rank, has_window = disk_rank(model, part, RULE)
        months.append((rank.to_numpy(), has_window.to_numpy().astype(bool)))
    return months


def curve(months) -> np.ndarray:
    """목표 FAR 격자마다 Recall 을 읽고 달끼리 평균한다.

    3.4 의 집계 규칙과 같다 — 달마다 따로 산출한 뒤 평균이지 풀링이 아니다.
    임곗값도 달마다 그 달의 미고장 점수 분위수로 잡으므로, 실현 FAR 이 전
    구간에서 격자값과 일치한다.
    """
    per_month = []
    for scores, failed in months:
        thresholds = np.quantile(scores[~failed], 1.0 - GRID)
        per_month.append((scores[failed][:, None] >= thresholds[None, :]).mean(axis=0))
    return np.asarray(per_month)          # (달, 격자)


def main() -> int:
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description="FAR-재현율 곡선")
    ap.add_argument("--replot", action="store_true",
                    help="results/recall_far_curve.npz 로 그림만 다시 그린다")
    args = ap.parse_args()

    results = {}
    if args.replot:
        z = np.load(ROOT / "results" / "recall_far_curve.npz")
        # 범례 이름을 바꿔도 예전 npz 를 계속 쓸 수 있게, 키가 없으면
        # experiment/model 로 저장된 별칭을 찾는다.
        alias = {"Optimized GRU": "GRU (tuned)", "GRU": "GRU (default)"}
        for label, _, _, _, _, _ in ARMS:
            key = label if label in z.files else alias.get(label, label)
            results[label] = z[key]
        print(f"[curve] npz 재사용: {list(results)}", flush=True)
    for label, experiment, model_name, _, _, _ in (
            [] if args.replot else ARMS):
        print(f"[curve] {label} ({experiment} / {model_name})", flush=True)
        months = disk_scores(experiment, model_name)
        results[label] = curve(months)
        at = float(np.interp(FAR_MARK, GRID, results[label].mean(axis=0)))
        nf = sum(int(f.sum()) for _, f in months)
        nh = sum(int((~f).sum()) for _, f in months)
        print(f"  고장 {nf}대 / 정상 {nh:,}대 (월 {len(months)}창)"
              f" | FAR {FAR_MARK:.0%} 에서 Recall {at:.3f}", flush=True)

    if not args.replot:
        np.savez_compressed(ROOT / "results" / "recall_far_curve.npz",
                            grid=GRID, **results)

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    for label, _, _, color, lw, ls in ARMS:
        # 선은 여섯 달의 평균. 달별 최소~최대 띠는 그리지 않는다 — 달 간 폭이
        # 두 곡선의 간격보다 훨씬 커서 띠를 얹으면 튜닝 전후 차이가 묻힌다.
        # 달별 변동 수치는 본문에서 따로 보고한다.
        ax.plot(GRID * 100, results[label].mean(axis=0), color=color, lw=lw,
                ls=ls, label=label, zorder=3)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate (%)")
    ax.set_ylabel("Recall")
    ax.grid(True, which="major", alpha=0.25, lw=0.6)
    # 네모 박스: 네 변을 모두 남긴다.
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
    # 곡선이 둘이라 범례가 필요하다. 곡선이 오른쪽 위로 붙으므로 아래가 빈다.
    ax.legend(loc="lower right", frameon=True, framealpha=0.95,
              edgecolor="0.8", fontsize=9)
    fig.tight_layout()
    out = ROOT / "results" / "recall_far_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\n[저장] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
