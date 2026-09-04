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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hddpred import config as cfg_mod  # noqa: E402
from hddpred import paths  # noqa: E402
from hddpred.experiments.runner import Pipeline, prepare_drive  # noqa: E402
from hddpred.models import registry  # noqa: E402
from export_results import disk_rank, load_part  # noqa: E402

DRIVE = "HGST_20HUH721212ALN604"
SEED = 42
RULE = "in_horizon"
FAR_MARK = 0.01  # 표가 읽는 운영점
# (범례, experiment, model, 색, 선굵기)
ARMS = [
    ("Baseline", "feat_raw_only", "xgboost", "#7f8c8d", 1.6),
    ("+ Feature", "feat_asfd7", "xgboost", "#2e6fb7", 1.8),
    ("Proposed", "feat_asfd7_tuned", "xgboost_tuned", "#c0392b", 2.2),
]
GRID = np.logspace(np.log10(1e-4), 0.0, 300)  # 0.01% ~ 100%


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
    scores, failed = [], []
    for fold in folds:
        start, end = fold.window("test")
        part = load_part(prepared, pipeline, mcfg["family"], start, end, horizon, 8)
        rank, has_window = disk_rank(model, part, RULE)
        scores.append(rank.to_numpy())
        failed.append(has_window.to_numpy())
    return np.concatenate(scores), np.concatenate(failed).astype(bool)


def curve(scores: np.ndarray, failed: np.ndarray) -> np.ndarray:
    """목표 FAR 격자마다 Recall 을 읽는다."""
    healthy = scores[~failed]
    broken = scores[failed]
    # 미고장 점수의 상위 f 지점. f 가 곧 FAR 이 된다.
    thresholds = np.quantile(healthy, 1.0 - GRID)
    return (broken[:, None] >= thresholds[None, :]).mean(axis=0)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    results = {}
    for label, experiment, model_name, _, _ in ARMS:
        print(f"[curve] {label} ({experiment} / {model_name})", flush=True)
        scores, failed = disk_scores(experiment, model_name)
        results[label] = curve(scores, failed)
        at = float(np.interp(FAR_MARK, GRID, results[label]))
        print(f"  고장 {int(failed.sum())}대 / 정상 {int((~failed).sum()):,}대"
              f" | FAR {FAR_MARK:.0%} 에서 Recall {at:.3f}", flush=True)

    np.savez_compressed(ROOT / "results" / "recall_far_curve.npz",
                        grid=GRID, **results)

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    ax.axvline(FAR_MARK * 100, color="#b0b0b0", lw=0.9, ls=":", zorder=1)
    for label, _, _, color, lw in ARMS:
        ax.plot(GRID * 100, results[label], color=color, lw=lw, label=label, zorder=3)

    ax.set_xscale("log")
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Alarm Rate (%, log scale)")
    ax.set_ylabel("Recall")
    ax.set_xticks([0.01, 0.1, 1, 10, 100])
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:g}" if v >= 1 else f"{v}".rstrip("0").rstrip(".")))
    ax.grid(True, which="major", alpha=0.25, lw=0.6)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = ROOT / "results" / "recall_far_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\n[저장] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
