"""제안 모델의 SHAP 분석 — notion.md 4장의 [SHAP Summary Plot].

    python scripts/run_shap.py
    python scripts/run_shap.py --sample 100000

최종 구성(xgboost_tuned + ASFD 7일, 26피처)을 그대로 쓴다. XGBoost 는 트리
모델이라 TreeSHAP 로 정확한 기여도를 얻는다 — 근사가 아니다.

━━ 무엇을 설명하는가 ━━

test 세 달(2025-12 / 2026-01 / 2026-02)의 행을 무작위 표집해 설명한다.
양성 비율이 0.055% 라 표본에 양성이 적게 들어가지만, 그게 실제 운영에서
모델이 마주하는 분포다. 양성만 따로 뽑아 그리면 "고장 직전 구간에서 무엇이
중요한가" 라는 다른 질문에 답하게 되므로 섞지 않는다.

━━ 시드 ━━

시드 42 한 개로 계산한다. 같은 설정도 시드마다 다른 트리를 만들지만
(실측: 고장 47대 중 28~31대로 변동), 여기서는 대표 모델 하나의 기여도를
보인다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hddpred import config as cfg_mod  # noqa: E402
from hddpred import paths  # noqa: E402
from hddpred.experiments.runner import Pipeline, prepare_drive  # noqa: E402
from hddpred.models import registry  # noqa: E402
from export_results import load_part  # noqa: E402

DRIVE = "HGST_20HUH721212ALN604"
EXPERIMENT = "feat_asfd7_tuned"
MODEL = "xgboost_tuned"
SEED = 42

def main() -> int:
    ap = argparse.ArgumentParser(description="제안 모델 SHAP 분석")
    ap.add_argument("--sample", type=int, default=50000, help="설명할 행 수")
    args = ap.parse_args()

    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = cfg_mod.load_yaml(paths.CONFIG_DIR / "experiments" / f"{EXPERIMENT}.yaml")
    pipeline = Pipeline.from_experiment(cfg)
    prepared = prepare_drive(DRIVE, pipeline)
    folds = sorted(prepared.folds, key=lambda f: f.fold)
    mcfg = next(
        m for m in (registry.load_model_config(x) for x in cfg["models"])
        if m["name"] == MODEL
    )
    horizon = int(pipeline.labeling["horizon_days"])

    print("[shap] test 세 달 적재", flush=True)
    parts = []
    for fold in folds:
        start, end = fold.window("test")
        parts.append(load_part(prepared, pipeline, mcfg["family"], start, end,
                               horizon, 8))
    X = np.concatenate([p.X for p in parts])
    y = np.concatenate([p.y for p in parts])
    # 컬럼명을 그대로 쓴다. SMART 속성의 통용명은 벤더마다 표기가 갈리지만
    # 번호는 명확하고 데이터·코드와 바로 대조된다. 그림에서는 정보가 없는
    # 접미사만 덜어낸다: smart_197_raw -> smart_197, _asfd7 -> _asfd.
    columns = [c.replace("_raw", "").replace("_asfd7", "_asfd")
               for c in parts[0].columns]

    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(args.sample, len(X)), replace=False)
    Xs, ys = X[idx], y[idx]
    print(f"  전체 {len(X):,}행 -> 표본 {len(Xs):,}행 (양성 {int(ys.sum()):,}행, "
          f"{ys.mean():.4%})", flush=True)

    model = registry.resolve_class(mcfg["class"]).load(
        paths.run_dir(EXPERIMENT, DRIVE, MODEL, SEED, 0) / "model"
    )
    shap_plot = shap.TreeExplainer(model._model).shap_values(Xs)
    print(f"  seed {SEED} 완료", flush=True)

    summary = pd.DataFrame({
        "column": columns,
        "mean_abs_shap": np.abs(shap_plot).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    np.savez_compressed(ROOT / "results" / "shap_values.npz",
                        shap=shap_plot, X=Xs, y=ys, columns=np.array(columns))
    out_csv = ROOT / "results" / "shap_importance.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n--- 평균 |SHAP| 상위 12개 (seed {SEED}) ---")
    print(summary.head(12).to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # (1) beeswarm — 값의 크기와 방향을 같이 본다.
    plt.figure()
    shap.summary_plot(shap_plot, Xs, feature_names=columns, max_display=15,
                      show=False, plot_size=(7.0, 5.2))
    plt.tight_layout()
    plt.savefig(ROOT / "results" / "shap_summary.png", dpi=160)
    plt.close()

    # (2) 막대 — 기여도 순위.
    top = summary.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=160)
    ax.barh(top.column, top.mean_abs_shap, color="#c0392b", alpha=0.85)
    ax.set_xlabel("mean |SHAP value|")
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "shap_importance.png")
    plt.close(fig)

    print(f"\n[저장] {ROOT / 'results' / 'shap_summary.png'}")
    print(f"[저장] {ROOT / 'results' / 'shap_importance.png'}")
    print(f"[저장] {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
