"""LSTM 의 Integrated Gradients 분석 — notion.md 4장의 XAI 그림.

    python scripts/run_ig.py --experiment toslb_14 --model lstm
    python scripts/run_ig.py --experiment tos_proposed --model lstm_tuned --sample 20000

LSTM 은 트리 모델이 아니라 TreeSHAP 을 못 쓴다. 대신 Integrated Gradients
(Sundararajan, Taly & Yan, ICML 2017)를 쓴다. 미분 가능한 모델이면 어디에나
적용되고, 아래 완결성 공리로 구현이 맞는지 스스로 검증할 수 있다.

━━ 방법 ━━

    IG_i(x) = (x_i - x'_i) * ∫_0^1 ∂F(x' + a(x - x')) / ∂x_i  da

기준점 x' 는 0 벡터다. 입력이 min-max 로 [0,1] 에 놓여 있으므로 0 은 "train
구간에서 관측된 최솟값", 즉 가장 건강한 상태에 해당한다. 적분은 STEPS 개
구간의 리만 합으로 근사한다.

F 는 시그모이드 이전의 로짓이다. 확률로 미분하면 포화 구간에서 기울기가
0 에 가까워져 기여도가 뭉개진다.

━━ 완결성 검증 ━━

IG 는 sum_i IG_i(x) = F(x) - F(x') 를 만족해야 한다 (공리). 근사 오차가 이
등식에서 얼마나 벗어나는지 매번 출력한다. 크게 어긋나면 STEPS 를 늘려야
한다는 신호다.

━━ 두 가지 축소 ━━

입력이 (lookback, 피처) 라 기여도도 2차원이다. 어느 축으로 접느냐에 따라
다른 질문에 답한다.

  시간축을 접는다  -> 변수별 중요도. |IG| 를 14일에 대해 평균낸다.
                     SHAP summary plot 과 같은 형태로 그린다.
  피처축을 접는다  -> 시점별 중요도. 예측일로부터 며칠 전이 중요한가.
                     선 그래프로 그린다. (추가 분석)

━━ 무엇을 설명하는가 ━━

test 구간 표본을 무작위로 뽑는다. 양성 비율이 낮아 표본에 고장이 적게 들어
가지만, 그게 실제 운영에서 모델이 마주하는 분포다. 양성만 뽑으면 "고장 직전
구간에서 무엇이 중요한가" 라는 다른 질문에 답하게 된다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hddpred import config as cfg_mod  # noqa: E402
from hddpred import paths  # noqa: E402
from hddpred.experiments.runner import Pipeline, prepare_drive  # noqa: E402
from hddpred.features import fold as fold_mod  # noqa: E402
from hddpred.models import registry  # noqa: E402
from export_results import load_part, rescale  # noqa: E402

DRIVE = "TOSHIBA_20MG07ACA14TA"
SEED = 42
STEPS = 32          # 리만 합 구간 수
BATCH = 64          # 표본 배치 (STEPS 배로 늘어나므로 작게 잡는다)


def integrated_gradients(net, x, steps=STEPS):
    """x: (B, T, F) -> IG: (B, T, F). 기준점은 0 벡터.

    cuDNN 의 RNN 커널은 eval 모드에서 역전파를 지원하지 않는다. net.train() 으로
    바꾸면 되지만 그러면 드롭아웃이 켜져 설명 대상 함수 자체가 달라진다. 그래서
    cuDNN 만 끄고 일반 커널로 미분한다 — 느리지만 같은 함수를 설명한다.
    """
    with torch.backends.cudnn.flags(enabled=False):
        total = torch.zeros_like(x)
        for step in range(steps):
            # 중점 규칙. 양 끝점을 쓰는 사다리꼴보다 같은 STEPS 에서 오차가 작다.
            alpha = (step + 0.5) / steps
            point = (alpha * x).detach().requires_grad_(True)
            logit = net(point)
            grad, = torch.autograd.grad(logit.sum(), point)
            total += grad
    return (x * total / steps).detach()


def main() -> int:
    ap = argparse.ArgumentParser(description="LSTM Integrated Gradients")
    ap.add_argument("--experiment", default="toslb_14")
    ap.add_argument("--model", default="lstm")
    ap.add_argument("--sample", type=int, default=20000)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 그림 안의 글자는 전부 영문이다. 한글 폰트가 환경마다 달라 축 라벨이
    # 깨지는 것을 막고, 논문 그림 관례와도 맞는다.

    cfg = cfg_mod.load_yaml(paths.CONFIG_DIR / "experiments" / f"{args.experiment}.yaml")
    pipeline = Pipeline.from_experiment(cfg)
    prepared = prepare_drive(DRIVE, pipeline)
    folds = sorted(prepared.folds, key=lambda f: f.fold)
    horizon = int(pipeline.labeling["horizon_days"])
    mcfg = next(m for m in (registry.load_model_config(x) for x in cfg["models"])
                if m["name"] == args.model)
    run = paths.run_dir(args.experiment, DRIVE, args.model, SEED, 0)
    model = registry.resolve_class(mcfg["class"]).load(run / "model")
    record = json.loads((run / "fold_metrics.json").read_text(encoding="utf-8"))
    scaler = (fold_mod.Scaler.from_state_dict(record["scaler"])
              if record.get("scaler") else None)

    print("[ig] test 구간 적재", flush=True)
    parts = [rescale(load_part(prepared, pipeline, "sequence", *f.window("test"),
                               horizon, 8), scaler) for f in folds]

    # 표본을 달에 걸쳐 고르게 뽑는다.
    rng = np.random.default_rng(0)
    per_month = max(1, args.sample // len(parts))
    net = model.net.eval()
    device = model.device

    igs, values, labels = [], [], []
    completeness = []
    for part in parts:
        take = min(per_month, len(part))
        idx = np.sort(rng.choice(len(part), size=take, replace=False))
        loader = model._loader(part, shuffle=False, batch_size=BATCH)
        # 배치 순서가 인덱스 순서와 같으므로, 필요한 배치만 골라 쓴다.
        cursor, want = 0, set(idx.tolist())
        for batch_x, _ in loader:
            n = batch_x.shape[0]
            local = [i - cursor for i in range(cursor, cursor + n) if i in want]
            cursor += n
            if not local:
                continue
            x = batch_x[local].to(device).float()
            ig = integrated_gradients(net, x)
            with torch.no_grad():
                fx = net(x)
                f0 = net(torch.zeros_like(x))
            completeness.append(
                (ig.sum(dim=(1, 2)) - (fx - f0)).abs().cpu().numpy()
            )
            igs.append(ig.cpu().numpy())
            values.append(x.cpu().numpy())
        labels.append(part.y[idx])

    ig = np.concatenate(igs)              # (N, T, F)
    val = np.concatenate(values)
    err = np.concatenate(completeness)
    scale = np.abs(ig.sum(axis=(1, 2)))
    print(f"  표본 {ig.shape[0]:,}개, 입력 ({ig.shape[1]}일 x {ig.shape[2]}채널)")
    print(f"  완결성 오차 |sum(IG) - (F(x)-F(0))| : 중앙값 {np.median(err):.3e}, "
          f"상대 {np.median(err / np.maximum(scale, 1e-12)):.2%}")

    columns = list(parts[0].columns)
    if ig.shape[2] == len(columns) + 1:
        columns = columns + ["_mask"]     # pad 모드에서 붙는 채널
    clean = [c.replace("smart_", "").replace("_raw", "") for c in columns]

    # --- (1) 시간축을 접어 변수별 중요도 ---------------------------------
    per_feature = np.abs(ig).mean(axis=1)          # (N, F)
    importance = per_feature.mean(axis=0)
    order = np.argsort(-importance)[:15]

    np.savez_compressed(ROOT / "results" / "ig_values.npz",
                        ig=ig.astype(np.float32), x=val.astype(np.float32),
                        columns=np.array(columns))
    import pandas as pd
    pd.DataFrame({"column": [clean[i] for i in np.argsort(-importance)],
                  "mean_abs_ig": importance[np.argsort(-importance)]}).to_csv(
        ROOT / "results" / "ig_importance.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200)
    rows = order[::-1]
    for row, feat in enumerate(rows):
        v = per_feature[:, feat]
        # 색은 예측일(마지막 시점)의 그 변수 값. SHAP summary 의 색과 같은 역할.
        c = val[:, -1, feat]
        lo, hi = np.percentile(c, [5, 95])
        c = np.clip((c - lo) / (hi - lo), 0, 1) if hi > lo else np.full_like(c, 0.5)
        jitter = (rng.random(v.shape[0]) - 0.5) * 0.34
        ax.scatter(v, row + jitter, c=c, cmap="coolwarm", s=2.2,
                   alpha=0.45, linewidths=0, rasterized=True)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([clean[i] for i in rows])
    ax.set_xlabel("mean |Integrated Gradients| over lookback")
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap="coolwarm")
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["Low", "High"])
    cb.set_label("Feature value at prediction day", rotation=270, labelpad=14)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "ig_summary.png")
    plt.close(fig)

    # --- (2) 피처축을 접어 시점별 중요도 (추가 분석) ----------------------
    per_time = np.abs(ig).mean(axis=2)             # (N, T)
    lookback = ig.shape[1]
    days = np.arange(lookback - 1, -1, -1)         # 예측일로부터 며칠 전
    mean = per_time.mean(axis=0)
    lo = np.percentile(per_time, 25, axis=0)
    hi = np.percentile(per_time, 75, axis=0)

    pd.DataFrame({"days_before": days, "mean_abs_ig": mean,
                  "q25": lo, "q75": hi}).to_csv(
        ROOT / "results" / "ig_time_importance.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    ax.fill_between(days, lo, hi, color="#2e6fb7", alpha=0.15, lw=0)
    ax.plot(days, mean, color="#2e6fb7", lw=2.0, marker="o", ms=3.5)
    ax.invert_xaxis()                              # 왼쪽이 과거
    ax.set_xlabel("Days before prediction (0 = prediction day)")
    ax.set_ylabel("mean |Integrated Gradients|")
    ax.grid(alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "ig_time_importance.png")
    plt.close(fig)

    print(f"\n--- 변수별 중요도 상위 10 ---")
    for i in np.argsort(-importance)[:10]:
        print(f"  {clean[i]:<16}{importance[i]:.6f}")
    print(f"\n--- 시점별 중요도 (예측일로부터) ---")
    for day, value in zip(days, mean):
        print(f"  {day:>2}일 전  {value:.6f}")
    for name in ("ig_summary.png", "ig_time_importance.png",
                 "ig_importance.csv", "ig_time_importance.csv"):
        print(f"[저장] {ROOT / 'results' / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
