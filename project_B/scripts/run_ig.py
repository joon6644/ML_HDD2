"""GRU 의 Integrated Gradients 분석 — notion.md 4장의 XAI 그림.

    python scripts/run_ig.py                       # 기본: tos_proposed / gru_tuned
    python scripts/run_ig.py --experiment toslb_14 --model gru --sample 20000

GRU 는 트리 모델이 아니라 TreeSHAP 을 못 쓴다. 대신 Integrated Gradients
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


def integrated_gradients(net, x, base, steps=STEPS):
    """x: (B, T, F) -> IG: (B, T, F). base 는 기준점 (x 와 브로드캐스트).

    cuDNN 의 RNN 커널은 eval 모드에서 역전파를 지원하지 않는다. net.train() 으로
    바꾸면 되지만 그러면 드롭아웃이 켜져 설명 대상 함수 자체가 달라진다. 그래서
    cuDNN 만 끄고 일반 커널로 미분한다 — 느리지만 같은 함수를 설명한다.
    """
    with torch.backends.cudnn.flags(enabled=False):
        total = torch.zeros_like(x)
        for step in range(steps):
            # 중점 규칙. 양 끝점을 쓰는 사다리꼴보다 같은 STEPS 에서 오차가 작다.
            alpha = (step + 0.5) / steps
            point = (base + alpha * (x - base)).detach().requires_grad_(True)
            logit = net(point)
            grad, = torch.autograd.grad(logit.sum(), point)
            total += grad
    return ((x - base) * total / steps).detach()


def expected_gradients(net, x, base_pool, draws=STEPS):
    """Expected Gradients — 학습 표본을 기준점으로 뽑아 IG 를 평균한다.

        EG_i(x) = E[ (x_i - x'_i) * dF(x' + a(x-x'))/dx_i ],  x' ~ 학습분포, a ~ U(0,1)

    고정 기준점 하나에 32 스텝을 쓰는 대신 기준점 32개에 무작위 a 를 하나씩
    쓴다. 기울기 계산 횟수가 같아 비용이 동일하고, 기준점 선택의 자의성이
    사라진다. 0 기준점이 값이 큰 변수를, 평균 기준점이 퍼짐이 큰 변수를
    부풀리던 편향을 함께 완화한다.
    """
    with torch.backends.cudnn.flags(enabled=False):
        total = torch.zeros_like(x)
        gen = torch.Generator(device="cpu").manual_seed(SEED)
        for k in range(draws):
            base = base_pool[k % base_pool.shape[0]].unsqueeze(0).expand_as(x)
            alpha = torch.rand(x.shape[0], 1, 1, generator=gen).to(x.device)
            point = (base + alpha * (x - base)).detach().requires_grad_(True)
            grad, = torch.autograd.grad(net(point).sum(), point)
            total += (x - base) * grad
    return (total / draws).detach()


def main() -> int:
    ap = argparse.ArgumentParser(description="GRU Integrated Gradients")
    ap.add_argument("--experiment", default="tos_proposed")
    ap.add_argument("--model", default="gru_tuned")
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--draws", type=int, default=STEPS,
                    help="Expected Gradients 의 (기준점, alpha) 표본 수")
    ap.add_argument("--baseline", default="zero",
                    choices=["zero", "train_mean", "expected"],
                    help="기준점. zero=min-max 의 최솟값, train_mean=학습 구간 평균, "
                         "expected=Expected Gradients (학습 표본에서 뽑아 평균)")
    ap.add_argument("--positives-only", action="store_true",
                    help="고장 임박(y=1) 표본만 설명한다. 답하는 질문이 다르다 — "
                         "'운영 분포에서 무엇이 기여하나' 가 아니라 "
                         "'고장 직전에 무엇이 모델을 움직였나' 가 된다")
    ap.add_argument("--drop-mask", action="store_true",
                    help="패딩 표시 채널(_mask)을 그림에서 뺀다")
    ap.add_argument("--replot", action="store_true",
                    help="results/ig_values.npz 로 그림만 다시 그린다")
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

    # 논문용(tos_proposed) 산출물을 절제 실험이 덮어쓰지 않도록 접미사를 붙인다.
    SUF = "" if args.experiment == "tos_proposed" else f"_{args.experiment}"
    if args.positives_only:
        SUF += "_pos"
    args.suffix = SUF
    args.cache_name = {"zero": f"ig_values{SUF}.npz",
                       "train_mean": f"ig_values_mean{SUF}.npz",
                       "expected": f"ig_values_eg{SUF}.npz"}[args.baseline]
    cache = ROOT / "results" / args.cache_name
    if args.replot:
        z = np.load(cache, allow_pickle=True)
        ig, val = z["ig"], z["x"]
        columns = [str(c) for c in z["columns"]]
        print(f"[ig] {cache.name} 재사용, 표본 {ig.shape[0]:,}개")
        rng = np.random.default_rng(0)
        return draw(ig, val, columns, rng, args.drop_mask, args.suffix)

    base_pool = None
    if args.baseline == "expected":
        # Expected Gradients (Erion et al., 2019): 고정 기준점 하나 대신
        # 학습 분포에서 뽑은 실제 표본들을 기준점으로 쓰고 평균한다.
        # 기울기 계산 횟수는 IG 의 적분 스텝 수와 같으므로 비용이 동일하다.
        from hddpred.experiments.runner import _load_part
        from hddpred.models.sequence import WindowDataset
        threads = int(pipeline.preprocessing.get("duckdb", {}).get("threads", 8))
        tr = _load_part(prepared, folds[0], "train", "sequence",
                        pipeline.features, pipeline.labeling, threads)
        tr.matrix = scaler.transform(tr.matrix, fill_nan=True)
        ds = WindowDataset(tr.matrix, tr.end_index,
                           int(pipeline.features["sequence"]["lookback_days"]),
                           None, getattr(tr, "valid_len", None))
        pick = np.random.default_rng(SEED).choice(len(ds), size=args.draws, replace=False)
        base_pool = torch.stack([ds[int(i)][0] for i in pick]).to(model.device).float()
        del tr, ds
        with torch.no_grad():
            base_logit = model.net.eval()(base_pool).mean()
        print(f"[ig] Expected Gradients — 학습 표본 {base_pool.shape[0]}개를 "
              f"기준점으로 사용 {tuple(base_pool.shape)}, "
              f"E[F(x')] = {float(base_logit):.4f}", flush=True)
        base_vec = None
    elif args.baseline == "train_mean":
        # 학습 구간 입력 창들의 성분별 평균을 기준점으로 쓴다.
        #
        # matrix 의 평균이 아니라 WindowDataset 이 만든 창의 평균이어야 한다.
        # mask 채널은 matrix 에 없고 창을 만들 때 붙으므로, matrix 평균만 쓰면
        # mask 기준점이 0 이 되어 그 채널만 0 기준점으로 남는다.
        from hddpred.experiments.runner import _load_part
        from hddpred.models.sequence import WindowDataset
        threads = int(pipeline.preprocessing.get("duckdb", {}).get("threads", 8))
        tr = _load_part(prepared, folds[0], "train", "sequence",
                        pipeline.features, pipeline.labeling, threads)
        tr.matrix = scaler.transform(tr.matrix, fill_nan=True)
        ds = WindowDataset(tr.matrix, tr.end_index,
                           int(pipeline.features["sequence"]["lookback_days"]),
                           None, getattr(tr, "valid_len", None))
        take = min(len(ds), 50000)
        pick = np.random.default_rng(SEED).choice(len(ds), size=take, replace=False)
        acc = None
        for i in pick:
            w = ds[int(i)][0].numpy()
            acc = w.astype(np.float64) if acc is None else acc + w
        base_mat = (acc / take).astype(np.float32)          # (T, F[+mask])
        del tr, ds
        base_vec = base_mat
        print(f"[ig] 기준점 = 학습 창 {take:,}개의 성분별 평균 "
              f"{base_mat.shape}, 범위 {base_mat.min():.3f}~{base_mat.max():.3f}",
              flush=True)
    else:
        base_vec = None
        print("[ig] 기준점 = 0 벡터 (min-max 의 최솟값)", flush=True)

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
        pool = np.where(part.y == 1)[0] if args.positives_only else np.arange(len(part))
        take = min(per_month, len(pool))
        idx = np.sort(rng.choice(pool, size=take, replace=False))
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
            if base_pool is not None:
                ig = expected_gradients(net, x, base_pool, args.draws)
                with torch.no_grad():
                    fx = net(x)
                    f0 = base_logit.expand_as(fx)
                completeness.append(
                    (ig.sum(dim=(1, 2)) - (fx - f0)).abs().cpu().numpy())
                igs.append(ig.cpu().numpy())
                values.append(x.cpu().numpy())
                continue
            if base_vec is None:
                base = torch.zeros_like(x)
            else:
                b = torch.as_tensor(base_vec, device=device, dtype=x.dtype)
                base = b.unsqueeze(0).expand_as(x).contiguous()
            ig = integrated_gradients(net, x, base)
            with torch.no_grad():
                fx = net(x)
                f0 = net(base)
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
    np.savez_compressed(ROOT / "results" / args.cache_name,
                        ig=ig.astype(np.float32), x=val.astype(np.float32),
                        columns=np.array(columns))
    return draw(ig, val, columns, rng, args.drop_mask, args.suffix)


def draw(ig, val, columns, rng, drop_mask=False, suffix=""):
    """저장된 IG 로 그림 두 장과 CSV 두 개를 만든다."""
    import matplotlib.pyplot as plt
    import pandas as pd
    clean = [c.replace("smart_", "").replace("_raw", "") for c in columns]

    # --- (1) 시간축을 접어 변수별 중요도 ---------------------------------
    per_feature = np.abs(ig).mean(axis=1)          # (N, F)
    importance = per_feature.mean(axis=0)
    order = np.argsort(-importance)[:15]

    pd.DataFrame({"column": [clean[i] for i in np.argsort(-importance)],
                  "mean_abs_ig": importance[np.argsort(-importance)]}).to_csv(
        ROOT / "results" / f"ig_importance{suffix}.csv", index=False, encoding="utf-8-sig")

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
    # 197 처럼 드물게 매우 큰 기여를 내는 변수가 있어, 그대로 두면 x 축이
    # 그쪽에 끌려가 나머지 분포가 0 근처에 뭉갠다. 상위 0.1% 지점에서 자른다.
    ax.set_xlim(0, float(np.percentile(per_feature, 99.9)) * 1.2)
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
    fig.savefig(ROOT / "results" / f"ig_summary{suffix}.png")
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
        ROOT / "results" / f"ig_time_importance{suffix}.csv", index=False, encoding="utf-8-sig")

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
    fig.savefig(ROOT / "results" / f"ig_time_importance{suffix}.png")
    plt.close(fig)

    print(f"\n--- 변수별 중요도 상위 10 ---")
    for i in np.argsort(-importance)[:10]:
        print(f"  {clean[i]:<16}{importance[i]:.6f}")
    print(f"\n--- 시점별 중요도 (예측일로부터) ---")
    for day, value in zip(days, mean):
        print(f"  {day:>2}일 전  {value:.6f}")
    # --- (3) 두 축을 함께: 시점 x 변수 히트맵 -----------------------------
    # 행이 변수, 열이 시점이다. 이 한 장이 "어떤 변수가" 와 "언제" 를 같이
    # 담는다. 색조는 제곱근 — 변수 간 크기가 100 배 넘게 벌어져 선형으로는
    # 최상위 한둘이 눈금을 독점하고, 로그까지 가면 하위권 잡음이 구조처럼
    # 보인다. 그 사이 지점이다.
    from matplotlib.colors import PowerNorm
    from matplotlib.ticker import FormatStrFormatter

    # 축 라벨용 이름. 대부분 벤더 공통 정의지만 226 은 제조사마다 갈리므로
    # 본문에서 단정하지 않는다.
    SMART_NAME = {
        # smartmontools drivedb.h (RELEASE_7_5) 의 DEFAULT 항목 기준.
        # 이 드라이브는 "TOSHIBA MG07ACA1[24]T[AE]Y?" 항목에 매칭되며 속성
        # 재정의가 없어 표준 정의가 그대로 적용된다. 밑줄만 공백으로 바꾸고
        # 표기는 원문을 유지한다 — 임의로 다듬으면 근거가 흐려진다.
        "3": "Spin Up Time", "4": "Start Stop Count",
        "5": "Reallocated Sector Ct", "9": "Power On Hours",
        "12": "Power Cycle Count", "191": "G-Sense Error Rate",
        "192": "Power-Off Retract Count", "193": "Load Cycle Count",
        "194": "Temperature Celsius", "196": "Reallocated Event Count",
        "197": "Current Pending Sector", "198": "Offline Uncorrectable",
        "199": "UDMA CRC Error Count", "220": "Disk Shift",
        "222": "Loaded Hours", "226": "Load-in Time",
    }

    M = np.abs(ig).mean(axis=0)                    # (T, F)
    hcols = list(clean)
    if drop_mask and "_mask" in hcols:
        k = hcols.index("_mask")
        M = np.delete(M, k, axis=1)
        hcols = [c for i, c in enumerate(hcols) if i != k]
    pick = np.argsort(-M.sum(axis=0))[:12]         # 중요한 것이 위로
    H = M[:, pick].T
    T_ = H.shape[1]

    fig, hx = plt.subplots(figsize=(8.6, 3.9), dpi=200)
    im = hx.imshow(H, aspect="auto", cmap="viridis", norm=PowerNorm(gamma=0.5),
                   extent=(-0.5, T_ - 0.5, len(pick) - 0.5, -0.5))
    # 시간은 왼쪽에서 오른쪽으로 흐른다. 창 첫날이 왼쪽, 예측일이 오른쪽.
    hx.set_xticks(range(T_)); hx.set_xticklabels(range(T_ - 1, -1, -1), fontsize=8)
    hx.set_yticks(range(len(pick)))
    # 이름을 앞, 번호를 괄호로 뒤에. 축이 우측 정렬이라 번호가 뒤에 있어야
    # 축 옆에서 세로로 가지런히 맞는다.
    hx.set_yticklabels(
        [f"{SMART_NAME[c]} ({c})" if c in SMART_NAME else c
         for c in (hcols[i] for i in pick)], fontsize=8.5)
    hx.set_xlabel("Days before prediction (0 = prediction day)")
    hx.set_title("Integrated Gradients: Feature × Time", fontsize=11)
    hx.tick_params(axis="x", length=3, width=0.8, direction="out")
    hx.tick_params(axis="y", length=0)
    cb = fig.colorbar(im, ax=hx, pad=0.015, fraction=0.03)
    cb.set_label("mean |IG|  (sqrt scale)", fontsize=9)
    # 눈금값은 소수 둘째 자리까지만. 자릿수가 길면 색 막대가 폭을 잡아먹는다.
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / f"ig_heatmap{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    for name in ("ig_summary.png", "ig_time_importance.png", "ig_heatmap.png",
                 "ig_importance.csv", "ig_time_importance.csv"):
        print(f"[저장] {ROOT / 'results' / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
