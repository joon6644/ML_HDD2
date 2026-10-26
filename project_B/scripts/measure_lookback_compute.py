"""입력 윈도우 길이별 순전파·역전파 연산 시간 — 고정 비용을 분리한 측정.

    python scripts/measure_lookback_compute.py

measure_lookback_time.py 는 에포크 전체를 재서 73~81초(폭 11%)가 나왔다.
시점 수가 8.6배 차이나는데 시간이 거의 같았다. 원인은 에포크당 비용이
윈도우 길이와 무관한 두 항목에 지배되기 때문이다.

  1. 배치마다 loss.item() 으로 GPU 를 동기화한다 (sequence.py 의 학습 루프).
     배치 512 로 1,102만 표본이면 에포크당 21,529 번이고, 한 번의 비용은
     시점 수와 무관하다.
  2. 에포크 끝마다 검증 구간 전체를 채점한다.

그래서 여기서는 셋을 걷어낸다.
  - 배치를 미리 GPU 에 올려두고 재므로 데이터 적재가 타이머에 없다
  - 루프 안에서 동기화하지 않는다 (앞뒤로 한 번씩만)
  - 검증 채점을 하지 않는다

남는 것은 GRU 순환 연산이고, 그것은 시점 수에 비례해야 한다. 배치 크기를
두 가지로 재서 커널 실행 오버헤드의 영향도 같이 본다 (배치가 크면 실행
횟수가 줄어 연산이 더 드러난다).

주의: 여기서 재는 값은 벽시계 학습 시간이 아니다. "모델의 계산량"을 나타내는
값이고, 적재·동기화·검증 채점 같은 길이 무관 비용은 빠져 있다.
"""

from __future__ import annotations

import copy
import gc
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hddpred import config as cfg_mod  # noqa: E402
from hddpred import paths  # noqa: E402
from hddpred.experiments.runner import (  # noqa: E402
    Pipeline,
    _load_part,
    prepare_drive,
)
from hddpred.features import fold as fold_mod  # noqa: E402
from hddpred.models import registry  # noqa: E402
from hddpred.models.sequence import build_criterion  # noqa: E402

DRIVE = "TOSHIBA_20MG07ACA14TA"
SEED = 42
LOOKBACKS = [7, 14, 21, 30, 60]
# (배치 크기, 미리 올려둘 배치 수, 반복 라운드)
CASES = [(512, 20, 10), (4096, 8, 10)]
WARMUP = 15


def time_case(model, net, batches, criterion, optimizer, use_amp, clip, rounds):
    """순전파+역전파. 루프 안에서 동기화하지 않는다."""
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def step(bx, by):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = criterion(net(bx), by)
        scaler.scale(loss).backward()
        if clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(net.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

    net.train()
    for i in range(WARMUP):                 # cuDNN 자동조정·할당기 예열
        bx, by = batches[i % len(batches)]
        step(bx, by)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(rounds):
        for bx, by in batches:
            step(bx, by)
    torch.cuda.synchronize()
    train_s = time.time() - t0

    net.eval()
    with torch.no_grad():
        for i in range(WARMUP):
            bx, _ = batches[i % len(batches)]
            with torch.amp.autocast("cuda", enabled=use_amp):
                net(bx)
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(rounds):
            for bx, _ in batches:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    net(bx)
        torch.cuda.synchronize()
        fwd_s = time.time() - t1

    n_iter = rounds * len(batches)
    return train_s / n_iter * 1000, fwd_s / n_iter * 1000


def measure(lookback: int) -> list[dict]:
    cfg = cfg_mod.load_yaml(
        paths.CONFIG_DIR / "experiments" / f"toslb_{lookback}_pauc.yaml")
    pipeline = Pipeline.from_experiment(cfg)
    prepared = prepare_drive(DRIVE, pipeline)
    fold = sorted(prepared.folds, key=lambda f: f.fold)[0]
    threads = int(pipeline.preprocessing.get("duckdb", {}).get("threads", 8))
    train = _load_part(prepared, fold, "train", "sequence",
                       pipeline.features, pipeline.labeling, threads)
    scaling = pipeline.features.get("scaling", {})
    scaler = fold_mod.Scaler(
        scaling.get("method", "standard"), scaling.get("clip_quantile")
    ).fit(train.matrix)
    train.matrix = scaler.transform(train.matrix, fill_nan=True)

    mcfg = copy.deepcopy(registry.load_model_config(cfg["models"][0]))
    rows = []
    for size, n_batch, rounds in CASES:
        model = registry.create(mcfg, SEED)
        net = model._build(train.n_features)
        loader = model._loader(train, shuffle=True, batch_size=size)
        batches = []
        for bx, by in loader:                # 미리 GPU 로 올려 적재를 타이머에서 뺀다
            batches.append((bx.to(model.device, non_blocking=True),
                            by.to(model.device, non_blocking=True)))
            if len(batches) >= n_batch:
                break
        criterion, _ = build_criterion(model.training, train.y, model.device)
        criterion = (criterion.to(model.device)
                     if hasattr(criterion, "to") else criterion)
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=float(model.training.get("learning_rate", 1e-3)),
            weight_decay=float(model.training.get("weight_decay", 0.0)),
        )
        use_amp = bool(model.training.get("amp", True))
        clip = float(model.training.get("grad_clip", 0.0))
        train_ms, fwd_ms = time_case(model, net, batches, criterion,
                                     optimizer, use_amp, clip, rounds)
        n_iter_epoch = -(-len(train) // size)
        rows.append({
            "lookback_days": lookback,
            "batch_size": size,
            "train_ms_per_batch": round(train_ms, 3),
            "fwd_ms_per_batch": round(fwd_ms, 3),
            "batches_per_epoch": n_iter_epoch,
            "train_s_per_epoch_compute": round(train_ms * n_iter_epoch / 1000, 1),
        })
        print("  " + "  ".join(f"{k}={v}" for k, v in rows[-1].items()), flush=True)
        del net, batches, optimizer, criterion, model, loader
        gc.collect()
        torch.cuda.empty_cache()
    del train, prepared, scaler
    gc.collect()
    return rows


def main() -> int:
    rows = []
    for lookback in LOOKBACKS:
        print(f"\n===== lookback {lookback}일 =====", flush=True)
        rows.extend(measure(lookback))
    frame = pd.DataFrame(rows)
    out = ROOT / "results" / "lookback_compute.csv"
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    print("", flush=True)
    for size, *_ in CASES:
        sub = frame[frame["batch_size"] == size]
        base = sub.iloc[0]["train_ms_per_batch"]
        print(f"\n[배치 {size}] 7일 대비 배수 (시점 수 배수와 비교)", flush=True)
        for _, r in sub.iterrows():
            print("  %2d일  fwd+bwd %7.3f ms  (%.2f배)  시점 %.2f배"
                  % (r["lookback_days"], r["train_ms_per_batch"],
                     r["train_ms_per_batch"] / base, r["lookback_days"] / 7),
                  flush=True)
    print(f"\n[저장] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
