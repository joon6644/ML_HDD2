"""입력 윈도우 길이별 학습·추론 시간 — 통제 측정.

    python scripts/measure_lookback_time.py

기존 run 의 fold_metrics.json 에 남은 fit_seconds 는 못 쓴다. 다섯 길이가
서로 다른 시점에 따로 돌아서 GPU 경합이 섞였고, 조기종료가 걸린 에포크 수가
길이마다 달라(7~13) 총 시간이 길이와 무관하게 흔들린다. 실측: 7일 1078초(9
에포크) > 30일 492초(7 에포크).

그래서 여기서는 세 가지를 통제한다.
  - 한 세션에서 다섯 길이를 순차로 돈다 (같은 GPU 상태)
  - 조기종료를 끄고 에포크 수를 EPOCHS 로 고정한다 (에포크당 시간을 본다)
  - 데이터 적재는 타이머 밖으로 빼서 따로 기록한다

runs/ 에 아무것도 쓰지 않는다. 결과는 results/lookback_time.csv 한 장이다.
"""

from __future__ import annotations

import copy
import gc
import sys
import time
from pathlib import Path

import pandas as pd

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

DRIVE = "TOSHIBA_20MG07ACA14TA"
SEED = 42
EPOCHS = 3          # 조기종료를 끄고 이만큼만 돈다
LOOKBACKS = [7, 14, 21, 30, 60]


def measure(lookback: int) -> dict:
    cfg = cfg_mod.load_yaml(
        paths.CONFIG_DIR / "experiments" / f"toslb_{lookback}_pauc.yaml")
    pipeline = Pipeline.from_experiment(cfg)
    prepared = prepare_drive(DRIVE, pipeline)
    fold = sorted(prepared.folds, key=lambda f: f.fold)[0]
    threads = int(pipeline.preprocessing.get("duckdb", {}).get("threads", 8))

    t0 = time.time()
    parts = {
        name: _load_part(prepared, fold, name, "sequence",
                         pipeline.features, pipeline.labeling, threads)
        for name in ("train", "val", "test")
    }
    load_s = time.time() - t0

    scaling = pipeline.features.get("scaling", {})
    scaler = fold_mod.Scaler(
        scaling.get("method", "standard"), scaling.get("clip_quantile")
    ).fit(parts["train"].matrix)
    for part in parts.values():
        part.matrix = scaler.transform(part.matrix, fill_nan=True)

    mcfg = copy.deepcopy(registry.load_model_config(cfg["models"][0]))
    # 조기종료를 끄지 않으면 에포크 수가 길이마다 달라져 비교가 깨진다.
    mcfg["training"]["epochs"] = EPOCHS
    mcfg["training"]["early_stopping_patience"] = 10 ** 6
    model = registry.create(mcfg, SEED)

    t1 = time.time()
    model.fit(parts["train"], parts["val"])
    fit_s = time.time() - t1

    t2 = time.time()
    model.predict_proba(parts["test"])
    infer_s = time.time() - t2

    row = {
        "lookback_days": lookback,
        "n_train": len(parts["train"]),
        "n_test": len(parts["test"]),
        "load_s": round(load_s, 1),
        "fit_total_s": round(fit_s, 1),
        "epochs": EPOCHS,
        "fit_per_epoch_s": round(fit_s / EPOCHS, 1),
        "infer_s": round(infer_s, 2),
    }
    del parts, model, scaler, prepared
    gc.collect()
    return row


def main() -> int:
    rows = []
    for lookback in LOOKBACKS:
        print(f"\n===== lookback {lookback}일 =====", flush=True)
        row = measure(lookback)
        rows.append(row)
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    frame = pd.DataFrame(rows)
    out = ROOT / "results" / "lookback_time.csv"
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n" + frame.to_string(index=False), flush=True)
    print(f"\n[저장] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
