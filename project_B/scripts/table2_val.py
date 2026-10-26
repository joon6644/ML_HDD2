"""표 2 를 검증 구간에서 다시 산출한다 — 심사의견 7-1.

    python scripts/table2_val.py

현재 표 2 는 테스트 구간에서 각 월의 분위수로 임곗값을 잡아 만든 것이다.
14일 선택의 근거로 삼으려면 선택에 쓸 수 있는 구간, 즉 검증 구간에서
재야 한다는 지적이다.

저장된 predictions_val.parquet 를 그대로 쓰지 않는다. 그 파일은 검증 2개월을
한 창으로 보고 검열한 결과라, 창을 둘로 쪼개면 경계에서 검열이 어긋난다.
그래서 Optuna 탐색이 본 것과 같은 경로로 — month_windows 로 창을 나누고
창마다 load_part 로 다시 적재해 — 채점한다.

창별로 지표를 내고 창 평균을 적는다 (3.4 의 집계 규칙과 같다).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hddpred import config as cfg_mod  # noqa: E402
from hddpred import paths  # noqa: E402
from hddpred.experiments.runner import Pipeline, prepare_drive  # noqa: E402
from hddpred.features import fold as fold_mod  # noqa: E402
from hddpred.models import registry  # noqa: E402
from export_results import disk_rank, load_part, month_windows, rescale  # noqa: E402

DRIVE = "TOSHIBA_20MG07ACA14TA"
SEED = 0 if False else 42
MODEL = "gru_pauc"
LOOKBACKS = [7, 14, 21, 30, 60]
FAR_TARGETS = [0.001, 0.005, 0.01, 0.05]
PAUC_MAX_FPR = 0.05


def windows_for(lookback: int):
    cfg = cfg_mod.load_yaml(
        paths.CONFIG_DIR / "experiments" / f"toslb_{lookback}_pauc.yaml")
    pipeline = Pipeline.from_experiment(cfg)
    prepared = prepare_drive(DRIVE, pipeline)
    fold = sorted(prepared.folds, key=lambda f: f.fold)[0]
    horizon = int(pipeline.labeling["horizon_days"])
    run_dir = paths.run_dir(f"toslb_{lookback}_pauc", DRIVE, MODEL, SEED, 0)
    mcfg = next(m for m in (registry.load_model_config(x) for x in cfg["models"])
                if m["name"] == MODEL)
    model = registry.resolve_class(mcfg["class"]).load(run_dir / "model")
    record = json.loads((run_dir / "fold_metrics.json").read_text(encoding="utf-8"))
    scaler = (fold_mod.Scaler.from_state_dict(record["scaler"])
              if record.get("scaler") else None)

    out = []
    for start, end in month_windows(*fold.window("val")):
        part = load_part(prepared, pipeline, mcfg["family"], start, end,
                         horizon, 8)
        if mcfg["family"] == "sequence":
            part = rescale(part, scaler)
        rank, has_window = disk_rank(model, part, "in_horizon")
        out.append((rank.to_numpy(), has_window.to_numpy().astype(bool),
                    f"{start}~{end}"))
    return out


def main() -> int:
    rows = []
    for lookback in LOOKBACKS:
        print(f"\n===== lookback {lookback}일 =====", flush=True)
        wins = windows_for(lookback)
        per = {t: [] for t in FAR_TARGETS}
        paucs = []
        for score, failed, label in wins:
            healthy = score[~failed]
            paucs.append(roc_auc_score(failed, score, max_fpr=PAUC_MAX_FPR))
            for t in FAR_TARGETS:
                thr = float(np.quantile(healthy, 1.0 - t))
                per[t].append(float((score[failed] >= thr).mean()))
            print(f"  {label}  디스크 {len(score):,} / 고장 {int(failed.sum())}",
                  flush=True)
        row = {"lookback_days": lookback, "n_windows": len(wins)}
        for t in FAR_TARGETS:
            row[f"recall@{t:g}"] = round(float(np.mean(per[t])), 3)
        row["pAUC"] = round(float(np.mean(paucs)), 4)
        rows.append(row)
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    frame = pd.DataFrame(rows)
    out = ROOT / "results" / "table2_val.csv"
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n" + frame.to_string(index=False), flush=True)
    print(f"\n[저장] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
