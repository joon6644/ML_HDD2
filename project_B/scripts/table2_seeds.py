"""표 2 다중 시드 — 탐색 전 GRU 를 시드만 바꿔 다시 학습하고 검증 구간에서 채점한다.

    python scripts/table2_seeds.py --hours 8.5

표 2(검증 구간판)의 길이 간 차이가 검증 고장 디스크 1~4대 수준이라, 시드 하나로
순위를 말하기 어렵다. 탐색 없는 고정 설정(gru_pauc)이라 시드만 바꿔 재학습하면 된다.

보고 규칙 (결과를 보기 전에 정한다)
  - 시드는 SEEDS 순서로, 시드 하나에 다섯 길이를 다 돌린 뒤 다음 시드로 간다.
    시간이 끝나도 길이 간 표본 수가 같게 남는다.
  - 마감 전에 다섯 길이가 모두 끝난 시드만 평균에 넣는다.
  - 기존 시드 42(toslb_{L}_pauc) 결과를 포함해 평균을 낸다.

산출물은 t2seed_ 로 분리한다. 실험 config 는 configs/experiments/t2seed/ 아래에
시드별로 만들고, 기존 toslb_{L}_pauc 는 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
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
MODEL = "gru_pauc"
LOOKBACKS = [7, 14, 21, 30, 60]
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
FAR_TARGETS = [0.001, 0.005, 0.01, 0.05]
SET_MINUTES = 80                  # 다섯 길이 한 세트의 예상 소요 (여유 포함)
CFG_DIR = ROOT / "configs" / "experiments" / "t2seed"
OUT_LONG = ROOT / "results" / "table2_seeds_long.csv"
OUT_TABLE = ROOT / "results" / "table2_seeds.csv"


def make_config(lookback: int, seed: int) -> Path:
    src = ROOT / "configs" / "experiments" / f"toslb_{lookback}_pauc.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["experiment"] = f"t2seed_lb{lookback}"
    cfg["seeds"] = [seed]
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    out = CFG_DIR / f"t2seed_lb{lookback}_s{seed}.yaml"
    out.write_text(
        f"# 표 2 다중 시드. toslb_{lookback}_pauc 와 시드만 다르다.\n"
        + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    return out


def score_val(experiment: str, cfg_path: Path, seed: int) -> dict:
    cfg = cfg_mod.load_yaml(cfg_path)
    pipeline = Pipeline.from_experiment(cfg)
    prepared = prepare_drive(DRIVE, pipeline)
    fold = sorted(prepared.folds, key=lambda f: f.fold)[0]
    horizon = int(pipeline.labeling["horizon_days"])
    run_dir = paths.run_dir(experiment, DRIVE, MODEL, seed, 0)
    mcfg = next(m for m in (registry.load_model_config(x) for x in cfg["models"])
                if m["name"] == MODEL)
    model = registry.resolve_class(mcfg["class"]).load(run_dir / "model")
    rec = json.loads((run_dir / "fold_metrics.json").read_text(encoding="utf-8"))
    scaler = (fold_mod.Scaler.from_state_dict(rec["scaler"])
              if rec.get("scaler") else None)
    per = {t: [] for t in FAR_TARGETS}
    paucs = []
    for start, end in month_windows(*fold.window("val")):
        part = rescale(load_part(prepared, pipeline, mcfg["family"], start, end,
                                 horizon, 8), scaler)
        rank, has_window = disk_rank(model, part, "in_horizon")
        score, failed = rank.to_numpy(), has_window.to_numpy().astype(bool)
        paucs.append(roc_auc_score(failed, score, max_fpr=0.05))
        healthy = score[~failed]
        for t in FAR_TARGETS:
            thr = float(np.quantile(healthy, 1.0 - t))
            per[t].append(float((score[failed] >= thr).mean()))
    row = {f"recall@{t:g}": float(np.mean(per[t])) for t in FAR_TARGETS}
    row["pAUC"] = float(np.mean(paucs))
    return row


def summarize(long: pd.DataFrame) -> None:
    long = long.drop_duplicates(["seed", "lookback_days"], keep="last")
    counts = long.groupby("seed")["lookback_days"].nunique()
    full = counts[counts == len(LOOKBACKS)].index.tolist()
    sub = long[long["seed"].isin(full)]
    cols = [f"recall@{t:g}" for t in FAR_TARGETS] + ["pAUC"]
    mean = sub.groupby("lookback_days")[cols].mean().round(3)
    sd = sub.groupby("lookback_days")[cols].std().round(3)
    table = mean.copy()
    for c in cols:
        table[c + "_sd"] = sd[c]
    table["n_seeds"] = len(full)
    table.to_csv(OUT_TABLE, encoding="utf-8-sig")
    print(f"\n[평균표] 완료 시드 {sorted(full)} (n={len(full)})", flush=True)
    print(mean.to_string(), flush=True)
    print("\n[표준편차]", flush=True)
    print(sd.to_string(), flush=True)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.5)
    args = ap.parse_args()
    t0 = time.time()
    deadline = t0 + args.hours * 3600

    long = (pd.read_csv(OUT_LONG) if OUT_LONG.exists() else pd.DataFrame())
    if long.empty or 42 not in set(long.get("seed", [])):
        rows = []
        for lb in LOOKBACKS:
            print(f"[seed 42] lookback {lb} 채점 (기존 run)", flush=True)
            r = score_val(f"toslb_{lb}_pauc",
                          ROOT / "configs" / "experiments" / f"toslb_{lb}_pauc.yaml", 42)
            rows.append({"seed": 42, "lookback_days": lb, **r})
        long = pd.concat([long, pd.DataFrame(rows)], ignore_index=True)
        long.to_csv(OUT_LONG, index=False)
    summarize(long)

    py = sys.executable
    for seed in SEEDS:
        if seed in set(long["seed"]) and \
                long[long["seed"] == seed]["lookback_days"].nunique() == len(LOOKBACKS):
            continue
        if time.time() + SET_MINUTES * 60 > deadline:
            print(f"\n[중단] 남은 시간으로 seed {seed} 세트를 끝낼 수 없다.", flush=True)
            break
        print(f"\n===== seed {seed} 시작 ({(time.time()-t0)/3600:.2f}h 경과) =====",
              flush=True)
        for lb in LOOKBACKS:
            # 중단 후 재시작하면 이미 채점한 (시드, 길이) 가 있다. 다시 덧붙이면
            # 그 길이만 평균에 두 번 들어간다.
            if ((long["seed"] == seed) & (long["lookback_days"] == lb)).any():
                continue
            cfg_path = make_config(lb, seed)
            ts = time.time()
            # 자식 출력의 한글·기호가 cp949 파이프에서 깨져 학습은 끝났는데
            # 종료 코드 1 을 내는 일이 있었다. 자식을 UTF-8 로 고정한다.
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            res = subprocess.run([py, str(ROOT / "scripts" / "run_experiment.py"),
                                  str(cfg_path)], cwd=ROOT, env=env,
                                 capture_output=True, text=True, encoding="utf-8",
                                 errors="replace")
            # 성공은 종료 코드가 아니라 산출물로 판정한다. 윈도우에서 학습을 마친
            # 자식이 종료 단계에서 코드 1 을 내는 일이 있었다 — 결과는 이미 다
            # 저장된 뒤였다. run_experiment 는 학습이 끝나야 이 둘을 쓴다.
            rd = paths.run_dir(f"t2seed_lb{lb}", DRIVE, MODEL, seed, 0)
            done = (rd / "fold_metrics.json").exists() and (rd / "model" / "model.pt").exists()
            if not done:
                print(res.stdout[-2000:], res.stderr[-2000:], flush=True)
                raise SystemExit(f"학습 실패: lookback {lb}, seed {seed}")
            if res.returncode != 0:
                print(f"    (종료 코드 {res.returncode} — 산출물은 완전하여 진행)", flush=True)
            r = score_val(f"t2seed_lb{lb}", cfg_path, seed)
            long = pd.concat([long, pd.DataFrame(
                [{"seed": seed, "lookback_days": lb, **r}])], ignore_index=True)
            long.to_csv(OUT_LONG, index=False)
            print(f"  lookback {lb:2d}  pAUC={r['pAUC']:.4f}  "
                  f"R@1%={r['recall@0.01']:.3f}  ({(time.time()-ts)/60:.1f}분)",
                  flush=True)
        summarize(long)

    print(f"\n[종료] 총 {(time.time()-t0)/3600:.2f}시간", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
