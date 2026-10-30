"""입력 윈도우 전 구간 탐색 — 길이마다 같은 절차로 탐색하고 시드별로 그림 1 을 잰다.

    python scripts/window_sweep.py              # 남은 길이 전부
    python scripts/window_sweep.py --only 21    # 한 길이만

길이마다 30일판(tos30_gru)과 같은 절차다.
  탐색   run_optuna_rnn --base-experiment toslb_{L}_pauc --val-aggregate pool
         --trials 50, 배치 2048 고정, 시드 42
  학습   찾은 설정을 시드 42, 1~4 로 재학습
  비교   같은 시드의 탐색 전 모델(toslb_{L}_pauc / t2seed_lb{L})과 짝짓는다

14일도 다시 탐색한다. 논문의 14일 설정(gru_tuned_pool)은 배치를 탐색 축에
두던 때 찾은 것이라 이 비교와 절차가 다르다.
30일은 이미 끝났다(tos30_gru + f1seed). 요약에 그대로 합친다.

산출물은 wsweep_ 로 분리한다.
  configs/models/gru_tuned_wsweep_lb{L}.yaml       탐색 결과
  configs/experiments/wsweep/                      시드별 실험 config
  results/optuna_gru_wsweep_lb{L}_trials.csv
  results/wsweep_curves.npz, wsweep_summary.csv, wsweep_lb{L}_grid.png
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hddpred import paths  # noqa: E402
from run_curve import GRID, curve  # noqa: E402
import fig1_seeds as f1  # noqa: E402

DRIVE = "TOSHIBA_20MG07ACA14TA"
ORDER = [21, 60, 14, 7]           # 유력한 길이부터
SEEDS = [42, 1, 2, 3, 4]
CFG_DIR = ROOT / "configs" / "experiments" / "wsweep"
NPZ = ROOT / "results" / "wsweep_curves.npz"
CSV = ROOT / "results" / "wsweep_summary.csv"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def tag(lb):
    return f"wsweep_lb{lb}"


def search(lb):
    out = ROOT / "configs" / "models" / f"gru_tuned_{tag(lb)}.yaml"
    if out.exists():
        return
    print(f"\n===== lookback {lb} 탐색 =====", flush=True)
    log = open(ROOT / "results" / f"optuna_gru_{tag(lb)}_run.log", "w", encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_optuna_rnn.py"),
                    "--cell", "gru", "--trials", "50",
                    "--base-experiment", f"toslb_{lb}_pauc",
                    "--tag", tag(lb), "--val-aggregate", "pool"],
                   cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    if not out.exists():
        raise SystemExit(f"탐색 실패: lookback {lb} (results/optuna_gru_{tag(lb)}_run.log)")


def after_arm(lb, seed):
    return (tag(lb), CFG_DIR / f"{tag(lb)}_s{seed}.yaml", f"gru_tuned_{tag(lb)}")


def before_arm(lb, seed):
    if seed == 42:
        return (f"toslb_{lb}_pauc",
                paths.CONFIG_DIR / "experiments" / f"toslb_{lb}_pauc.yaml", "gru_pauc")
    return (f"t2seed_lb{lb}", ROOT / "configs" / "experiments" / "t2seed"
            / f"t2seed_lb{lb}_s{seed}.yaml", "gru_pauc")


def train(lb, seed):
    exp, cfg_path, model = after_arm(lb, seed)
    rd = paths.run_dir(exp, DRIVE, model, seed, 0)
    if (rd / "fold_metrics.json").exists() and (rd / "model" / "model.pt").exists():
        return
    # 탐색 전 실험과 파이프라인이 같아야 짝이 맞는다. 모델만 바꾼다.
    cfg = yaml.safe_load((paths.CONFIG_DIR / "experiments" / f"toslb_{lb}_pauc.yaml")
                         .read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["experiment"] = exp
    cfg["models"] = [f"configs/models/{model}.yaml"]
    cfg["seeds"] = [seed]
    cfg["overrides"] = {k: v for k, v in cfg["overrides"].items()
                        if not k.startswith("models.")}
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(f"# 윈도우 탐색. toslb_{lb}_pauc 와 모델·시드만 다르다.\n"
                        + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_experiment.py"),
                          str(cfg_path)], cwd=ROOT, env=ENV, capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    if not ((rd / "fold_metrics.json").exists() and (rd / "model" / "model.pt").exists()):
        print(res.stdout[-2000:], res.stderr[-2000:], flush=True)
        raise SystemExit(f"학습 실패: lookback {lb}, seed {seed}")


def plot(lb, curves, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), dpi=130, sharey=True)
    for ax, seed in zip(axes.flat, SEEDS):
        ax.plot(GRID * 100, curves[f"lb{lb}_before_s{seed}"].mean(0), color="#8a8a8a",
                lw=1.6, ls="--", label="Before tuning")
        ax.plot(GRID * 100, curves[f"lb{lb}_after_s{seed}"].mean(0), color="#2e6fb7",
                lw=2.0, label="After tuning")
        r = summary.loc[seed]
        ax.set_title(f"lb {lb} seed {seed} | above {r.frac_after_above:.0%} | "
                     f"cross {int(r.crossings)} | test {r.test_pauc_after:.4f}", fontsize=9)
        ax.set_xscale("log")
        ax.set_xlim(0.1, 5)
        ax.xaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0.1", "0.2", "0.5", "1", "2", "5"]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(0.2, 0.9)
        ax.grid(True, alpha=0.25, lw=0.6)
    axes.flat[0].legend(loc="upper left", fontsize=8)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / f"wsweep_lb{lb}_grid.png")
    plt.close(fig)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int)
    args = ap.parse_args()
    curves = dict(np.load(NPZ)) if NPZ.exists() else {}
    rows = pd.read_csv(CSV).to_dict("records") if CSV.exists() else []
    for lb in ([args.only] if args.only else ORDER):
        search(lb)
        for seed in SEEDS:
            if any(r["lookback"] == lb and r["seed"] == seed for r in rows):
                continue
            train(lb, seed)
            row = {"lookback": lb, "seed": seed}
            for side, arm in (("before", before_arm(lb, seed)),
                              ("after", after_arm(lb, seed))):
                test = f1.scores(*arm, seed, "test")
                val = f1.scores(*arm, seed, "val")
                curves[f"lb{lb}_{side}_s{seed}"] = curve(test)
                row[f"val_pauc_{side}"] = f1.pauc(val)
                row[f"test_pauc_{side}"] = f1.pauc(test)
            row.update(f1.shape(curves[f"lb{lb}_before_s{seed}"],
                                curves[f"lb{lb}_after_s{seed}"]))
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
                  flush=True)
            np.savez_compressed(NPZ, **curves)
            pd.DataFrame(rows).to_csv(CSV, index=False)
        sub = pd.DataFrame([r for r in rows if r["lookback"] == lb]).set_index("seed")
        plot(lb, curves, sub)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
