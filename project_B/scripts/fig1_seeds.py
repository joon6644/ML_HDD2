"""그림 1(30일판) 시드 비교 — 시드마다 탐색 전·후 곡선을 그려 모양을 비교한다.

    python scripts/fig1_seeds.py            # 학습 + 채점 + 그림
    python scripts/fig1_seeds.py --replot   # 저장된 npz 로 그림만

탐색 전: gru_pauc, lookback 30 (시드 42 는 toslb_30_pauc, 나머지는 t2seed_lb30)
탐색 후: gru_tuned_lb30 을 시드만 바꿔 재학습 (시드 42 는 tos30_gru, 나머지는 f1seed_tos30_gru)
같은 시드끼리 짝짓는다. 탐색 설정은 시드 42 로 고른 것 그대로다.

산출물은 f1seed_ 로 분리한다.
  configs/experiments/f1seed/          시드별 실험 config
  results/f1seed_curves.npz            시드별 test 곡선
  results/f1seed_summary.csv           시드별 val/test 요약
  results/f1seed_grid.png              시드별 그림 1 격자
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
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
from run_curve import GRID, curve  # noqa: E402

DRIVE = "TOSHIBA_20MG07ACA14TA"
SEEDS = [42, 1, 2, 3, 4]
CFG_DIR = ROOT / "configs" / "experiments" / "f1seed"
NPZ = ROOT / "results" / "f1seed_curves.npz"
CSV = ROOT / "results" / "f1seed_summary.csv"
PNG = ROOT / "results" / "f1seed_grid.png"


def before_arm(seed):
    if seed == 42:
        return "toslb_30_pauc", paths.CONFIG_DIR / "experiments" / "toslb_30_pauc.yaml", "gru_pauc"
    return "t2seed_lb30", CFG_DIR.parent / "t2seed" / f"t2seed_lb30_s{seed}.yaml", "gru_pauc"


def after_arm(seed):
    if seed == 42:
        return "tos30_gru", paths.CONFIG_DIR / "experiments" / "tos30_gru.yaml", "gru_tuned_lb30"
    return "f1seed_tos30_gru", CFG_DIR / f"f1seed_tos30_gru_s{seed}.yaml", "gru_tuned_lb30"


def train_after(seed):
    exp, cfg_path, model = after_arm(seed)
    rd = paths.run_dir(exp, DRIVE, model, seed, 0)
    if (rd / "fold_metrics.json").exists() and (rd / "model" / "model.pt").exists():
        return
    cfg = yaml.safe_load((paths.CONFIG_DIR / "experiments" / "tos30_gru.yaml")
                         .read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["experiment"] = exp
    cfg["seeds"] = [seed]
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("# 그림 1 시드 비교. tos30_gru 와 시드만 다르다.\n"
                        + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_experiment.py"),
                          str(cfg_path)], cwd=ROOT, env=env, capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    # table2_seeds 와 같다 — 종료 코드가 아니라 산출물로 성공을 판정한다.
    if not ((rd / "fold_metrics.json").exists() and (rd / "model" / "model.pt").exists()):
        print(res.stdout[-2000:], res.stderr[-2000:], flush=True)
        raise SystemExit(f"학습 실패: seed {seed}")


def scores(exp, cfg_path, model_name, seed, window):
    """window 가 'test' 면 세 달, 'val' 이면 검증 2창의 (점수, 고장) 목록."""
    cfg = cfg_mod.load_yaml(cfg_path)
    pl = Pipeline.from_experiment(cfg)
    pr = prepare_drive(DRIVE, pl)
    folds = sorted(pr.folds, key=lambda f: f.fold)
    hz = int(pl.labeling["horizon_days"])
    mc = next(m for m in (registry.load_model_config(x) for x in cfg["models"])
              if m["name"] == model_name)
    rd = paths.run_dir(exp, DRIVE, model_name, seed, 0)
    model = registry.resolve_class(mc["class"]).load(rd / "model")
    rec = json.loads((rd / "fold_metrics.json").read_text(encoding="utf-8"))
    sc = fold_mod.Scaler.from_state_dict(rec["scaler"]) if rec.get("scaler") else None
    spans = ([f.window("test") for f in folds] if window == "test"
             else list(month_windows(*folds[0].window("val"))))
    out = []
    for a, b in spans:
        part = rescale(load_part(pr, pl, mc["family"], a, b, hz, 8), sc)
        rank, hw = disk_rank(model, part, "in_horizon")
        out.append((rank.to_numpy(), hw.to_numpy().astype(bool)))
    return out


def pauc(months):
    return float(np.mean([roc_auc_score(f, s, max_fpr=0.05) for s, f in months]))


def shape(before, after):
    """보이는 범위(0.1~5%)에서 두 곡선의 관계."""
    vis = GRID * 100 >= 0.1
    b, a = before.mean(axis=0)[vis], after.mean(axis=0)[vis]
    d = a - b
    sign = np.sign(d[d != 0])
    return {
        "frac_after_above": float((d > 0).mean()),
        "crossings": int((np.diff(sign) != 0).sum()),
        "mean_gap": float(d.mean()),
        "gap@0.5%": float(np.interp(0.005, GRID, after.mean(0) - before.mean(0))),
        "gap@1%": float(np.interp(0.01, GRID, after.mean(0) - before.mean(0))),
    }


def plot(curves, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), dpi=130, sharey=True)
    for ax, seed in zip(axes.flat, SEEDS):
        ax.plot(GRID * 100, curves[f"before_s{seed}"].mean(0), color="#8a8a8a",
                lw=1.6, ls="--", label="Before tuning")
        ax.plot(GRID * 100, curves[f"after_s{seed}"].mean(0), color="#2e6fb7",
                lw=2.0, label="After tuning")
        r = summary.loc[seed]
        ax.set_title(f"seed {seed} | above {r.frac_after_above:.0%} | "
                     f"cross {int(r.crossings)} | test pAUC {r.test_pauc_after:.4f}",
                     fontsize=9)
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
    fig.savefig(PNG)
    plt.close(fig)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    if args.replot:
        z = np.load(NPZ)
        plot({k: z[k] for k in z.files}, pd.read_csv(CSV, index_col="seed"))
        return 0
    curves, rows = {}, []
    for seed in SEEDS:
        train_after(seed)
        row = {"seed": seed}
        for tag, arm in (("before", before_arm(seed)), ("after", after_arm(seed))):
            test = scores(*arm, seed, "test")
            val = scores(*arm, seed, "val")
            curves[f"{tag}_s{seed}"] = curve(test)
            row[f"val_pauc_{tag}"] = pauc(val)
            row[f"test_pauc_{tag}"] = pauc(test)
            row[f"test_r1_{tag}"] = float(np.interp(0.01, GRID, curves[f"{tag}_s{seed}"].mean(0)))
        row.update(shape(curves[f"before_s{seed}"], curves[f"after_s{seed}"]))
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
              flush=True)
        np.savez_compressed(NPZ, grid=GRID, **curves)
        summary = pd.DataFrame(rows).set_index("seed")
        summary.to_csv(CSV)
    plot(curves, summary)
    print(summary.round(4).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
