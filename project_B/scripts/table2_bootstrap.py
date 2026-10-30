"""표 2(검증 구간판) 부트스트랩 — 길이 간 차이가 표본 변동 안에 있는지 본다.

    python scripts/table2_bootstrap.py

시드 42 의 다섯 길이 모델(toslb_{L}_pauc)로 검증 2창을 채점해 디스크 점수를
저장하고, 같은 디스크 표본을 다섯 길이에 공통으로 복원추출한다(쌍체). 표 1 의
부트스트랩과 같은 방식이다.

산출물
  results/table2_val_scores.npz   길이·창별 디스크 점수와 고장 여부 (일련번호로 정렬)
  results/table2_bootstrap.txt    길이별 95% 구간, 30일 - 14일 쌍체 차이, 1위 비율
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
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
FAR_TARGETS = [0.001, 0.005, 0.01, 0.05]
B = 2000
CACHE = ROOT / "results" / "table2_val_scores.npz"
OUT = ROOT / "results" / "table2_bootstrap.txt"


def collect():
    store = {}
    for lb in LOOKBACKS:
        exp = f"toslb_{lb}_pauc"
        cfg = cfg_mod.load_yaml(paths.CONFIG_DIR / "experiments" / f"{exp}.yaml")
        pl = Pipeline.from_experiment(cfg)
        pr = prepare_drive(DRIVE, pl)
        fold = sorted(pr.folds, key=lambda f: f.fold)[0]
        hz = int(pl.labeling["horizon_days"])
        rd = paths.run_dir(exp, DRIVE, MODEL, 42, 0)
        mc = next(m for m in (registry.load_model_config(x) for x in cfg["models"])
                  if m["name"] == MODEL)
        model = registry.resolve_class(mc["class"]).load(rd / "model")
        rec = json.loads((rd / "fold_metrics.json").read_text(encoding="utf-8"))
        sc = fold_mod.Scaler.from_state_dict(rec["scaler"]) if rec.get("scaler") else None
        for w, (a, b) in enumerate(month_windows(*fold.window("val"))):
            part = rescale(load_part(pr, pl, mc["family"], a, b, hz, 8), sc)
            rank, hw = disk_rank(model, part, "in_horizon")
            store[(lb, w)] = (rank.sort_index(), hw.sort_index())
            print(f"  lookback {lb:2d} 창{w}  디스크 {len(rank):,}  고장 {int(hw.sum())}",
                  flush=True)
    return store


def main() -> int:
    store = collect()
    wins = sorted({w for _, w in store})
    arrays, fails = {}, []
    for w in wins:
        common = None
        for lb in LOOKBACKS:
            idx = store[(lb, w)][0].index
            common = idx if common is None else common.intersection(idx)
        f0 = store[(LOOKBACKS[0], w)][1].reindex(common).to_numpy().astype(bool)
        for lb in LOOKBACKS:
            r, hw = store[(lb, w)]
            assert np.array_equal(hw.reindex(common).to_numpy().astype(bool), f0), \
                "길이 간 고장 판정이 다르다"
            arrays[(lb, w)] = r.reindex(common).to_numpy()
        fails.append(f0)
        print(f"  창{w}: 공통 디스크 {len(common):,} / 고장 {int(f0.sum())}", flush=True)
    np.savez(CACHE, **{f"lb{lb}_w{w}": arrays[(lb, w)] for lb, w in arrays},
             **{f"fail_w{w}": fails[w] for w in wins})

    def metrics(lb, idx):
        out = {t: [] for t in FAR_TARGETS}
        out["pAUC"] = []
        for w in wins:
            s, f = arrays[(lb, w)], fails[w]
            if idx is not None:
                s, f = s[idx[w]], f[idx[w]]
            out["pAUC"].append(roc_auc_score(f, s, max_fpr=0.05))
            h = s[~f]
            for t in FAR_TARGETS:
                thr = np.quantile(h, 1 - t)
                out[t].append((s[f] >= thr).mean())
        return {k: float(np.mean(v)) for k, v in out.items()}

    keys = FAR_TARGETS + ["pAUC"]
    point = {lb: metrics(lb, None) for lb in LOOKBACKS}
    rng = np.random.default_rng(42)
    boot = {lb: {k: np.empty(B) for k in keys} for lb in LOOKBACKS}
    for b in range(B):
        idx = [rng.integers(0, len(fails[w]), len(fails[w])) for w in wins]
        for lb in LOOKBACKS:
            m = metrics(lb, idx)
            for k in keys:
                boot[lb][k][b] = m[k]

    name = {0.001: "R@0.1%", 0.005: "R@0.5%", 0.01: "R@1%", 0.05: "R@5%", "pAUC": "pAUC"}
    lines = [f"부트스트랩 {B}회, 검증 2창, 같은 디스크 표본을 다섯 길이에 공통 적용", ""]
    lines.append("## 길이별 점추정 [95% 구간]")
    for k in keys:
        lines.append(f"\n{name[k]}")
        for lb in LOOKBACKS:
            lo, hi = np.percentile(boot[lb][k], [2.5, 97.5])
            lines.append(f"  {lb:2d}일  {point[lb][k]:.3f}  [{lo:.3f}, {hi:.3f}]")
    lines.append("\n## 쌍체 차이 (30일 - 14일)")
    for k in keys:
        d = boot[30][k] - boot[14][k]
        lo, hi = np.percentile(d, [2.5, 97.5])
        lines.append(f"  {name[k]:7s} {point[30][k]-point[14][k]:+.3f}  "
                     f"[{lo:+.3f}, {hi:+.3f}]  P(30>14)={(d > 0).mean():.3f}  "
                     f"{'구분됨' if lo > 0 or hi < 0 else '0 포함'}")
    lines.append("\n## 지표별 1위 비율")
    for k in keys:
        arr = np.vstack([boot[lb][k] for lb in LOOKBACKS])
        top = np.array(LOOKBACKS)[arr.argmax(axis=0)]
        lines.append(f"  {name[k]:7s} " + "  ".join(
            f"{lb}일 {np.mean(top == lb):.2f}" for lb in LOOKBACKS))
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
