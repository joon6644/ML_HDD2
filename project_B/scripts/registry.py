"""실험 결과 등록부 — 돌린 실험을 한 곳에서 찾아본다.

    python scripts/registry.py add <experiment> [<experiment> ...]
    python scripts/registry.py add --all          # results/*_long.csv 전부
    python scripts/registry.py show               # 표로 출력
    python scripts/registry.py show --sort r01    # 특정 지표로 정렬

━━ 왜 필요한가 ━━

results/ 에 실험마다 <이름>_long.csv 가 쌓이는데, 파일만 봐서는 그 실험이 어떤
분할·학습창·표집·모델·피처였는지 알 수 없다. 몇 주 뒤에 "그때 그 숫자 어디서
나왔지"를 되짚으려면 설정 파일과 결과 파일을 대조해야 한다.

이 스크립트는 실험 하나당 한 줄로 그걸 합쳐 둔다. 설정(어떻게 돌렸나)과
지표(무엇이 나왔나)가 같은 줄에 있다.

  results/registry.csv   기계용. 열이 고정돼 있어 다시 읽어 쓰기 좋다.
  results/REGISTRY.md    사람용. 최근 것이 위로 온다.

━━ 지표 정의 ━━

전부 디스크 단위, in_horizon 판정, val 에서 잡은 임곗값을 test 에 그대로 적용한
값이다 (threshold_source=val_quantile). test 여러 달은 하나의 모집단으로
통합(pooling)해서 집계한다 — 달마다 고장 수가 달라서 균등가중 평균을 쓰면
표본이 적은 달이 과대 대표된다.

  r001/r005/r01/r05  목표 FAR 0.1/0.5/1/5% 에서의 Recall
  far01              목표 1% 일 때 test 에서 실제로 나온 FAR
  roc_auc            달별 ROC-AUC 의 평균
  far_drift          경과 개월과 실제 FAR 의 상관. 재학습 없이 예측을 이어갈 때
                     임곗값이 표류하는 정도다. + 면 갈수록 오탐이 늘어난다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CSV = RESULTS / "registry.csv"
MD = RESULTS / "REGISTRY.md"

COLUMNS = [
    "experiment", "drive", "model", "features",
    "train_window", "train_months", "val_months", "n_test_months",
    "sampling", "seeds", "n_train_rows", "n_test_failed",
    "r001", "r005", "r01", "r05", "far01", "roc_auc", "far_drift",
    "recorded_at", "config",
]


def _describe_features(ov: dict) -> str:
    """features.base.* 오버라이드를 한 낱말로 요약한다."""
    bits = []
    for key, label in (("asfd_windows", "ASFD"), ("cid_windows", "CID"),
                       ("rolling_windows", "roll"), ("diff_lags", "diff")):
        v = ov.get(f"features.base.{key}")
        if v:
            bits.append(f"{label}{'/'.join(str(x) for x in v)}")
    if ov.get("features.base.include_since_start"):
        bits.append("since_start")
    if ov.get("features.base.include_age"):
        bits.append("age")
    return "+".join(bits) if bits else "raw"


def _describe_sampling(ov: dict) -> str:
    strategy = ov.get("features.train_sampling.strategy", "none")
    if strategy == "stride":
        return f"stride{ov.get('features.train_sampling.negative_stride', 7)}"
    return strategy


def collect(experiment: str) -> dict | None:
    long_path = RESULTS / f"{experiment}_long.csv"
    cfg_path = ROOT / "configs" / "experiments" / f"{experiment}.yaml"
    if not long_path.exists():
        print(f"  [건너뜀] 결과 없음: {long_path.name}")
        return None
    if not cfg_path.exists():
        print(f"  [건너뜀] 설정 없음: {cfg_path.name}")
        return None

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ov = cfg.get("overrides", {})
    d = pd.read_csv(long_path, encoding="utf-8-sig")

    row = {
        "experiment": experiment,
        "drive": ",".join(cfg.get("drives", [])),
        "model": ",".join(Path(m).stem for m in cfg.get("models", [])),
        "features": _describe_features(ov),
        "train_window": ov.get("split.train_window", ""),
        "train_months": ov.get("split.train_months", "") if
                        ov.get("split.train_window") == "rolling" else "",
        "val_months": ov.get("split.val_months", ""),
        "n_test_months": d.test_month.nunique(),
        "sampling": _describe_sampling(ov),
        "seeds": ",".join(str(s) for s in cfg.get("seeds", [])),
        "config": str(cfg_path.relative_to(ROOT)).replace("\\", "/"),
        "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # 학습 행 수는 fold_metrics 에서 읽는다 (설정에는 없다).
    seed = cfg.get("seeds", [42])[0]
    drive = cfg.get("drives", [""])[0]
    model = Path(cfg.get("models", ["x"])[0]).stem
    fm = ROOT / "runs" / experiment / drive / model / f"seed{seed}" / "fold00" / "fold_metrics.json"
    if fm.exists():
        j = json.loads(fm.read_text(encoding="utf-8"))
        row["n_train_rows"] = (j.get("sampling") or {}).get("n_train") or ""
    else:
        row["n_train_rows"] = ""

    # test 여러 달을 하나의 모집단으로 통합해 집계한다.
    for target, key in ((0.001, "r001"), (0.005, "r005"), (0.01, "r01"), (0.05, "r05")):
        s = d[d.far_target == target]
        row[key] = round(s.tp.sum() / s.n_failed.sum(), 4) if len(s) else ""

    s1 = d[d.far_target == 0.01].sort_values("test_month")
    row["n_test_failed"] = int(s1.n_failed.sum()) if len(s1) else ""
    row["far01"] = round(s1.fp.sum() / s1.n_healthy.sum(), 5) if len(s1) else ""
    row["roc_auc"] = round(d.groupby("test_month").roc_auc.first().mean(), 4)
    # 임곗값 표류: 경과 개월과 실제 FAR 의 상관.
    if len(s1) >= 3:
        idx = np.arange(len(s1))
        row["far_drift"] = round(float(np.corrcoef(idx, s1.far.to_numpy())[0, 1]), 3)
    else:
        row["far_drift"] = ""
    return row


def write(rows: list[dict]) -> None:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if CSV.exists():
        old = pd.read_csv(CSV, encoding="utf-8-sig")
        # 같은 실험은 최신 것으로 갈아끼운다.
        old = old[~old.experiment.isin(frame.experiment)]
        frame = pd.concat([old, frame], ignore_index=True)
    frame = frame.sort_values("recorded_at", ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CSV, index=False, encoding="utf-8-sig")

    head = (
        "# 실험 등록부\n\n"
        "`scripts/registry.py` 가 생성한다. 직접 고치지 마라.\n\n"
        "모든 지표는 디스크 단위 · in_horizon 판정 · val 에서 잡은 임곗값을\n"
        "test 에 적용한 값이며, test 여러 달은 하나의 모집단으로 통합해 집계했다.\n"
        "`far_drift` 는 경과 개월과 실제 FAR 의 상관으로, + 면 재학습 없이\n"
        "시간이 갈수록 오탐이 늘어난다는 뜻이다.\n\n"
    )
    show = ["experiment", "drive", "features", "train_months", "sampling",
            "n_train_rows", "n_test_failed", "r001", "r005", "r01", "r05",
            "roc_auc", "far01", "far_drift"]
    # tabulate 의존성을 늘리지 않으려고 직접 만든다.
    view = frame[show].fillna("")
    lines = ["| " + " | ".join(show) + " |",
             "|" + "|".join("---" for _ in show) + "|"]
    for _, r in view.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    MD.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[저장] {CSV}  ({len(frame)}개 실험)")
    print(f"[저장] {MD}")


def main() -> int:
    ap = argparse.ArgumentParser(description="실험 결과 등록부")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="실험을 등록부에 넣는다")
    a.add_argument("experiments", nargs="*")
    a.add_argument("--all", action="store_true", help="results/*_long.csv 전부")
    s = sub.add_parser("show", help="등록부를 표로 출력")
    s.add_argument("--sort", default=None)
    args = ap.parse_args()

    if args.cmd == "show":
        if not CSV.exists():
            print("등록부가 아직 없다.")
            return 1
        frame = pd.read_csv(CSV, encoding="utf-8-sig")
        if args.sort:
            frame = frame.sort_values(args.sort, ascending=False)
        show = ["experiment", "features", "train_months", "sampling",
                "n_test_failed", "r001", "r01", "r05", "roc_auc", "far_drift"]
        print(frame[show].to_string(index=False))
        return 0

    names = list(args.experiments)
    if args.all:
        names = sorted({p.name[: -len("_long.csv")]
                        for p in RESULTS.glob("*_long.csv")})
    if not names:
        print("등록할 실험을 지정하라 (또는 --all).")
        return 1

    rows = []
    for name in names:
        print(f"[수집] {name}")
        row = collect(name)
        if row:
            rows.append(row)
    if not rows:
        print("등록할 것이 없다.")
        return 1
    write(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
