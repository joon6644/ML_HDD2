"""학습된 모델을 다시 채점해 결과를 CSV 로 내보낸다.

    python scripts/export_results.py zoo_3month_seeds
    python scripts/export_results.py zoo_3month --out results/zoo29

재학습은 하지 않는다. runs/<experiment>/ 에 저장된 모델과 파생 레이어만 쓴다.
fold 0 에서 학습한 모델 하나로 split manifest 의 모든 test 월을 채점하므로,
"한 번 학습해서 여러 달을 예측한다"는 프로토콜이 그대로 재현된다.

임곗값을 잡는 방법이 둘이다. --threshold-source 로 고른다.

    month_quantile (기본)
        달마다 "정상 디스크 상위 f%" 로 잡는다. 운영 예산이 달 단위로
        배정되기 때문이고, 세 달을 통으로 잡으면 모델이 낡으며 점수 분포가
        밀리는 것이 FAR 불균형으로 새어 나온다. 다만 그 달의 정상 디스크
        점수를 봐야 정해지므로 test 를 참조한다.

    validation
        run 디렉터리의 threshold.json 을 그대로 쓴다. runner 가 validation
        에서만 골라 둔 값이라 test 를 전혀 참조하지 않는다. 세 달에 같은
        임곗값이 적용되므로 실제 FAR 은 달마다 목표에서 벗어난다. 그 벗어남
        자체가 "val 에서 고른 운영점이 이후 달에 얼마나 유지되는가"이고,
        long CSV 의 far 열에 그대로 남는다.

내보내는 파일 두 개:

    <out>_long.csv     (모델, 시드, test월, FAR목표) 한 줄. 원자료.
                       여기서 어떤 집계든 다시 만들 수 있다.
    <out>_summary.csv  달을 풀링하고 시드로 평균낸 요약. 논문 표에 쓰는 값.

풀링은 고장 사건을 하나의 모집단으로 본다.
    Recall = sum(TP) / sum(고장),  FAR = sum(FP) / sum(정상)
월별 recall 을 단순 평균하면 고장이 적은 달이 과대 대표된다.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hddpred import config as cfg_mod  # noqa: E402
from hddpred import paths  # noqa: E402
from hddpred.evaluation import metrics as metrics_mod  # noqa: E402
from hddpred.experiments.runner import Pipeline, prepare_drive  # noqa: E402
from hddpred.features import fold as fold_mod  # noqa: E402
from hddpred.inference import threshold as threshold_mod  # noqa: E402
from hddpred.models import registry  # noqa: E402

FAR_TARGETS = [0.005, 0.01, 0.02, 0.04]

# threshold.policy -> 그 정책이 선언한 명목 목표 오탐률의 키.
# far_target 열에 무엇을 적을지 정하는 데만 쓴다.
POLICY_TARGET_KEY = {"fixed_disk_far": "fixed_disk_far", "fixed_fpr": "fixed_fpr"}


def load_part(prepared, pipeline, family, start, end, horizon, threads):
    common = dict(horizon_days=horizon, censoring_scope="window", threads=threads)
    if family == "tabular":
        return fold_mod.load_tabular(
            prepared.features_path, prepared.labels_path, start, end, **common
        )
    sequence = pipeline.features["sequence"]
    return fold_mod.load_sequence(
        prepared.features_path, prepared.labels_path, start, end,
        lookback=int(sequence["lookback_days"]),
        short_history=sequence.get("short_history", "pad"),
        **common,
    )


def rescale(part, scaler):
    """시퀀스 계열은 runner 가 fold 의 train 구간 스케일러를 적용해 학습했다."""
    if scaler is None:
        return part
    return fold_mod.SequenceFold(
        matrix=scaler.transform(part.matrix, fill_nan=True),
        end_index=part.end_index, lookback=part.lookback, y=part.y,
        record_date=part.record_date, serial=part.serial,
        failure_date=part.failure_date, columns=part.columns,
        valid_len=part.valid_len, window=part.window,
    )


def score_month(model, part, rule, thresholds=None):
    """한 달을 채점한다. 디스크 점수는 그 달 행 점수의 최댓값(OR 집계)이다.

    thresholds: {라벨: 임곗값}. None 이면 그 달 정상 디스크 상위 f% 로 잡는다.
    """
    frame = pd.DataFrame(
        {"serial": part.serial, "y": part.y, "score": model.predict_proba(part)}
    )
    grouped = frame.groupby("serial")
    disk_score = grouped["score"].max()
    # 정답 구간(고장 H일 전) 을 가진 디스크. 라벨이 곧 구간의 정의다.
    has_window = grouped["y"].max().astype(bool)
    inside = frame[frame["y"] == 1].groupby("serial")["score"].max()
    inside = inside.reindex(disk_score.index).fillna(-np.inf)

    out = {
        "n_failed": int(has_window.sum()),
        "n_healthy": int((~has_window).sum()),
        "roc_auc": float(roc_auc_score(has_window.astype(int), disk_score.to_numpy())),
        "pr_auc": float(
            average_precision_score(has_window.astype(int), disk_score.to_numpy())
        ),
    }
    if thresholds is None:
        healthy_scores = disk_score[~has_window].to_numpy()
        thresholds = {
            target: float(np.quantile(healthy_scores, 1.0 - target))
            for target in FAR_TARGETS
        }

    out["cells"] = {}
    for label, threshold in thresholds.items():
        alarm_any = disk_score >= threshold
        alarm_in = inside >= threshold
        if rule == "in_horizon":
            detected = has_window & alarm_in
        elif rule == "or":
            detected = has_window & alarm_any
        else:
            raise ValueError(f"지원하지 않는 rule: {rule!r} (in_horizon | or)")
        out["cells"][label] = {
            "threshold": float(threshold),
            "tp": int(detected.sum()),
            "fn": int((has_window & ~detected).sum()),
            "fp": int((~has_window & alarm_any).sum()),
            "tn": int((~has_window & ~alarm_any).sum()),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="학습된 모델 재채점 -> CSV")
    parser.add_argument("experiment", help="configs/experiments/*.yaml 의 이름")
    parser.add_argument("--out", default=None, help="출력 접두사 (기본 results/<이름>)")
    parser.add_argument("--rule", default=None, help="in_horizon | or (기본은 설정값)")
    parser.add_argument(
        "--threshold-source",
        default="month_quantile",
        choices=["month_quantile", "validation"],
        help="month_quantile: 달마다 정상 디스크 상위 f%% | "
        "validation: run 의 threshold.json (val 에서 고른 값) 을 세 달에 그대로",
    )
    args = parser.parse_args()

    cfg = cfg_mod.load_yaml(paths.CONFIG_DIR / "experiments" / f"{args.experiment}.yaml")
    pipeline = Pipeline.from_experiment(cfg)
    rule = args.rule or pipeline.evaluation["disk_level"].get("rule", "in_horizon")
    from_validation = args.threshold_source == "validation"
    # far_target 열에 적을 명목 목표. validation 소스에서만 쓴다.
    policy_cfg = pipeline.evaluation["threshold"]
    target_key = POLICY_TARGET_KEY.get(policy_cfg.get("policy", "max_f1"))
    nominal_target = float(policy_cfg[target_key]) if target_key else ""
    horizon = int(pipeline.labeling["horizon_days"])
    threads = int(pipeline.preprocessing.get("duckdb", {}).get("threads", 8))
    seeds = [int(s) for s in cfg["seeds"]]
    experiment = cfg["experiment"]

    prefix = Path(args.out or (paths.PROJECT_ROOT / "results" / args.experiment))
    prefix.parent.mkdir(parents=True, exist_ok=True)

    long_rows = []
    for drive_name in cfg["drives"]:
        prepared = prepare_drive(drive_name, pipeline)
        folds = sorted(prepared.folds, key=lambda f: f.fold)
        # 채점하는 모델은 항상 fold 0 에서 학습한 그 하나다. 달마다 fold 의
        # train 창을 적으면 "2026-02 행의 모델은 2025-12 까지 봤다"는 거짓이
        # CSV 에 남는다. 학습 창은 fold 0 것으로 고정해서 적는다.
        trained = folds[0]
        cache = {}
        for model_path in cfg["models"]:
            model_cfg = registry.load_model_config(model_path)
            name, family = model_cfg["name"], model_cfg["family"]
            for seed in seeds:
                run_dir = paths.run_dir(experiment, drive_name, name, seed, 0)
                if not (run_dir / "fold_metrics.json").exists():
                    continue
                with (run_dir / "fold_metrics.json").open(encoding="utf-8") as fh:
                    record = json.load(fh)
                model = registry.resolve_class(model_cfg["class"]).load(run_dir / "model")
                scaler = (
                    fold_mod.Scaler.from_state_dict(record["scaler"])
                    if record.get("scaler") else None
                )
                thresholds = None
                if from_validation:
                    chosen = threshold_mod.load(run_dir)
                    thresholds = {nominal_target: float(chosen["threshold"])}
                for fold in folds:
                    start, end = fold.window("test")
                    key = (drive_name, family, start)
                    if key not in cache:
                        cache[key] = load_part(
                            prepared, pipeline, family, start, end, horizon, threads
                        )
                    part = cache[key]
                    if family == "sequence":
                        part = rescale(part, scaler)
                    scored = score_month(model, part, rule, thresholds)
                    for target, cell in scored["cells"].items():
                        tp, fn, fp, tn = (cell[k] for k in ("tp", "fn", "fp", "tn"))
                        long_rows.append({
                            "experiment": experiment, "drive": drive_name,
                            "model": name, "family": family, "seed": seed,
                            "test_month": fold.test_month,
                            "train_start": trained.train_start,
                            "train_end": trained.train_end,
                            "rule": rule, "horizon_days": horizon,
                            "threshold_source": args.threshold_source,
                            "far_target": target, "threshold": cell["threshold"],
                            "n_failed": scored["n_failed"], "n_healthy": scored["n_healthy"],
                            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                            "recall": tp / (tp + fn) if tp + fn else 0.0,
                            "precision": tp / (tp + fp) if tp + fp else 0.0,
                            "far": fp / (fp + tn) if fp + tn else 0.0,
                            "roc_auc": scored["roc_auc"], "pr_auc": scored["pr_auc"],
                        })
                print(f"  {drive_name} | {name} | seed {seed} 채점 완료")

    if not long_rows:
        print("채점할 결과가 없습니다.")
        return 1

    long_path = prefix.with_name(prefix.name + "_long.csv")
    fields = list(long_rows[0].keys())
    with long_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(long_rows)
    print(f"\n[원자료] {long_path}  ({len(long_rows)} 행)")

    # --- 요약: 달을 풀링하고 시드로 평균 -----------------------------------
    frame = pd.DataFrame(long_rows)
    summary = []
    keys = [
        "experiment", "drive", "model", "family",
        "rule", "threshold_source", "far_target",
    ]
    for key, group in frame.groupby(keys, sort=False):
        per_seed = []
        for _, seed_group in group.groupby("seed"):
            tp, fn = int(seed_group["tp"].sum()), int(seed_group["fn"].sum())
            fp, tn = int(seed_group["fp"].sum()), int(seed_group["tn"].sum())
            per_seed.append({
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "far": fp / (fp + tn) if fp + tn else 0.0,
                "roc_auc": float(seed_group["roc_auc"].mean()),
                "pr_auc": float(seed_group["pr_auc"].mean()),
                "tp": float(tp),
            })
        row = dict(zip(keys, key))
        row["n_seeds"] = len(per_seed)
        row["n_failed_pooled"] = int(group["n_failed"].sum() / len(per_seed))
        row["n_healthy_pooled"] = int(group["n_healthy"].sum() / len(per_seed))
        for metric in ("recall", "precision", "far", "roc_auc", "pr_auc", "tp"):
            values = [s[metric] for s in per_seed]
            row[f"{metric}_mean"] = st.mean(values)
            row[f"{metric}_sd"] = st.pstdev(values) if len(values) > 1 else 0.0
        summary.append(row)

    summary_path = prefix.with_name(prefix.name + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"[요약]   {summary_path}  ({len(summary)} 행)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
