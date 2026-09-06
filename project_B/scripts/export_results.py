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

    val_quantile
        FAR 목표마다 val 월의 정상 디스크 상위 f% 지점을 임곗값으로 잡고,
        그 값을 세 test 월에 그대로 적용한다. test 를 전혀 참조하지 않으면서
        Recall@FAR 을 여러 지점에서 읽을 수 있어 논문 표에 쓰는 소스다.
        val 에서 f% 였던 운영점이 test 에서 그대로 f% 로 재현되지는 않으므로
        long CSV 의 far 열에 실제 오탐률이 따로 남는다.

    validation
        run 디렉터리의 threshold.json 을 그대로 쓴다. runner 가 validation
        에서만 골라 둔 값이라 test 를 전혀 참조하지 않는다. 세 달에 같은
        임곗값이 적용되므로 실제 FAR 은 달마다 목표에서 벗어난다. 그 벗어남
        자체가 "val 에서 고른 운영점이 이후 달에 얼마나 유지되는가"이고,
        long CSV 의 far 열에 그대로 남는다.

내보내는 파일 두 개:

    <out>_long.csv     (모델, 시드, test월, FAR목표) 한 줄. 원자료.
                       여기서 어떤 집계든 다시 만들 수 있다.
    <out>_summary.csv  달마다 지표를 내고 달로 평균, 그 다음 시드로 평균낸
                       요약. 논문 표에 쓰는 값.

집계는 월별 평균이다. 달을 하나의 모집단으로 합치지 않는다.
    달마다  Recall = TP / 고장,  FAR = FP / 정상
    대표값 = 그 달들의 평균,  recall_month_sd = 달 간 표준편차

평가 단위가 (디스크 x 월) 이고 배치 서사도 "한 번 학습해 매달 예측한다" 이므로
이쪽이 서술과 맞고, 달 간 표준편차가 그대로 안정성 근거가 된다. 풀링 대비
표준오차 손해는 실측 1.02배였다 (도시바, 월별 고장 18~35).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from datetime import date, timedelta
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

# notion.md 4장 표의 Recall@FAR 행에 맞춘다.
FAR_TARGETS = [0.001, 0.005, 0.01, 0.05]

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


def disk_rank(model, part, rule: str):
    """행 점수를 디스크 하나당 점수 하나로 접는다. (rank, has_window) 를 준다."""
    if rule not in ("in_horizon", "or"):
        raise ValueError(f"지원하지 않는 rule: {rule!r} (in_horizon | or)")

    frame = pd.DataFrame(
        {"serial": part.serial, "y": part.y, "score": model.predict_proba(part)}
    )
    grouped = frame.groupby("serial")
    disk_score = grouped["score"].max()
    # 정답 구간(고장 H일 전) 을 가진 디스크. 라벨이 곧 구간의 정의다.
    has_window = grouped["y"].max().astype(bool)

    if rule == "or":
        return disk_score, has_window
    inside = frame[frame["y"] == 1].groupby("serial")["score"].max()
    # has_window 인 디스크에는 y=1 행이 반드시 있으므로 결측이 나지 않는다.
    rank = disk_score.where(~has_window, inside.reindex(disk_score.index))
    return rank, has_window


def healthy_quantiles(rank, has_window) -> dict[float, float]:
    """미고장 디스크 점수의 상위 f% 지점을 FAR 목표마다 잡는다."""
    healthy = rank[~has_window].to_numpy()
    return {t: float(np.quantile(healthy, 1.0 - t)) for t in FAR_TARGETS}


def month_windows(start: date, end: date, min_days: int = 15) -> list[tuple[date, date]]:
    """구간을 달력 월 경계로 쪼갠다. 양 끝은 원래 구간에 맞춰 자른다.

    val 창이 월 경계에서 시작하지 않는 경우가 있다. 분할이 날짜 오프셋으로
    잡히기 때문이다 — 실측으로 2025-07-31 ~ 2025-09-30 이 나왔고, 그대로
    쪼개면 첫 조각이 하루(7/31)짜리가 된다. 그 조각은 창 검열(마지막 H일
    제거)로 표본이 0 이 되어 채점이 죽는다.

    그래서 min_days 보다 짧은 조각은 이웃에 붙인다. 목적이 "test 와 같은
    길이의 창에서 디스크 점수를 접는 것" 이므로, 한 달 남짓으로 뭉치는 편이
    하루짜리를 따로 두는 것보다 목적에 맞는다.
    """
    out, cursor = [], start
    while cursor <= end:
        if cursor.month == 12:
            nxt = date(cursor.year + 1, 1, 1)
        else:
            nxt = date(cursor.year, cursor.month + 1, 1)
        out.append([cursor, min(end, nxt - timedelta(days=1))])
        cursor = nxt

    merged: list[list[date]] = []
    for chunk in out:
        span = (chunk[1] - chunk[0]).days + 1
        if span < min_days and merged:
            merged[-1][1] = chunk[1]      # 앞 조각에 붙인다
        elif span < min_days and len(out) > 1:
            out[1][0] = chunk[0]          # 첫 조각이면 뒤 조각에 넘긴다
        else:
            merged.append(chunk)
    return [(a, b) for a, b in merged]


def val_disk_rank(prepared, pipeline, family, window, horizon, threads,
                  model, scaler, rule, cache):
    """검증 구간의 디스크 점수. **월별 창으로 쪼개서** 풀링한다.

    디스크 점수는 창 안의 최댓값이라 창이 길수록 커진다. val 을 두 달 한 창으로
    두고 test 는 한 달씩 채점하면, 50일 최댓값으로 잡은 임곗값을 20일 최댓값에
    들이대는 셈이 되어 임곗값이 계통적으로 높아진다.

    실측(도시바 10-2-6, LSTM): 두 달 한 창의 FAR 1% 임곗값이 0.007782,
    월별로 쪼개 풀링하면 0.003249 로 2.4배 차이가 났다. 중앙값은 거의 같고
    (0.000062 vs 0.000060) 상위 꼬리에서만 벌어진다.

    test 와 같은 단위(디스크 x 월)로 맞추려면 val 도 월별로 쪼개야 한다.
    """
    ranks, flags = [], []
    for wstart, wend in month_windows(*window):
        key = (prepared.name, family, "val", wstart)
        if key not in cache:
            cache[key] = load_part(
                prepared, pipeline, family, wstart, wend, horizon, threads
            )
        part = cache[key]
        if family == "sequence":
            part = rescale(part, scaler)
        rank, has_window = disk_rank(model, part, rule)
        ranks.append(rank)
        flags.append(has_window)
    return pd.concat(ranks, ignore_index=True), pd.concat(flags, ignore_index=True)


def score_month(model, part, rule, thresholds=None):
    """한 달을 채점한다. 모든 지표가 디스크 단위다.

    디스크의 순위 점수는 rule 이 정한다. 그래야 ROC/PR 곡선의 임의의 점이
    아래 혼동행렬과 정확히 같은 판정을 뜻한다 — Recall@FAR 과 ROC-AUC 가
    같은 곡선의 두 읽기가 되고, 논문 표에서 나란히 놓을 수 있다.

        in_horizon  고장 디스크는 정답 구간(y=1 행) 안 최댓값.
                    미고장 디스크는 그 달 전체 최댓값 (아무 때나 울리면 FP).
        or          둘 다 그 달 전체 최댓값.

    thresholds: {라벨: 임곗값}. None 이면 그 달 정상 디스크 상위 f% 로 잡는다.
    """
    rank, has_window = disk_rank(model, part, rule)

    actual = has_window.astype(int).to_numpy()
    out = {
        "n_failed": int(has_window.sum()),
        "n_healthy": int((~has_window).sum()),
        "roc_auc": float(roc_auc_score(actual, rank.to_numpy())),
        "pr_auc": float(average_precision_score(actual, rank.to_numpy())),
    }
    if thresholds is None:
        thresholds = healthy_quantiles(rank, has_window)

    out["cells"] = {}
    for label, threshold in thresholds.items():
        alarm = rank >= threshold
        out["cells"][label] = {
            "threshold": float(threshold),
            "tp": int((has_window & alarm).sum()),
            "fn": int((has_window & ~alarm).sum()),
            "fp": int((~has_window & alarm).sum()),
            "tn": int((~has_window & ~alarm).sum()),
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
        choices=["month_quantile", "val_quantile", "validation"],
        help="month_quantile: 달마다 test 정상 디스크 상위 f%% (test 참조) | "
        "val_quantile: val 월 정상 디스크 상위 f%% 를 세 달에 그대로 (test 미참조) | "
        "validation: run 의 threshold.json 한 점을 세 달에 그대로",
    )
    args = parser.parse_args()

    cfg = cfg_mod.load_yaml(paths.CONFIG_DIR / "experiments" / f"{args.experiment}.yaml")
    pipeline = Pipeline.from_experiment(cfg)
    rule = args.rule or pipeline.evaluation["disk_level"].get("rule", "in_horizon")
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
                if args.threshold_source == "validation":
                    chosen = threshold_mod.load(run_dir)
                    thresholds = {nominal_target: float(chosen["threshold"])}
                elif args.threshold_source == "val_quantile":
                    thresholds = healthy_quantiles(
                        *val_disk_rank(
                            prepared, pipeline, family, trained.window("val"),
                            horizon, threads, model, scaler, rule, cache,
                        )
                    )
                for fold in folds:
                    start, end = fold.window("test")
                    key = (drive_name, family, "test", start)
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

    # --- 요약: 달마다 지표를 내고 달로 평균 ---------------------------------
    #
    # 달을 하나의 모집단으로 합치지(풀링) 않는다. 평가 단위가 (디스크 x 월) 이고
    # 배치 서사도 "한 번 학습해 매달 예측한다" 이므로, 달마다 지표를 내고 그
    # 평균을 대표값으로 쓴다. 그러면 달 간 표준편차가 그대로 안정성 근거가 되고,
    # 디스크-월이라는 합성 단위를 표에 안 꺼내도 된다.
    #
    # 통계적 손해는 거의 없다. 균등가중 평균의 분산은 (1/M^2) * sum p(1-p)/n_i
    # 이고 풀링은 p(1-p)/sum(n_i) 인데, 실측(도시바 월별 고장 18~35)에서 표준
    # 오차가 0.0354 vs 0.0348 로 1.02배였다. 달별 표본이 크게 어긋나면 이 차이가
    # 벌어지므로, 그때는 풀링을 다시 검토해야 한다.
    #
    # Recall 과 FAR 을 같은 방식으로 평균낸다. 한쪽만 풀링하면 분자와 분모의
    # 가중이 어긋난다.
    #
    # 시드는 하나다. 변동을 보이는 축은 시드가 아니라 달이다 — 같은 모델을
    # 여러 시드로 돌린 평균은 배치에서 재현되지 않는 수치이고, 달 간 변동은
    # "재학습 없이 다음 달을 예측한다" 는 이 문제의 실제 불확실성이다.
    frame = pd.DataFrame(long_rows)
    seeds = sorted(frame["seed"].unique())
    if len(seeds) != 1:
        raise SystemExit(
            f"시드가 {len(seeds)}개다: {seeds}. 이 파이프라인은 단일 시드만 쓴다 — "
            "experiment yaml 의 seeds 를 하나로 두어라."
        )

    summary = []
    keys = [
        "experiment", "drive", "model", "family",
        "rule", "threshold_source", "far_target",
    ]
    for key, group in frame.groupby(keys, sort=False):
        monthly = []
        for _, m in group.groupby("test_month"):
            tp, fn = int(m["tp"].sum()), int(m["fn"].sum())
            fp, tn = int(m["fp"].sum()), int(m["tn"].sum())
            monthly.append({
                "recall": tp / (tp + fn) if tp + fn else float("nan"),
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "far": fp / (fp + tn) if fp + tn else 0.0,
                "roc_auc": float(m["roc_auc"].mean()),
                "pr_auc": float(m["pr_auc"].mean()),
                "tp": float(tp),
            })
        # 고장 0 인 달은 recall 이 정의되지 않는다. split 의 min_test_failures
        # 가 막고 있지만, 뚫렸을 때 조용히 0 으로 세지 않도록 빼고 평균낸다.
        valid = [m for m in monthly if m["recall"] == m["recall"]]

        row = dict(zip(keys, key))
        row["seed"] = int(seeds[0])
        row["n_months"] = len(monthly)
        row["n_failed_total"] = int(group["n_failed"].sum())
        row["n_healthy_total"] = int(group["n_healthy"].sum())
        for metric in ("recall", "precision", "far", "roc_auc", "pr_auc", "tp"):
            source = valid if metric == "recall" else monthly
            values = [m[metric] for m in source]
            row[metric] = st.mean(values) if values else 0.0
            # 달 간 산포. 재학습 없이 여러 달을 예측할 때의 안정성 근거다.
            row[f"{metric}_month_sd"] = st.stdev(values) if len(values) > 1 else 0.0
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
