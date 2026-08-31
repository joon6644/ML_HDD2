"""fold 지표 -> 최종 비교 결과와 모델 선정.

선정 규칙은 configs/evaluation.yaml 의 selection 에 미리 선언해 둔다.
실험을 다 돌린 뒤에 기준을 고르면 그 자체가 test 를 본 것과 같아진다.

    1. validation PR-AUC 평균을 1차 기준으로 쓴다
    2. fold 간 변동성이 큰 모델은 불리하게 본다 (mean - w * std)
    3. near-tie 이면 precision floor 아래의 recall 로 비교한다
    4. 그래도 비슷하면 모델 복잡도로 비교한다
    5. test 결과는 선정에 쓰지 않는다
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_FILE = "fold_metrics.json"


def collect(runs_root: Path, experiment: str) -> pd.DataFrame:
    """runs/<experiment> 아래의 모든 fold 지표를 한 표로 모은다."""
    rows = []
    for metrics_path in sorted((runs_root / experiment).rglob(RESULTS_FILE)):
        with metrics_path.open("r", encoding="utf-8") as fh:
            record = json.load(fh)
        flat = {
            "experiment": experiment,
            "drive": record["drive"],
            "model": record["model"],
            "seed": record["seed"],
            "fold": record["fold"]["fold"],
            "test_month": record["fold"]["test_month"],
            "threshold": record["threshold"]["threshold"],
            "path": metrics_path.parent.as_posix(),
        }
        for split in ("val", "test"):
            for layer in ("row_level", "disk_level"):
                for key, value in record["metrics"][split][layer].items():
                    flat[f"{split}_{layer}_{key}"] = value
        flat["fit_seconds"] = record.get("timing", {}).get("fit_seconds")
        flat["complexity"] = record.get("complexity")
        rows.append(flat)

    if not rows:
        raise FileNotFoundError(
            f"{runs_root / experiment} 아래에 {RESULTS_FILE} 가 없습니다."
        )
    return pd.DataFrame(rows)


def summarize(fold_table: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """(drive, model) 단위로 fold/seed 평균과 표준편차를 낸다."""
    metrics = metrics or [
        c
        for c in fold_table.columns
        if c.startswith(("val_", "test_")) and fold_table[c].dtype.kind in "fi"
    ]
    grouped = fold_table.groupby(["drive", "model"], sort=False)
    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_runs"] = grouped.size()
    summary["complexity_median"] = grouped["complexity"].median()
    return summary.reset_index()


def select_model(summary: pd.DataFrame, selection_cfg: dict) -> dict:
    """선언된 규칙으로 모델을 고른다. test 컬럼은 보지 않는다."""
    primary = selection_cfg.get("primary", "val_pr_auc_mean")
    column = {
        "val_pr_auc_mean": "val_row_level_pr_auc_mean",
        "val_disk_f1_mean": "val_disk_level_f1_mean",
    }.get(primary, primary)
    if column not in summary.columns:
        raise KeyError(f"선정 기준 컬럼이 없습니다: {column}")

    table = summary.copy()
    std_column = column.replace("_mean", "_std")
    penalty = float(selection_cfg.get("variance_weight", 1.0))
    if selection_cfg.get("penalize_variance", True) and std_column in table.columns:
        table["score"] = table[column] - penalty * table[std_column].fillna(0.0)
    else:
        table["score"] = table[column]

    table = table.sort_values("score", ascending=False).reset_index(drop=True)
    best = table.iloc[0]
    margin = float(selection_cfg.get("near_tie_margin", 0.0))
    contenders = table.loc[table["score"] >= best["score"] - margin]

    reason = "primary"
    if len(contenders) > 1:
        for tiebreaker in selection_cfg.get("tiebreakers", []):
            if tiebreaker == "model_complexity":
                contenders = contenders.sort_values("complexity_median")
                reason = "model_complexity"
                break
            candidate = {
                "val_recall_at_precision_floor": "val_row_level_recall_mean",
            }.get(tiebreaker, tiebreaker)
            if candidate in contenders.columns:
                contenders = contenders.sort_values(candidate, ascending=False)
                reason = tiebreaker
                break
        best = contenders.iloc[0]

    return {
        "selected_model": str(best["model"]),
        "drive": str(best["drive"]),
        "primary_metric": column,
        "primary_value": float(best[column]),
        "score": float(best["score"]),
        "decided_by": reason,
        "near_tie_models": [str(m) for m in contenders["model"].tolist()],
        "ranking": table[["model", column, "score"]].to_dict("records"),
        "test_used_for_selection": False,
    }


def write_report(
    fold_table: pd.DataFrame, summary: pd.DataFrame, selection: dict, out_dir: Path
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_table.to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "model_summary.csv", index=False, encoding="utf-8-sig")
    with (out_dir / "selection.json").open("w", encoding="utf-8") as fh:
        json.dump(selection, fh, indent=2, ensure_ascii=False, default=float)
    return out_dir
