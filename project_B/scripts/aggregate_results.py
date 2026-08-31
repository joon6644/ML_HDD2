"""fold 지표를 모아 모델 비교표와 선정 결과를 만든다.

    python scripts/aggregate_results.py model_comparison

선정은 validation 기준으로만 한다. test 컬럼은 표에 실리되 선정에는
들어가지 않는다 (configs/evaluation.yaml 의 selection).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hddpred import config as cfg_mod, paths  # noqa: E402
from hddpred.evaluation import aggregate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="실험 결과 집계")
    parser.add_argument("experiment", help="runs/ 아래 실험 이름")
    parser.add_argument("--evaluation", default="configs/evaluation.yaml")
    args = parser.parse_args()

    evaluation_cfg = cfg_mod.load_yaml(args.evaluation)
    fold_table = aggregate.collect(paths.RUNS_ROOT, args.experiment)
    summary = aggregate.summarize(fold_table)
    selection = aggregate.select_model(summary, evaluation_cfg["selection"])

    out_dir = paths.PROJECT_ROOT / "reports" / args.experiment
    aggregate.write_report(fold_table, summary, selection, out_dir)

    display = [
        "model",
        "val_row_level_pr_auc_mean",
        "val_row_level_pr_auc_std",
        "test_row_level_pr_auc_mean",
        "test_disk_level_recall_mean",
        "test_disk_level_far_mean",
    ]
    with pd.option_context("display.width", 160, "display.max_columns", 40):
        print("\n[모델 비교]")
        print(summary[[c for c in display if c in summary.columns]].to_string(index=False))

    print(f"\n[선정] {selection['selected_model']} "
          f"({selection['primary_metric']}={selection['primary_value']:.4f}, "
          f"결정 근거: {selection['decided_by']})")
    print(f"보고서: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
