"""실험 설정 하나를 끝까지 실행한다.

    python scripts/run_experiment.py configs/experiments/smoke.yaml
    python scripts/run_experiment.py configs/experiments/model_comparison.yaml
    python scripts/run_experiment.py <path> --force   # 완료된 run 도 다시

이미 fold_metrics.json 이 있는 run 은 건너뛴다. 중간에 끊겨도 이어서 돌린다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hddpred.experiments.runner import run_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="실험 실행")
    parser.add_argument("experiment", help="configs/experiments/*.yaml")
    parser.add_argument("--force", action="store_true", help="완료된 run 도 다시 실행")
    args = parser.parse_args()

    out = run_experiment(args.experiment, force=args.force)
    print(f"\n실험 완료: {out}")
    print("결과 집계: python scripts/aggregate_results.py <experiment_name>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
