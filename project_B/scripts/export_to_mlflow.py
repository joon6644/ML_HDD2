"""완료된 run 을 MLflow 로 적재한다.

    python scripts/export_to_mlflow.py                     # runs/ 전체
    python scripts/export_to_mlflow.py --experiment smoke  # 하나만
    python scripts/export_to_mlflow.py --with-predictions  # 예측 parquet 까지

이미 적재된 run 은 건너뛴다. 여러 번 돌려도 안전하다.
보기: mlflow ui --backend-store-uri <출력된 tracking uri>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hddpred import paths  # noqa: E402
from hddpred.tracking import mlflow_sink  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="run 디렉터리를 MLflow 로 적재")
    parser.add_argument("--experiment", help="runs/ 아래 실험 이름. 생략하면 전체")
    parser.add_argument(
        "--with-predictions",
        action="store_true",
        help="예측 parquet 도 아티팩트로 올린다 (run 당 약 4MB)",
    )
    args = parser.parse_args()

    result = mlflow_sink.export(
        paths.RUNS_ROOT, args.experiment, with_predictions=args.with_predictions
    )
    print(f"적재 {len(result['logged'])}건 / 이미 있음 {len(result['skipped'])}건")
    for name in result["logged"]:
        print(f"  + {name}")
    print(f"\ntracking uri: {result['tracking_uri']}")
    print(f"보기: mlflow ui --backend-store-uri {result['tracking_uri']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
