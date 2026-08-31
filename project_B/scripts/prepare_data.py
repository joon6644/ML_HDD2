"""raw -> canonical -> features / labels -> split manifest 까지 만든다.

    python scripts/prepare_data.py
    python scripts/prepare_data.py --drive HGST_20HUH721212ALN604
    python scripts/prepare_data.py --force        # 캐시 무시하고 재생성

원본(../data/raw)은 읽기만 한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hddpred import config as cfg_mod  # noqa: E402
from hddpred.experiments.runner import Pipeline, prepare_drive  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="파생 데이터 레이어 생성")
    parser.add_argument(
        "--drive", help="드라이브 이름. 생략하면 data.yaml 의 enabled 전체"
    )
    parser.add_argument(
        "--experiment",
        default="configs/experiments/model_comparison.yaml",
        help="사용할 설정 조합을 정의한 실험 파일",
    )
    parser.add_argument("--force", action="store_true", help="캐시 무시하고 재생성")
    args = parser.parse_args()

    pipeline = Pipeline.from_experiment(cfg_mod.load_yaml(args.experiment))
    drives = (
        [args.drive]
        if args.drive
        else [d["name"] for d in cfg_mod.enabled_drives(pipeline.data)]
    )

    for name in drives:
        prepared = prepare_drive(name, pipeline, force=args.force)
        print(f"\n[{name}] 준비 완료")
        print(f"  canonical : {prepared.canonical_path}")
        print(f"  features  : {prepared.features_path}")
        print(f"  labels    : {prepared.labels_path}")
        print(f"  splits    : {prepared.splits_path} (fold {len(prepared.folds)}개)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
