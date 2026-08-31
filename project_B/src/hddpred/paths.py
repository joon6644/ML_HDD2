"""프로젝트 경로 상수.

이 모듈만이 파일시스템 레이아웃을 알고 있다. 다른 모듈은 여기서 얻은
경로만 사용한다.

레이아웃:

    26_2_COIN/
    ├─ data/raw/                  <- 원본. 읽기 전용. 절대 쓰지 않는다.
    └─ project_B/
       ├─ configs/
       ├─ data/                   <- 파생 데이터. 지워도 재생성된다.
       │  ├─ canonical/<drive>/canon=<hash>/
       │  ├─ features/<drive>/canon=<hash>/feat=<hash>/
       │  ├─ labels/<drive>/canon=<hash>/label=<hash>/
       │  └─ splits/<drive>/split=<hash>/
       ├─ runs/<experiment>/<drive>/<model>/seed<N>/fold<N>/
       └─ src/hddpred/
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent

CONFIG_DIR = PROJECT_ROOT / "configs"
DERIVED_ROOT = PROJECT_ROOT / "data"
RUNS_ROOT = PROJECT_ROOT / "runs"
TMP_DIR = PROJECT_ROOT / ".tmp"

RAW_ROOT = REPO_ROOT / "data" / "raw"

CANONICAL_ROOT = DERIVED_ROOT / "canonical"
FEATURES_ROOT = DERIVED_ROOT / "features"
LABELS_ROOT = DERIVED_ROOT / "labels"
SPLITS_ROOT = DERIVED_ROOT / "splits"
MANIFEST_ROOT = DERIVED_ROOT / "manifests"


def canonical_dir(drive: str, canon_hash: str) -> Path:
    return CANONICAL_ROOT / drive / f"canon={canon_hash}"


def features_dir(drive: str, canon_hash: str, feat_hash: str) -> Path:
    return FEATURES_ROOT / drive / f"canon={canon_hash}" / f"feat={feat_hash}"


def labels_dir(drive: str, canon_hash: str, label_hash: str) -> Path:
    return LABELS_ROOT / drive / f"canon={canon_hash}" / f"label={label_hash}"


def splits_dir(drive: str, split_hash: str) -> Path:
    return SPLITS_ROOT / drive / f"split={split_hash}"


def run_dir(
    experiment: str, drive: str, model: str, seed: int, fold: int | None = None
) -> Path:
    base = RUNS_ROOT / experiment / drive / model / f"seed{seed}"
    return base if fold is None else base / f"fold{fold:02d}"


def resolve_placeholders(value: str) -> str:
    """설정 문자열의 ${repo_root} / ${project_root} 를 실제 경로로 바꾼다."""
    return value.replace("${repo_root}", REPO_ROOT.as_posix()).replace(
        "${project_root}", PROJECT_ROOT.as_posix()
    )
