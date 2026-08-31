"""파생 데이터 계보 기록.

각 파생 폴더에 provenance.json 과 _SUCCESS 마커를 남긴다.

_SUCCESS 가 없는 폴더는 "생성 도중 중단된 것"으로 보고 재사용하지 않는다.
중간에 죽은 parquet을 다음 실행이 조용히 읽는 사고를 막는다.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUCCESS_MARKER = "_SUCCESS"
PROVENANCE_FILE = "provenance.json"


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for module in ("pandas", "numpy", "pyarrow", "duckdb", "sklearn", "torch"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - 없는 패키지는 그냥 건너뛴다
            continue
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def write(
    directory: Path,
    *,
    stage: str,
    config_hash: str,
    configs: dict[str, Any],
    parents: dict[str, str] | None = None,
    inputs: list[str] | None = None,
    output_schema: dict[str, str] | None = None,
    stats: dict[str, Any] | None = None,
) -> Path:
    """provenance.json 을 쓰고 _SUCCESS 마커를 남긴다."""
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": stage,
        "config_hash": config_hash,
        "parents": parents or {},
        "inputs": inputs or [],
        "configs": configs,
        "output_schema": output_schema or {},
        "stats": stats or {},
        "git_commit": git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment(),
    }
    target = directory / PROVENANCE_FILE
    with target.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False, default=str)
    (directory / SUCCESS_MARKER).touch()
    return target


def is_complete(directory: Path) -> bool:
    """완성된 파생 폴더인지 확인한다."""
    return (directory / SUCCESS_MARKER).exists() and (
        directory / PROVENANCE_FILE
    ).exists()


def read(directory: Path) -> dict[str, Any]:
    with (directory / PROVENANCE_FILE).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def clear(directory: Path) -> None:
    """재생성 전에 마커를 지운다. 도중에 죽어도 미완성으로 남는다."""
    marker = directory / SUCCESS_MARKER
    if marker.exists():
        marker.unlink()
