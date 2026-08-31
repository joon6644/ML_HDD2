"""테스트 공통 픽스처.

파생 데이터 루트를 tmp_path 로 갈아끼워서 테스트가 실제 data/ 를 건드리지
않게 한다. paths 모듈의 상수를 참조하는 시점이 함수 호출 시점이므로
monkeypatch 로 교체하면 그대로 반영된다.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import pytest

from hddpred import paths
from hddpred.tracking import provenance


@pytest.fixture
def derived_root(tmp_path, monkeypatch):
    for name in ("CANONICAL_ROOT", "FEATURES_ROOT", "LABELS_ROOT", "SPLITS_ROOT"):
        monkeypatch.setattr(paths, name, tmp_path / name.split("_")[0].lower())
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(paths, "TMP_DIR", tmp_path / "tmp")
    return tmp_path


def make_canonical(
    root,
    disks: dict[str, dict],
    *,
    drive: str = "TEST_DRIVE",
    smart_columns: tuple[str, ...] = ("smart_5_raw", "smart_197_raw"),
):
    """합성 canonical 레이어를 만든다.

    disks 예:
        {"A": {"start": date(2020, 1, 1), "end": date(2020, 3, 31),
               "failure_date": date(2020, 3, 31)}}
    """
    rows = []
    for serial, spec in disks.items():
        current = spec["start"]
        index = 0
        while current <= spec["end"]:
            rows.append(
                {
                    "serial_number": serial,
                    "record_date": current,
                    "month": current.strftime("%Y-%m"),
                    "segment": spec.get("segment", 0),
                    "model": drive,
                    "failure": 1 if current == spec.get("failure_date") else 0,
                    **{c: float(index) for c in smart_columns},
                }
            )
            current += timedelta(days=1)
            index += 1

    # record_date 는 datetime.date 그대로 둔다. pyarrow 가 date32 로 저장하므로
    # 실제 canonical 레이어와 같은 타입이 된다.
    frame = pd.DataFrame(rows)
    directory = root / "canonical" / drive / "canon=testhash"
    data_dir = directory / "data"
    for month, part in frame.groupby("month"):
        target = data_dir / f"month={month}"
        target.mkdir(parents=True, exist_ok=True)
        part.drop(columns=["month"]).to_parquet(target / "part.parquet", index=False)

    provenance.write(
        directory,
        stage="canonical",
        config_hash="testhash",
        configs={},
        stats={"smart_columns": list(smart_columns), "n_disks": len(disks)},
    )
    return directory


@pytest.fixture
def preprocessing_cfg(tmp_path):
    return {"duckdb": {"threads": 2, "max_memory": "2GB", "temp_dir": str(tmp_path / "duck")}}


def read_labels(labels_path) -> pd.DataFrame:
    frames = [
        pd.read_parquet(p) for p in sorted((labels_path / "data").rglob("*.parquet"))
    ]
    frame = pd.concat(frames, ignore_index=True)
    for column in ("record_date", "failure_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame.sort_values(["serial_number", "record_date"]).reset_index(drop=True)


def read_provenance(directory) -> dict:
    with (directory / "provenance.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


__all__ = ["make_canonical", "read_labels", "read_provenance", "date"]
