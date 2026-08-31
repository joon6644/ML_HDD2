"""canonical -> features (인과적 파생 피처).

여기서 만드는 변환은 전부 "같은 디스크의 같은 segment 안에서 과거만 참조"
한다. 따라서 fold와 무관하고 전 구간에서 한 번만 계산하면 된다. fold마다
다시 계산하는 것은 낭비이고, 그렇다고 미래를 보는 것도 아니다.

fold에 종속되는 것은 스케일러뿐이다. 스케일러는 여러 디스크에 걸친 통계를
적합시키므로 train 구간에서만 적합시켜야 한다. training/dataset.py 가 처리한다.

윈도우 함수는 ROWS 기준이다. canonical 레이어가 segment 안의 날짜를 빠짐없이
채워 두었기 때문에 ROWS n PRECEDING 이 곧 n일 전이 된다.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import duckdb

from .. import config as cfg_mod
from .. import paths
from ..data import canonicalize
from ..tracking import provenance

AGE_COLUMNS = ["age_days", "segment_age_days"]
KEY_COLUMNS = ["serial_number", "record_date", "month", "segment"]


def _quoted(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def feature_hash(features_cfg: dict, canon_hash: str) -> str:
    # 스케일링과 학습 표본 추출은 이 레이어의 내용에 영향을 주지 않는다.
    relevant = {k: v for k, v in features_cfg.items() if k in {"base", "tabular"}}
    return cfg_mod.config_hash(relevant, {"canon": canon_hash})


def resolve_derived_columns(features_cfg: dict, available: list[str]) -> list[str]:
    """파생을 적용할 SMART 컬럼을 결정한다."""
    base = features_cfg["base"]
    mode = base.get("derived_on", "critical")
    if mode == "all":
        return list(available)
    if mode == "critical":
        wanted = list(base.get("critical_smart", []))
        selected = [c for c in wanted if c in available]
        skipped = [c for c in wanted if c not in available]
        if skipped:
            print(f"  [features] canonical 에 없어 건너뜀: {', '.join(skipped)}")
        if not selected:
            raise ValueError(
                "critical_smart 중 canonical 에 존재하는 컬럼이 없습니다. "
                "configs/features.yaml 의 critical_smart 를 확인하세요."
            )
        return selected
    raise ValueError(f"알 수 없는 derived_on: {mode!r}")


def _expressions(features_cfg: dict, derived: list[str]) -> list[tuple[str, str]]:
    """(별칭, SQL 식) 목록을 만든다."""
    base = features_cfg["base"]
    part = "PARTITION BY serial_number, segment ORDER BY record_date"
    out: list[tuple[str, str]] = []

    for column in derived:
        col = _quoted(column)
        for lag in base.get("diff_lags", []):
            out.append(
                (
                    f"{column}_d{lag}",
                    f"{col} - LAG({col}, {lag}) OVER ({part})",
                )
            )
        for window in base.get("rolling_windows", []):
            frame = f"{part} ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
            for stat in base.get("rolling_stats", []):
                func = {"mean": "AVG", "std": "STDDEV_SAMP", "max": "MAX", "min": "MIN"}[
                    stat
                ]
                out.append(
                    (f"{column}_{stat}{window}", f"{func}({col}) OVER ({frame})")
                )
        if base.get("include_since_start", True):
            frame = f"{part} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
            out.append(
                (
                    f"{column}_since_start",
                    f"{col} - FIRST_VALUE({col}) OVER ({frame})",
                )
            )
    return out


def build(
    drive_name: str,
    canonical_path: Path,
    features_cfg: dict,
    preprocessing_cfg: dict,
    *,
    force: bool = False,
) -> Path:
    """피처 레이어를 만들고 그 디렉터리를 돌려준다."""
    canon_hash = provenance.read(canonical_path)["config_hash"]
    feat_hash = feature_hash(features_cfg, canon_hash)
    out_dir = paths.features_dir(drive_name, canon_hash, feat_hash)

    if provenance.is_complete(out_dir) and not force:
        print(f"[features] 재사용: {out_dir}")
        return out_dir

    provenance.clear(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"

    smart = canonicalize.smart_columns(canonical_path)
    derived = resolve_derived_columns(features_cfg, smart)
    expressions = _expressions(features_cfg, derived)

    duck = preprocessing_cfg.get("duckdb", {})
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={int(duck.get('threads', 8))}")
    con.execute(f"PRAGMA max_memory='{duck.get('max_memory', '16GB')}'")
    con.execute(
        f"PRAGMA temp_directory='{Path(duck.get('temp_dir', paths.TMP_DIR)).as_posix()}'"
    )
    con.execute("SET preserve_insertion_order=false")

    source = canonicalize.dataset_glob(canonical_path)
    started = time.time()
    print(f"\n[features] {drive_name}: 파생 대상 {len(derived)}개 컬럼")

    select_parts = [
        "serial_number",
        "record_date",
        "month",
        "segment",
    ]
    # 피처 이름은 여기서 확정한다. 기록된 parquet 을 DESCRIBE 해서 얻으면
    # 안 된다. 파생 폴더 경로가 canon=<hash>/feat=<hash> 형태라 DuckDB 의
    # hive partitioning 이 canon / feat 를 컬럼으로 만들어 목록에 섞인다.
    feature_columns: list[str] = []
    if features_cfg["base"].get("include_raw", True):
        select_parts += [f"{_quoted(c)}::FLOAT AS {_quoted(c)}" for c in smart]
        feature_columns += list(smart)
    if features_cfg["base"].get("include_age", True):
        select_parts += [
            "date_diff('day', MIN(record_date) OVER (PARTITION BY serial_number),"
            " record_date)::INTEGER AS age_days",
            "date_diff('day', MIN(record_date) OVER"
            " (PARTITION BY serial_number, segment), record_date)::INTEGER"
            " AS segment_age_days",
        ]
        feature_columns += list(AGE_COLUMNS)
    select_parts += [f"({expr})::FLOAT AS {_quoted(alias)}" for alias, expr in expressions]
    feature_columns += [alias for alias, _ in expressions]

    overlap = sorted(set(feature_columns) & set(KEY_COLUMNS))
    if overlap:
        raise ValueError(f"피처 이름이 키 컬럼과 충돌합니다: {overlap}")

    if data_dir.exists():
        shutil.rmtree(data_dir)
    con.execute(
        f"""
        COPY (
            SELECT {", ".join(select_parts)}
            FROM read_parquet('{source}', hive_partitioning=true)
            ORDER BY month, serial_number, record_date
        )
        TO '{data_dir.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (month), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
        """
    )

    written = f"'{(data_dir / '**' / '*.parquet').as_posix()}', hive_partitioning=true"
    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet({written})").fetchall()
    stored = {row[0] for row in schema}
    missing = [c for c in feature_columns if c not in stored]
    if missing:
        raise RuntimeError(f"기록된 피처 레이어에 없는 컬럼: {missing}")
    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet({written})").fetchone()[0]

    print(
        f"  {rows:,}행 / 피처 {len(feature_columns)}개 "
        f"({time.time() - started:.1f}s)"
    )

    sequence_columns = list(smart)
    if features_cfg["base"].get("include_age", True):
        sequence_columns = sequence_columns + AGE_COLUMNS

    provenance.write(
        out_dir,
        stage="features",
        config_hash=feat_hash,
        configs={"features": {k: features_cfg[k] for k in ("base", "tabular")}},
        parents={"canonical": canon_hash},
        inputs=[canonical_path.as_posix()],
        output_schema={row[0]: row[1] for row in schema},
        stats={
            "rows": rows,
            "n_features": len(feature_columns),
            "feature_columns": feature_columns,
            "tabular_columns": feature_columns,
            "sequence_columns": sequence_columns,
            "derived_on": derived,
            "elapsed_seconds": round(time.time() - started, 1),
        },
    )
    con.close()
    print(f"[features] 완료: {out_dir}")
    return out_dir


def dataset_glob(features_path: Path) -> str:
    return (features_path / "data" / "**" / "*.parquet").as_posix()


def tabular_columns(features_path: Path) -> list[str]:
    return list(provenance.read(features_path)["stats"]["tabular_columns"])


def sequence_columns(features_path: Path) -> list[str]:
    return list(provenance.read(features_path)["stats"]["sequence_columns"])
