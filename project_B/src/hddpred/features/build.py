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


def _needs_complexity(features_cfg: dict) -> bool:
    base = features_cfg["base"]
    return bool(base.get("cid_windows")) or bool(base.get("asfd_windows"))


def _d1_helper_alias(column: str) -> str:
    """CID/ASFD 가 참조하는 내부 차분 컬럼 이름. 실제 SMART 컬럼(smart_N_raw)과
    겹치지 않도록 이중 밑줄로 시작한다. feature_columns 에는 안 들어간다 —
    사용자에게 노출되는 피처가 아니라 계산용 중간값이다."""
    return f"__d1_{column}"


def _complexity_expressions(
    features_cfg: dict, derived: list[str]
) -> list[tuple[str, str]]:
    """CID / ASFD (별칭, SQL 식) 목록. 창 안의 연속 차분을 합산하는 지표라
    diff/rolling 처럼 한 단계 SQL로 못 쓴다 — 먼저 lag-1 차분을 만들고, 그
    위에 다시 윈도우 합을 얹어야 한다(_d1_helper_alias 로 만든 중간 컬럼을
    참조). build() 가 이 값을 내부 서브쿼리에 먼저 만들어 둔다.

    CID (Complexity-Invariant Distance 의 복잡도 추정치, Batista et al. 2011):
        CE(w) = sqrt( sum_{i=1}^{w-1} (x_i - x_{i-1})^2 )
        시계열이 창 안에서 얼마나 요동쳤는지를 재는 표준 지표. 값이 크면
        그 구간이 "복잡"(변화가 잦음)하다는 뜻이다.

    ASFD (Absolute Sum of First Differences):
        sum_{i=1}^{w-1} |x_i - x_{i-1}|
        CID 의 L1 버전. 창 안에서 값이 오르내린 총량(총변동)이다.
    """
    base = features_cfg["base"]
    part = "PARTITION BY serial_number, segment ORDER BY record_date"
    out: list[tuple[str, str]] = []

    windows = list(base.get("cid_windows", [])) + list(base.get("asfd_windows", []))
    bad = [w for w in windows if w < 2]
    if bad:
        raise ValueError(
            f"cid_windows/asfd_windows 는 2 이상이어야 한다 (차분이 최소 1개 "
            f"필요): {bad}"
        )

    for column in derived:
        helper = _quoted(_d1_helper_alias(column))
        # rolling_windows 와 같은 정의(원값 w개)를 맞추려면 차분은 w-1 개만
        # 모아야 한다. 차분 하나가 원값 2개를 쓰므로, 차분을 w 개 모으면
        # 원값을 w+1 개 덮어 rolling 과 창 길이가 어긋난다.
        for window in base.get("cid_windows", []):
            frame = f"{part} ROWS BETWEEN {window - 2} PRECEDING AND CURRENT ROW"
            out.append(
                (f"{column}_cid{window}", f"SQRT(SUM(POWER({helper}, 2)) OVER ({frame}))")
            )
        for window in base.get("asfd_windows", []):
            frame = f"{part} ROWS BETWEEN {window - 2} PRECEDING AND CURRENT ROW"
            out.append((f"{column}_asfd{window}", f"SUM(ABS({helper})) OVER ({frame})"))
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
    complexity = _complexity_expressions(features_cfg, derived)

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

    if not complexity:
        # 기존 경로. 단일 SELECT 로 전부 계산한다.
        query = f"""
            SELECT {", ".join(select_parts)}
            FROM read_parquet('{source}', hive_partitioning=true)
            ORDER BY month, serial_number, record_date
        """
    else:
        # CID/ASFD 는 "차분 -> 창 안에서 합산"의 2단계 계산이라 한 SELECT 로
        # 못 쓴다. 안쪽에서 raw + 기존 파생 + lag-1 차분(내부용)을 만들고,
        # 바깥에서 그 차분 위에 윈도우 합을 얹는다. 내부 차분 컬럼은
        # feature_columns 에 없으므로 최종 출력에서 EXCLUDE 로 뺀다.
        part = "PARTITION BY serial_number, segment ORDER BY record_date"
        inner_parts = list(select_parts)
        helper_aliases = []
        for column in derived:
            col = _quoted(column)
            alias = _d1_helper_alias(column)
            helper_aliases.append(alias)
            inner_parts.append(
                f"({col} - LAG({col}, 1) OVER ({part}))::FLOAT AS {_quoted(alias)}"
            )
        outer_exclude = ", ".join(_quoted(a) for a in helper_aliases)
        outer_parts = [f"* EXCLUDE ({outer_exclude})"]
        outer_parts += [
            f"({expr})::FLOAT AS {_quoted(alias)}" for alias, expr in complexity
        ]
        feature_columns += [alias for alias, _ in complexity]
        query = f"""
            SELECT {", ".join(outer_parts)}
            FROM (
                SELECT {", ".join(inner_parts)}
                FROM read_parquet('{source}', hive_partitioning=true)
            )
            ORDER BY month, serial_number, record_date
        """

    con.execute(
        f"""
        COPY ({query})
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
