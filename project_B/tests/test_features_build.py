"""CID/ASFD 파생 SQL 이 손계산과 일치하는지 확인한다.

차분 -> 창 합산의 2단계 계산이라 rolling_windows 와 같은 "원값 w개" 정의를
지키려면 차분 컬럼의 윈도우 프레임이 w-1 개여야 한다(차분 하나가 원값 2개를
쓰므로). 이 off-by-one 을 실측(_d1_helper_alias 프레임 폭)으로 고정해 둔다.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pytest

from hddpred.features.build import _complexity_expressions, _d1_helper_alias


def _synthetic_series(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n).round(3)


def _run_sql(values: np.ndarray, windows: list[int], kind: str) -> np.ndarray:
    """build() 이 실제로 쓰는 것과 같은 2단계(내부 차분 -> 외부 윈도우 합)
    쿼리를 작은 합성 테이블에 그대로 재현해서 돌린다."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE t AS SELECT * FROM "
        "(SELECT unnest(range(len(?))) AS record_date, unnest(?) AS x, "
        "'s' AS serial_number, 0 AS segment)",
        [values.tolist(), values.tolist()],
    )
    features_cfg = {
        "base": {
            "cid_windows": windows if kind == "cid" else [],
            "asfd_windows": windows if kind == "asfd" else [],
        }
    }
    complexity = _complexity_expressions(features_cfg, ["x"])
    part = "PARTITION BY serial_number, segment ORDER BY record_date"
    helper = f'"{_d1_helper_alias("x")}"'

    con.execute(
        f"CREATE TABLE inner_t AS SELECT record_date, x, serial_number, segment, "
        f"(x - LAG(x, 1) OVER ({part}))::DOUBLE AS {helper} FROM t"
    )
    outer = ", ".join(f"({expr}) AS c{i}" for i, (_, expr) in enumerate(complexity))
    out = con.execute(
        f"SELECT record_date, {outer} FROM inner_t ORDER BY record_date"
    ).fetchdf()
    con.close()
    return out[[c for c in out.columns if c.startswith("c")]].to_numpy()


def _manual(values: np.ndarray, window: int, kind: str) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - (window - 1))  # window 개 원값 = window-1 개 차분
        pts = values[start : i + 1]
        if len(pts) < 2:
            continue
        diffs = np.diff(pts)
        out[i] = np.sqrt(np.sum(diffs**2)) if kind == "cid" else np.sum(np.abs(diffs))
    return out


@pytest.mark.parametrize("kind", ["cid", "asfd"])
@pytest.mark.parametrize("window", [2, 3, 7, 14])
def test_complexity_matches_manual(kind, window):
    values = _synthetic_series()
    sql_out = _run_sql(values, [window], kind)[:, 0]
    manual_out = _manual(values, window, kind)
    valid = ~np.isnan(manual_out)
    assert valid.sum() > 0
    np.testing.assert_allclose(sql_out[valid], manual_out[valid], atol=1e-6)
    assert np.isnan(sql_out[~valid]).all()


def test_window_covers_exactly_w_raw_points():
    """window=7 이면 rolling_windows=7 과 같은 폭(원값 7개, 차분 6개)을 덮어야
    한다 — 8개(차분 7개)를 덮으면 off-by-one 회귀다."""
    values = np.arange(20, dtype=float)  # 단조증가라 diff 가 전부 1.0
    sql_out = _run_sql(values, [7], "asfd")[:, 0]
    # 원값 7개(차분 6개)를 덮으면 asfd7 = 6.0 이어야 한다 (7개 덮으면 7.0).
    assert sql_out[10] == pytest.approx(6.0)


def test_rejects_window_below_two():
    with pytest.raises(ValueError, match="2 이상"):
        _complexity_expressions({"base": {"cid_windows": [1]}}, ["x"])
