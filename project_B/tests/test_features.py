"""피처 레이어와 fold 조립 검증.

여기서 잡는 실제 사고: 파생 폴더 경로가 canon=<hash>/feat=<hash> 라서
DuckDB 의 hive partitioning 이 canon / feat 를 컬럼으로 만들어 낸다. 그것을
피처 목록으로 쓰면 모델 입력에 hash 문자열이 섞인다.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from conftest import make_canonical

from hddpred import config as cfg_mod, paths
from hddpred.features import build as features_build
from hddpred.features import fold as fold_mod
from hddpred.labeling import horizon_labels

SMART = ("smart_5_raw", "smart_197_raw")
LABELING_CFG = {
    "horizon_days": 10,
    "exclude_post_failure_rows": True,
    "require_horizon_observability": True,
    "min_history_days": 0,
}


@pytest.fixture
def features_cfg():
    cfg = cfg_mod.load_yaml(paths.CONFIG_DIR / "features.yaml")
    cfg["base"]["critical_smart"] = list(SMART)
    return cfg


@pytest.fixture
def canonical(derived_root):
    return make_canonical(
        derived_root,
        {
            "A": {
                "start": date(2020, 1, 1),
                "end": date(2020, 6, 30),
                "failure_date": date(2020, 6, 30),
            },
            "B": {"start": date(2020, 1, 1), "end": date(2020, 6, 30)},
        },
        smart_columns=SMART,
    )


@pytest.fixture
def built(canonical, features_cfg, preprocessing_cfg):
    features_path = features_build.build(
        "TEST_DRIVE", canonical, features_cfg, preprocessing_cfg
    )
    labels_path = horizon_labels.build(
        "TEST_DRIVE", canonical, LABELING_CFG, preprocessing_cfg
    )
    return features_path, labels_path


def test_feature_columns_exclude_hive_partition_keys(built):
    features_path, _ = built
    columns = features_build.tabular_columns(features_path)
    assert "canon" not in columns
    assert "feat" not in columns
    assert "month" not in columns
    assert "segment" not in columns


def test_feature_count_matches_the_configuration(built, features_cfg):
    features_path, _ = built
    base = features_cfg["base"]
    per_column = (
        len(base["diff_lags"])
        + len(base["rolling_windows"]) * len(base["rolling_stats"])
        + (1 if base["include_since_start"] else 0)
    )
    expected = len(SMART) + 2 + per_column * len(SMART)
    assert len(features_build.tabular_columns(features_path)) == expected


def test_diff_of_a_linear_counter_is_the_lag(built):
    """합성 데이터의 SMART 값은 0,1,2,... 이므로 d1 = 1, d7 = 7 이어야 한다."""
    features_path, labels_path = built
    matrix = fold_mod.load_tabular(
        features_path,
        labels_path,
        date(2020, 3, 1),
        date(2020, 3, 31),
        threads=2,
    )
    columns = matrix.columns
    d1 = matrix.X[:, columns.index("smart_5_raw_d1")]
    d7 = matrix.X[:, columns.index("smart_5_raw_d7")]
    np.testing.assert_allclose(d1, 1.0)
    np.testing.assert_allclose(d7, 7.0)


def test_tabular_matrix_is_numeric(built):
    features_path, labels_path = built
    matrix = fold_mod.load_tabular(
        features_path, labels_path, date(2020, 3, 1), date(2020, 3, 31), threads=2
    )
    assert matrix.X.dtype == np.float32
    assert matrix.X.shape[1] == len(matrix.columns)
    assert len(matrix) > 0


def test_drop_mode_keeps_only_full_lookbacks(built):
    features_path, labels_path = built
    lookback = 30
    sequence = fold_mod.load_sequence(
        features_path,
        labels_path,
        date(2020, 3, 1),
        date(2020, 3, 31),
        lookback=lookback,
        short_history="drop",
        threads=2,
    )
    assert len(sequence) > 0
    assert int(sequence.end_index.min()) >= lookback - 1
    assert sequence.valid_len is None
    assert sequence.n_features == sequence.matrix.shape[1]


def test_pad_mode_keeps_every_labelled_sample(built):
    """관측 이력이 짧다고 표본을 버리지 않는다. 트리 모델과 모집단이 같아야 한다."""
    features_path, labels_path = built
    window = (date(2020, 1, 1), date(2020, 1, 31))
    tabular = fold_mod.load_tabular(features_path, labels_path, *window, threads=2)
    padded = fold_mod.load_sequence(
        features_path, labels_path, *window, lookback=30, short_history="pad", threads=2
    )
    dropped = fold_mod.load_sequence(
        features_path, labels_path, *window, lookback=30, short_history="drop", threads=2
    )
    assert len(padded) == len(tabular)
    assert len(dropped) < len(padded)
    # mask 채널이 하나 붙는다.
    assert padded.n_features == padded.matrix.shape[1] + 1
    assert padded.valid_len.max() <= padded.lookback
    assert padded.short_sample_rate > 0


def test_padded_window_replicates_the_first_row_and_marks_the_mask():
    """패딩 구간은 그 segment 의 첫 행 복제 + mask 0 이어야 한다."""
    from hddpred.models.sequence import WindowDataset

    matrix = np.arange(20, dtype=np.float32).reshape(10, 2)
    dataset = WindowDataset(
        matrix,
        end_index=np.array([2]),
        lookback=5,
        y=np.array([1], dtype=np.int8),
        valid_len=np.array([3]),
    )
    window, _ = dataset[0]
    window = window.numpy()
    assert window.shape == (5, 3)  # 피처 2 + mask 1
    # 앞 두 칸은 실제 첫 행(행 0)의 복제
    np.testing.assert_allclose(window[0, :2], matrix[0])
    np.testing.assert_allclose(window[1, :2], matrix[0])
    # 뒤 세 칸은 행 0,1,2
    np.testing.assert_allclose(window[2:, :2], matrix[0:3])
    assert window[:, 2].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_unknown_short_history_is_rejected(built):
    features_path, labels_path = built
    with pytest.raises(ValueError, match="short_history"):
        fold_mod.load_sequence(
            features_path,
            labels_path,
            date(2020, 3, 1),
            date(2020, 3, 31),
            lookback=30,
            short_history="nope",
            threads=2,
        )


def test_sequence_window_never_passes_the_observation_time(built):
    """윈도우 마지막 행이 곧 표본의 관측 시점이다."""
    features_path, labels_path = built
    sequence = fold_mod.load_sequence(
        features_path,
        labels_path,
        date(2020, 3, 1),
        date(2020, 3, 31),
        lookback=10,
        short_history="drop",
        threads=2,
    )
    # age_days 는 단조 증가하므로 윈도우 끝 값이 최댓값이어야 한다.
    age = sequence.columns.index("age_days")
    for sample in range(0, len(sequence), max(len(sequence) // 20, 1)):
        end = int(sequence.end_index[sample])
        window = sequence.matrix[end - sequence.lookback + 1 : end + 1, age]
        assert window[-1] == window.max()


def test_stride_sampling_keeps_every_positive(built, features_cfg):
    features_path, labels_path = built
    full = fold_mod.load_tabular(
        features_path, labels_path, date(2020, 6, 1), date(2020, 6, 20), threads=2
    )
    sampled = fold_mod.load_tabular(
        features_path,
        labels_path,
        date(2020, 6, 1),
        date(2020, 6, 20),
        sampling_cfg={"strategy": "stride", "negative_stride": 5},
        threads=2,
    )
    assert sampled.y.sum() == full.y.sum()
    assert len(sampled) < len(full)
