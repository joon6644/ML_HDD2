"""월 단위 재학습 프로토콜 검증.

  Expanding-window retraining + recent-month retention + negative subsampling

  t월까지 학습 : t+1월 검증 : t+2월 테스트, 한 달씩 이동
  창 기준 검열 (창 끝 H일은 창 안에서 고장이 확인되지 않으면 제외)
  학습 표본 = 과거 양성 전량 + 직전 K개월 전량 + 그 이전 음성 일부

warm start 기계는 코드에 남아 있고 (models/*.warm_start) 실험 설정의
warm_start 플래그로만 켠다. 이 프로토콜은 매달 새로 학습하므로 false 다.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from conftest import make_canonical

from hddpred import config as cfg_mod, paths
from hddpred.features import build as features_build
from hddpred.features import fold as fold_mod
from hddpred.labeling import horizon_labels
from hddpred.models import registry
from hddpred.splits import forward

MONTHS = [f"{y}-{m:02d}" for y in (2021, 2022) for m in range(1, 13)]
MONTHLY_CFG = {
    "protocol": "forward_chaining",
    "train_window": "rolling",
    "train_months": 1,
    "min_train_months": 1,
    "val_months": 1,
    "test_months": 1,
    "n_folds": 6,
    "embargo_days": 0,
    "allow_label_overlap": True,
    "min_test_failures": 1,
    "min_val_failures": 1,
}


# --------------------------------------------------------------------------
# 분할
# --------------------------------------------------------------------------
def test_windows_are_three_consecutive_months():
    folds = forward.plan_folds(MONTHS, MONTHLY_CFG, embargo_days=0)
    assert folds
    for fold in folds:
        train_start, train_end = fold.window("train")
        val_start, val_end = fold.window("val")
        test_start, test_end = fold.window("test")
        # 사이에 빈 날이 없어야 한다.
        assert val_start == train_end + timedelta(days=1)
        assert test_start == val_end + timedelta(days=1)
        # 각 구간이 한 달이어야 한다.
        assert (val_end - val_start).days in range(27, 31)
        assert test_start.day == 1


def test_folds_slide_one_month_at_a_time():
    folds = forward.plan_folds(MONTHS, MONTHLY_CFG, embargo_days=0)
    months = [f.test_month for f in folds]
    assert months == sorted(months)
    for earlier, later in zip(folds, folds[1:]):
        gap = later.window("train")[0] - earlier.window("train")[0]
        assert 28 <= gap.days <= 31


def test_zero_embargo_requires_explicit_acknowledgement(tmp_path, monkeypatch):
    from hddpred.tracking import provenance

    monkeypatch.setattr(paths, "SPLITS_ROOT", tmp_path / "splits")
    labels_dir = tmp_path / "labels"
    provenance.write(labels_dir, stage="labels", config_hash="x", configs={})

    strict = dict(MONTHLY_CFG, allow_label_overlap=False)
    with pytest.raises(ValueError, match="allow_label_overlap"):
        forward.build("TEST_DRIVE", labels_dir, strict, {"horizon_days": 10})


# --------------------------------------------------------------------------
# 창 기준 검열
# --------------------------------------------------------------------------
HORIZON = 10
LABELING = {
    "horizon_days": HORIZON,
    "exclude_post_failure_rows": True,
    "require_horizon_observability": True,
    "min_history_days": 0,
}


@pytest.fixture
def built(derived_root, preprocessing_cfg):
    canonical = make_canonical(
        derived_root,
        {
            # 3월 5일 고장 -> 2월 창 안에서는 확인되지 않는다
            "FAIL_MAR": {
                "start": date(2020, 1, 1),
                "end": date(2020, 3, 5),
                "failure_date": date(2020, 3, 5),
            },
            # 2월 20일 고장 -> 2월 창 안에서 확인된다
            "FAIL_FEB": {
                "start": date(2020, 1, 1),
                "end": date(2020, 2, 20),
                "failure_date": date(2020, 2, 20),
            },
            # 끝까지 생존. 월 단위 fold 를 여러 개 만들려면 기간이 길어야 한다.
            "ALIVE": {"start": date(2020, 1, 1), "end": date(2020, 12, 31)},
            # 달마다 양성이 있어야 세 구간 모두 검사를 통과하는 fold 가 생긴다.
            **{
                f"FAIL_M{month:02d}": {
                    "start": date(2020, 1, 1),
                    "end": date(2020, month, 15),
                    "failure_date": date(2020, month, 15),
                }
                for month in range(4, 12)
            },
        },
        smart_columns=("smart_5_raw", "smart_197_raw"),
    )
    features_cfg = cfg_mod.load_yaml(paths.CONFIG_DIR / "features.yaml")
    features_cfg["base"]["critical_smart"] = ["smart_5_raw", "smart_197_raw"]
    features_path = features_build.build(
        "TEST_DRIVE", canonical, features_cfg, preprocessing_cfg
    )
    labels_path = horizon_labels.build(
        "TEST_DRIVE", canonical, LABELING, preprocessing_cfg
    )
    return features_path, labels_path


FEB = (date(2020, 2, 1), date(2020, 2, 29))


def _load(built, scope):
    features_path, labels_path = built
    return fold_mod.load_tabular(
        features_path,
        labels_path,
        *FEB,
        horizon_days=HORIZON,
        censoring_scope=scope,
        threads=2,
    )


def test_window_scope_drops_the_unconfirmable_tail(built):
    """창 끝 10일은 창 안에서 고장이 확인되지 않으면 표본이 아니다."""
    alive_rows = {}
    for scope in ("global", "window"):
        matrix = _load(built, scope)
        keep = matrix.serial == "ALIVE"
        alive_rows[scope] = matrix.record_date[keep].max()

    # 전역 기준은 3월 데이터로 2월 말까지 생존을 확인할 수 있다.
    assert alive_rows["global"] == np.datetime64("2020-02-29")
    # 창 기준은 2월 19일까지만 (2/19 + 10일 = 2/29 <= 창 끝).
    assert alive_rows["window"] == np.datetime64("2020-02-19")


def test_window_scope_hides_a_failure_that_lands_after_the_window(built):
    """3월 5일 고장은 2월 창에서 양성 라벨을 만들지 않는다."""
    global_matrix = _load(built, "global")
    window_matrix = _load(built, "window")

    def positives(matrix, serial):
        keep = (matrix.serial == serial) & (matrix.y == 1)
        return int(keep.sum())

    # 전역 기준: 2/24~2/29 행이 양성이 된다.
    assert positives(global_matrix, "FAIL_MAR") > 0
    # 창 기준: 3월 고장을 2월 안에서는 모른다.
    assert positives(window_matrix, "FAIL_MAR") == 0


def test_window_scope_keeps_a_failure_inside_the_window(built):
    """2월 20일 고장은 두 기준 모두에서 양성이다."""
    for scope in ("global", "window"):
        matrix = _load(built, scope)
        keep = (matrix.serial == "FAIL_FEB") & (matrix.y == 1)
        assert int(keep.sum()) > 0, scope


def test_unknown_scope_is_rejected(built):
    with pytest.raises(ValueError, match="censoring_scope"):
        _load(built, "nope")


# --------------------------------------------------------------------------
# warm start
# --------------------------------------------------------------------------
def _sequence_fold(n: int, seed: int):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, 4)).astype(np.float32)
    end_index = np.arange(4, n)
    return fold_mod.SequenceFold(
        matrix=matrix,
        end_index=end_index,
        lookback=5,
        y=(matrix[end_index, 0] > 0.8).astype(np.int8),
        record_date=end_index.astype("datetime64[D]"),
        columns=[f"f{i}" for i in range(4)],
    )


SEQ_CFG = {
    "name": "gru",
    "family": "sequence",
    "class": "hddpred.models.sequence.RNNModel",
    "params": {"cell": "gru", "hidden_size": 8, "num_layers": 1},
    "training": {"epochs": 1, "batch_size": 64, "amp": False},
}


def test_sequence_model_starts_from_the_previous_weights():
    import torch

    train, val = _sequence_fold(400, 0), _sequence_fold(200, 1)
    first = registry.create(SEQ_CFG, seed=42)
    first.fit(train, val)

    second = registry.create(SEQ_CFG, seed=42)
    assert second.warm_start(first) is True
    second.fit(_sequence_fold(400, 2), val)
    assert second.fit_info["warm_started"] is True

    # 초기값이 이어졌으므로 첫 모델과 같은 구조여야 한다.
    for key, value in first.net.state_dict().items():
        assert second.net.state_dict()[key].shape == value.shape
    assert isinstance(second.net, torch.nn.Module)


def test_warm_start_without_a_previous_model_is_a_cold_start():
    train, val = _sequence_fold(400, 0), _sequence_fold(200, 1)
    model = registry.create(SEQ_CFG, seed=42)
    assert model.warm_start(None) is False
    model.fit(train, val)
    assert model.fit_info["warm_started"] is False


def test_tree_models_report_that_they_do_not_warm_start():
    tree_cfg = {
        "name": "lightgbm",
        "family": "tabular",
        "class": "hddpred.models.trees.LightGBMModel",
        "params": {"n_estimators": 5, "verbose": -1},
        "training": {},
    }
    model = registry.create(tree_cfg, seed=42)
    assert model.warm_start(object()) is False


@pytest.mark.parametrize("scope", ["global", "window"])
def test_split_stats_match_what_actually_loads(built, scope):
    """manifest 의 통계가 실제 적재량과 같아야 한다.

    이게 어긋나면 "train 양성 7" 이라고 적힌 fold 가 실제로는 양성 0개로
    실려서 라이브러리 내부에서 알아보기 힘든 오류가 난다. 실제로 그렇게
    터졌던 자리다.
    """
    import duckdb

    _, labels_path = built
    con = duckdb.connect(database=":memory:")
    rows, positives, _, _ = forward._window_stats(
        con,
        horizon_labels.dataset_glob(labels_path),
        *FEB,
        horizon_days=HORIZON,
        censoring_scope=scope,
    )
    con.close()

    matrix = _load(built, scope)
    assert len(matrix) == rows
    assert int(matrix.y.sum()) == positives


def test_window_scope_never_reports_more_positives_than_global(built):
    import duckdb

    _, labels_path = built
    con = duckdb.connect(database=":memory:")
    source = horizon_labels.dataset_glob(labels_path)
    counts = {
        scope: forward._window_stats(
            con, source, *FEB, horizon_days=HORIZON, censoring_scope=scope
        )
        for scope in ("global", "window")
    }
    con.close()
    assert counts["window"][0] < counts["global"][0]  # 행 수
    assert counts["window"][1] <= counts["global"][1]  # 양성 수


def test_a_month_without_positives_is_skipped(built, tmp_path, monkeypatch):
    """세 구간 중 하나라도 양성이 0이면 그 달은 fold 가 되지 않는다."""
    _, labels_path = built
    monkeypatch.setattr(paths, "SPLITS_ROOT", tmp_path / "splits")

    splits_path = forward.build(
        "TEST_DRIVE",
        labels_path,
        dict(MONTHLY_CFG, n_folds=None),
        dict(LABELING, censoring_scope="global"),
    )
    folds = forward.load_folds(splits_path)
    assert folds
    for fold in folds:
        assert fold.train_positives >= 1, fold.test_month
        assert fold.val_positives >= 1, fold.test_month
        assert fold.test_positives >= 1, fold.test_month


# --------------------------------------------------------------------------
# 확장 창
# --------------------------------------------------------------------------
EXPANDING_CFG = dict(MONTHLY_CFG, train_window="expanding")


def test_expanding_train_window_is_anchored_and_grows_a_month_at_a_time():
    folds = forward.plan_folds(MONTHS, EXPANDING_CFG, embargo_days=0)
    assert len(folds) > 1
    # 시작은 데이터 시작에 고정된다.
    assert {f.window("train")[0] for f in folds} == {_month_start_of(MONTHS[0])}
    ends = [f.window("train")[1] for f in folds]
    assert ends == sorted(ends)
    for earlier, later in zip(ends, ends[1:]):
        assert 28 <= (later - earlier).days <= 31


def _month_start_of(month: str) -> date:
    return date.fromisoformat(f"{month}-01")


# --------------------------------------------------------------------------
# 학습 표본 구성 (match_val)
#
#   [과거 양성 전량] + [직전 1개월 전량] + [그 이전 음성 일부]
#   표집률은 학습 표본의 양성 비율을 val 창의 양성 비율에 맞춰 정한다.
# --------------------------------------------------------------------------
TRAIN_WINDOW = (date(2020, 1, 1), date(2020, 6, 30))
RECENT_START = np.datetime64("2020-06-01")


def _load_train(built, *, strategy="none", target=None, seed=42):
    features_path, labels_path = built
    return fold_mod.load_tabular(
        features_path,
        labels_path,
        *TRAIN_WINDOW,
        sampling_cfg={
            "strategy": strategy,
            "keep_recent_months": 1,
            "seed": seed,
        },
        target_positive_rate=target,
        horizon_days=HORIZON,
        censoring_scope="global",
        threads=2,
    )


def test_negative_budget_math():
    # 양성 10, 목표 10% -> 음성 90개 필요. 직전 창이 40개니 과거에서 50개.
    assert fold_mod._negative_budget(10, 40, 100, 0.10) == (50, pytest.approx(0.5))
    # 직전 창만으로 이미 넘치면 과거는 한 행도 안 쓴다.
    assert fold_mod._negative_budget(10, 200, 100, 0.10) == (0, 0.0)
    # 보유량이 모자라면 전부 남긴다 (목표에 못 닿는다).
    assert fold_mod._negative_budget(10, 0, 10, 0.10) == (10, 1.0)


def test_match_val_requires_a_target_rate(built):
    """평가 구간에는 표본 추출을 안 하므로 목표가 없으면 성립하지 않는다."""
    with pytest.raises(ValueError, match="양성 비율"):
        _load_train(built, strategy="match_val", target=None)


def test_match_val_keeps_every_positive_and_the_whole_recent_month(built):
    full = _load_train(built)
    sampled = _load_train(built, strategy="match_val", target=0.10)

    def recent_rows(matrix):
        return int((matrix.record_date >= RECENT_START).sum())

    assert int(full.y.sum()) > 0
    assert int(sampled.y.sum()) == int(full.y.sum())  # 양성은 한 행도 안 버린다
    assert recent_rows(sampled) == recent_rows(full)  # 직전 1개월은 통째로
    assert len(sampled) < len(full)  # 과거 음성만 줄었다


def test_match_val_moves_the_rate_toward_the_target(built):
    full = _load_train(built)
    sampled = _load_train(built, strategy="match_val", target=0.10)
    assert sampled.positive_rate > full.positive_rate
    stats = sampled.sampling
    assert stats["strategy"] == "match_val"
    # 직전 1개월이 통째로 들어가 목표를 넘길 수도, 보유 음성이 모자라 못 닿을
    # 수도 있다. 그 사이에서는 정확히 맞아야 한다.
    if 0.0 < stats["keep_fraction"] < 1.0:
        assert sampled.positive_rate == pytest.approx(0.10, rel=0.05)


def test_match_val_keeps_all_negatives_when_the_target_is_unreachable(built):
    """train 이 이미 목표보다 양성 희박하면 줄일 근거가 없다."""
    full = _load_train(built)
    sampled = _load_train(built, strategy="match_val", target=1e-9)
    assert len(sampled) == len(full)
    assert sampled.sampling["keep_fraction"] == 1.0


def test_match_val_is_deterministic(built):
    first = _load_train(built, strategy="match_val", target=0.10)
    second = _load_train(built, strategy="match_val", target=0.10)
    assert np.array_equal(first.record_date, second.record_date)
    assert np.array_equal(first.serial, second.serial)


def test_match_val_seed_changes_which_negatives_survive(built):
    first = _load_train(built, strategy="match_val", target=0.10, seed=1)
    second = _load_train(built, strategy="match_val", target=0.10, seed=2)
    assert int(first.y.sum()) == int(second.y.sum())
    assert not np.array_equal(first.record_date, second.record_date)


def test_ratio_keeps_all_positives_and_hits_the_target_ratio(built):
    """고전적 언더샘플링 — 양성 전량 + 음성 (ratio x 양성) 개."""
    features_path, labels_path = built
    full = _load_train(built)
    sampled = fold_mod.load_tabular(
        features_path, labels_path, *TRAIN_WINDOW,
        sampling_cfg={"strategy": "ratio", "negative_ratio": 3, "seed": 42},
        horizon_days=HORIZON, censoring_scope="global", threads=2,
    )
    stats = sampled.sampling
    assert stats["strategy"] == "ratio"
    assert int(sampled.y.sum()) == int(full.y.sum()) > 0   # 양성 전량
    assert len(sampled) < len(full)
    if stats["keep_fraction"] < 1.0:
        negatives = len(sampled) - int(sampled.y.sum())
        assert negatives == pytest.approx(3 * int(sampled.y.sum()), rel=0.15)


def test_ratio_keeps_everything_when_negatives_are_scarce(built):
    full = _load_train(built)
    sampled = fold_mod.load_tabular(
        *built, *TRAIN_WINDOW,
        sampling_cfg={"strategy": "ratio", "negative_ratio": 10_000, "seed": 42},
        horizon_days=HORIZON, censoring_scope="global", threads=2,
    )
    assert len(sampled) == len(full)
    assert sampled.sampling["keep_fraction"] == 1.0


def test_match_val_applies_to_sequence_samples_too(built):
    """행렬은 lookback 재료라 그대로 두고 채점 대상 인덱스만 추린다."""
    features_path, labels_path = built
    common = {
        "lookback": 5,
        "horizon_days": HORIZON,
        "censoring_scope": "global",
        "threads": 2,
    }
    full = fold_mod.load_sequence(features_path, labels_path, *TRAIN_WINDOW, **common)
    sampled = fold_mod.load_sequence(
        features_path,
        labels_path,
        *TRAIN_WINDOW,
        sampling_cfg={"strategy": "match_val", "keep_recent_months": 1, "seed": 42},
        target_positive_rate=0.10,
        **common,
    )
    assert int(sampled.y.sum()) == int(full.y.sum())
    assert len(sampled) < len(full)
    assert sampled.matrix.shape == full.matrix.shape


def test_censoring_scope_changes_the_split_hash():
    """fold 집합이 라벨 기준에 따라 달라지므로 hash 도 달라야 한다."""
    base = dict(MONTHLY_CFG, embargo_days=0)
    hashes = {
        forward.split_hash(dict(base, label_censoring_scope=scope), "labelhash")
        for scope in ("global", "window")
    }
    assert len(hashes) == 2


def test_monthly_experiment_config_is_consistent():
    cfg = cfg_mod.load_yaml(paths.CONFIG_DIR / "experiments" / "monthly_expanding.yaml")
    overrides = cfg["overrides"]
    # 가중치를 이어받지 않는다. 매달 새 모델이다.
    assert cfg["warm_start"] is False
    assert overrides["split.train_window"] == "expanding"
    assert overrides["split.val_months"] == 1
    assert overrides["split.test_months"] == 1
    assert overrides["split.embargo_days"] == 0
    assert overrides["split.allow_label_overlap"] is True
    assert overrides["labeling.horizon_days"] == 10
    assert overrides["labeling.censoring_scope"] == "window"
    assert overrides["evaluation.disk_level.rule"] == "in_horizon"
    assert overrides["features.train_sampling.strategy"] == "match_val"
    assert overrides["features.train_sampling.keep_recent_months"] == 1
    # 세 구간 모두 양성 하한이 걸려 있어야 "양성 0인 달은 건너뛴다"가 성립한다.
    for key in (
        "split.min_train_failures",
        "split.min_val_failures",
        "split.min_test_failures",
    ):
        assert overrides[key] >= 1
