"""재현성 검사.

config hash 는 파생 폴더 이름이 된다. 값이 같은데 hash 가 달라지면 27M행을
매번 다시 만들고, 값이 다른데 hash 가 같으면 다른 설정의 결과를 덮어쓴다.
"""

from __future__ import annotations

import pytest

from hddpred import config as cfg_mod
from hddpred.tracking import provenance


def test_hash_is_independent_of_key_order():
    left = {"a": 1, "b": {"x": 2, "y": 3}}
    right = {"b": {"y": 3, "x": 2}, "a": 1}
    assert cfg_mod.config_hash(left) == cfg_mod.config_hash(right)


def test_hash_changes_when_a_value_changes():
    base = {"horizon_days": 10}
    assert cfg_mod.config_hash(base) != cfg_mod.config_hash({"horizon_days": 30})


def test_runtime_only_settings_do_not_change_the_hash():
    """스레드 수를 바꿨다고 canonical 을 다시 만들 이유는 없다."""
    left = {"missing_ratio_threshold": 0.9, "duckdb": {"threads": 8}}
    right = {"missing_ratio_threshold": 0.9, "duckdb": {"threads": 2}}
    assert cfg_mod.config_hash(left) == cfg_mod.config_hash(right)


def test_dotted_overrides_do_not_mutate_the_original():
    base = {"split": {"n_folds": 6}}
    changed = cfg_mod.apply_overrides(base, {"split.n_folds": 1})
    assert base["split"]["n_folds"] == 6
    assert changed["split"]["n_folds"] == 1


def test_incomplete_directory_is_not_reused(tmp_path):
    """_SUCCESS 가 없으면 미완성으로 본다."""
    directory = tmp_path / "canon=abc"
    directory.mkdir()
    assert not provenance.is_complete(directory)

    provenance.write(directory, stage="canonical", config_hash="abc", configs={})
    assert provenance.is_complete(directory)

    provenance.clear(directory)
    assert not provenance.is_complete(directory)


def test_provenance_records_the_parent_hash(tmp_path):
    directory = tmp_path / "label=def"
    provenance.write(
        directory,
        stage="labels",
        config_hash="def",
        configs={"labeling": {"horizon_days": 10}},
        parents={"canonical": "abc"},
    )
    record = provenance.read(directory)
    assert record["parents"]["canonical"] == "abc"
    assert record["stage"] == "labels"
    assert "created_at" in record


def test_all_model_configs_load(tmp_path):
    """configs/models/*.yaml 이 전부 registry 로 해석되는지 확인한다."""
    from hddpred import paths
    from hddpred.models import registry

    files = sorted((paths.CONFIG_DIR / "models").glob("*.yaml"))
    assert files, "모델 설정 파일이 없습니다."
    for path in files:
        model_cfg = registry.load_model_config(path)
        cls = registry.resolve_class(model_cfg["class"])
        assert cls is not None


@pytest.mark.parametrize(
    "name", ["data", "preprocessing", "labeling", "split", "features", "evaluation"]
)
def test_pipeline_configs_load(name):
    from hddpred import paths

    cfg = cfg_mod.load_yaml(paths.CONFIG_DIR / f"{name}.yaml")
    assert isinstance(cfg, dict) and cfg
