"""설정 로딩과 config hash.

원칙:
  파생 데이터 폴더 이름에 config hash를 넣는다. 설정이 바뀌면 새 폴더가
  생기고 기존 결과는 덮어쓰이지 않는다.

hash는 "그 산출물에 실제로 영향을 주는 설정"만으로 계산한다. 예를 들어
duckdb.threads는 canonical 내용에 영향을 주지 않으므로 hash에서 제외한다.
그렇게 하지 않으면 스레드 수만 바꿔도 27M행을 다시 만들게 된다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from . import paths

# hash 계산에서 제외할 dotted key. 산출물 내용에 영향을 주지 않는 것들.
_HASH_EXCLUDE = {
    "duckdb.threads",
    "duckdb.max_memory",
    "duckdb.temp_dir",
    # 창별로 fold 적재 시점에 적용되는 값이라 라벨 레이어 내용에 영향이 없다.
    # 여기 넣지 않으면 값만 바꿔도 동일한 라벨을 다시 만들게 된다.
    "censoring_scope",
}


def _walk_strings(node: Any) -> Any:
    if isinstance(node, str):
        return paths.resolve_placeholders(node)
    if isinstance(node, dict):
        return {k: _walk_strings(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk_strings(v) for v in node]
    return node


def load_yaml(path: str | Path) -> dict:
    """YAML을 읽고 경로 placeholder를 치환한다."""
    path = Path(path)
    if not path.is_absolute():
        path = paths.PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {path}")
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        raise ValueError(f"설정 파일이 비어 있습니다: {path}")
    if not isinstance(loaded, dict):
        raise TypeError(f"설정 최상위는 매핑이어야 합니다: {path}")
    return _walk_strings(loaded)


def get_dotted(cfg: dict, key: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def apply_overrides(cfg: dict, overrides: dict[str, Any] | None) -> dict:
    """dotted key 오버라이드를 적용한 새 dict를 돌려준다."""
    merged = json.loads(json.dumps(cfg, default=str))
    for key, value in (overrides or {}).items():
        set_dotted(merged, key, value)
    return merged


def _strip_for_hash(cfg: dict, prefix: str = "") -> dict:
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        dotted = f"{prefix}{key}"
        if dotted in _HASH_EXCLUDE:
            continue
        out[key] = (
            _strip_for_hash(value, f"{dotted}.") if isinstance(value, dict) else value
        )
    return out


def config_hash(*configs: dict, length: int = 10) -> str:
    """설정 조합의 안정적인 짧은 hash.

    key 순서와 무관하게 같은 값이면 같은 hash가 나온다.
    """
    payload = [_strip_for_hash(cfg) for cfg in configs]
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def dump_resolved(cfg: dict, path: Path) -> None:
    """해석이 끝난 설정을 run 디렉터리에 스냅샷으로 남긴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)


def enabled_drives(data_cfg: dict) -> list[dict]:
    drives = [d for d in data_cfg["drives"] if d.get("enabled", False)]
    if not drives:
        raise ValueError("configs/data.yaml 에 enabled: true 인 드라이브가 없습니다.")
    return drives


def find_drive(data_cfg: dict, name: str) -> dict:
    for drive in data_cfg["drives"]:
        if drive["name"] == name:
            return drive
    known = ", ".join(d["name"] for d in data_cfg["drives"])
    raise KeyError(f"알 수 없는 드라이브 {name!r}. 사용 가능: {known}")
