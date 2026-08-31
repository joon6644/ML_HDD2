"""완료된 run 디렉터리를 MLflow 로 적재한다.

파이프라인 자체는 MLflow 에 의존하지 않는다. run 디렉터리(fold_metrics.json,
threshold.json, resolved_config.yaml, model/)가 진실의 원본이고 MLflow 는 그
위에 얹는 색인이다. 그래서:

  - MLflow 가 없어도 실험은 돌아간다.
  - mlruns 를 지워도 run 디렉터리에서 다시 만들 수 있다.
  - 같은 run 을 두 번 적재하지 않는다 (run_dir 태그로 확인).

저장 위치는 project_B/mlflow.db (SQLite) + project_B/mlartifacts 다. 서버가
필요 없다. MLflow 3.x 가 파일 스토어를 유지보수 모드로 돌렸으므로 SQLite 를 쓴다.

    python scripts/export_to_mlflow.py
    mlflow ui --backend-store-uri sqlite:///.../project_B/mlflow.db
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import paths

RUN_DIR_TAG = "hddpred.run_dir"
# 중복 판정 키. 경로만 쓰면 같은 자리에 다시 만든 run 이 적재되지 않는다.
# 경로 + fold_metrics.json 내용 해시라, 재실행 결과는 새 MLflow run 이 되고
# 같은 결과를 두 번 올리는 일은 없다.
RUN_KEY_TAG = "hddpred.run_key"
METRICS_FILE = "fold_metrics.json"

# 아티팩트로 올릴 작은 파일들. 예측 parquet 은 무거워서 기본 제외한다.
SMALL_ARTIFACTS = ("fold_metrics.json", "threshold.json", "resolved_config.yaml")
PREDICTION_ARTIFACTS = ("predictions_val.parquet", "predictions_test.parquet")


def tracking_uri() -> str:
    """SQLite 백엔드. MLflow 3.x 는 파일 스토어를 더 이상 갱신하지 않는다."""
    return f"sqlite:///{(paths.PROJECT_ROOT / 'mlflow.db').as_posix()}"


def artifact_root() -> str:
    root = paths.PROJECT_ROOT / "mlartifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root.as_uri()


def _ensure_experiment(name: str) -> str:
    """실험이 없으면 artifact 위치를 지정해서 만든다."""
    import mlflow

    client = mlflow.tracking.MlflowClient()
    found = client.get_experiment_by_name(name)
    if found is None:
        client.create_experiment(name, artifact_location=artifact_root())
    mlflow.set_experiment(name)
    return name


def _flatten(prefix: str, node: Any, out: dict) -> dict:
    """중첩 dict 을 점 표기 스칼라로 편다. 리스트는 문자열로 접는다."""
    if isinstance(node, dict):
        for key, value in node.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), value, out)
    elif isinstance(node, list):
        out[prefix] = json.dumps(node, ensure_ascii=False)[:480]
    else:
        out[prefix] = node
    return out


def _params(record: dict) -> dict:
    """재현에 필요한 설정. MLflow 파라미터는 문자열이라 그대로 눌러 담는다."""
    config = record.get("resolved", {})
    keep = {
        "drive": record["drive"],
        "model": record["model"],
        "family": record.get("family"),
        "seed": record["seed"],
        "fold": record["fold"]["fold"],
        "test_month": record["fold"]["test_month"],
        "train_start": record["fold"]["train_start"],
        "train_end": record["fold"]["train_end"],
        "val_start": record["fold"]["val_start"],
        "val_end": record["fold"]["val_end"],
        "threshold_policy": record["threshold"]["policy"],
    }
    for section in ("labeling", "split", "features", "evaluation"):
        if section in config:
            _flatten(section, config[section], keep)
    if "model" in config:
        _flatten("model_cfg", config["model"], keep)
    # 값이 너무 길면 MLflow 가 거부한다.
    return {k: str(v)[:500] for k, v in keep.items() if v is not None}


def _metrics(record: dict) -> dict:
    out: dict[str, float] = {}
    for split in ("val", "test"):
        for layer in ("row_level", "disk_level"):
            for key, value in record["metrics"][split][layer].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    out[f"{split}.{layer}.{key}"] = float(value)
    for key, band in (record.get("bootstrap_ci") or {}).items():
        out[f"ci.{key}.lower"] = float(band["lower"])
        out[f"ci.{key}.upper"] = float(band["upper"])
    for key, value in (record.get("fit_info") or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[f"fit.{key}"] = float(value)
    if record.get("timing", {}).get("fit_seconds") is not None:
        out["fit.seconds"] = float(record["timing"]["fit_seconds"])
    if record.get("complexity") is not None:
        out["complexity"] = float(record["complexity"])
    # NaN 은 MLflow 에서 비교가 안 되므로 버린다.
    return {k: v for k, v in out.items() if v == v and abs(v) != float("inf")}


def run_key(run_dir: Path) -> str:
    """경로 + 결과 내용 해시. 재실행하면 값이 바뀐다."""
    import hashlib

    blob = (run_dir / METRICS_FILE).read_bytes()
    digest = hashlib.sha256(blob).hexdigest()[:10]
    return f"{run_dir.relative_to(paths.PROJECT_ROOT).as_posix()}@{digest}"


def _tags(record: dict, run_dir: Path) -> dict:
    tags = {
        RUN_DIR_TAG: run_dir.relative_to(paths.PROJECT_ROOT).as_posix(),
        RUN_KEY_TAG: run_key(run_dir),
        "hddpred.experiment": record["experiment"],
        "hddpred.family": str(record.get("family")),
    }
    for name, value in (record.get("hashes") or {}).items():
        tags[f"hddpred.hash.{name}"] = str(value)
    if record.get("fit_info", {}).get("loss"):
        tags["hddpred.loss"] = str(record["fit_info"]["loss"])
    return tags


def load_run(run_dir: Path) -> dict:
    with (run_dir / METRICS_FILE).open("r", encoding="utf-8") as fh:
        record = json.load(fh)
    resolved = run_dir / "resolved_config.yaml"
    if resolved.exists():
        import yaml

        with resolved.open("r", encoding="utf-8") as fh:
            record["resolved"] = yaml.safe_load(fh)
    return record


def existing_run_keys(experiment: str) -> set[str]:
    """이미 적재된 run_key 집합."""
    import mlflow

    client = mlflow.tracking.MlflowClient()
    found = client.get_experiment_by_name(experiment)
    if found is None:
        return set()
    done = set()
    for run in mlflow.search_runs(
        experiment_ids=[found.experiment_id], output_format="list"
    ):
        value = run.data.tags.get(RUN_KEY_TAG)
        if value:
            done.add(value)
    return done


def log_run(run_dir: Path, *, with_predictions: bool = False) -> str:
    """run 디렉터리 하나를 MLflow 에 기록하고 run id 를 돌려준다."""
    import mlflow

    record = load_run(run_dir)
    _ensure_experiment(record["experiment"])
    name = (
        f"{record['drive']}/{record['model']}/seed{record['seed']}"
        f"/fold{record['fold']['fold']:02d}"
    )
    with mlflow.start_run(run_name=name) as active:
        mlflow.set_tags(_tags(record, run_dir))
        mlflow.log_params(_params(record))
        mlflow.log_metrics(_metrics(record))

        for history in (record.get("fit_info") or {}).get("history", []) or []:
            for key, value in history.items():
                if key != "epoch" and isinstance(value, (int, float)):
                    if value == value:  # NaN 제외
                        mlflow.log_metric(f"epoch.{key}", float(value), step=history["epoch"])

        names = SMALL_ARTIFACTS + (PREDICTION_ARTIFACTS if with_predictions else ())
        for filename in names:
            path = run_dir / filename
            if path.exists():
                mlflow.log_artifact(str(path))
        model_dir = run_dir / "model"
        if model_dir.exists():
            mlflow.log_artifacts(str(model_dir), artifact_path="model")
        return active.info.run_id


def export(
    runs_root: Path | None = None,
    experiment: str | None = None,
    *,
    with_predictions: bool = False,
) -> dict:
    """runs/ 아래의 완료된 run 을 전부 적재한다. 이미 있는 것은 건너뛴다."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    runs_root = runs_root or paths.RUNS_ROOT
    pattern = f"{experiment}/**/{METRICS_FILE}" if experiment else f"**/{METRICS_FILE}"

    logged, skipped = [], []
    cache: dict[str, set[str]] = {}
    for metrics_path in sorted(runs_root.glob(pattern)):
        run_dir = metrics_path.parent
        record = load_run(run_dir)
        name = record["experiment"]
        if name not in cache:
            cache[name] = existing_run_keys(name)
        key = run_key(run_dir)
        relative = run_dir.relative_to(paths.PROJECT_ROOT).as_posix()
        if key in cache[name]:
            skipped.append(relative)
            continue
        log_run(run_dir, with_predictions=with_predictions)
        cache[name].add(key)
        logged.append(relative)

    return {"logged": logged, "skipped": skipped, "tracking_uri": tracking_uri()}
