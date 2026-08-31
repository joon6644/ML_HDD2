"""실험 실행기.

run 하나는 (drive, model, seed, fold) 조합이고, 다음을 남긴다.

    runs/<experiment>/<drive>/<model>/seed<N>/fold<NN>/
    ├─ resolved_config.yaml     해석이 끝난 설정 전체
    ├─ model/                   체크포인트 + model_meta.json
    ├─ threshold.json           validation 에서 고른 임곗값과 근거
    ├─ predictions_val.parquet  임곗값을 다시 고를 때 재학습이 필요 없게
    ├─ predictions_test.parquet
    └─ fold_metrics.json        fold 정보 + val/test 지표 + 신뢰구간

파이프라인 단계는 서로를 직접 호출하지 않는다. 이 모듈만이 순서를 안다.

    prepare_drive()  raw -> canonical -> features / labels -> split manifest
    run_fold()       fold 데이터 -> 학습 -> 임곗값 -> 예측 -> 평가
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config as cfg_mod
from .. import paths
from ..data import canonicalize
from ..evaluation import metrics as metrics_mod
from ..features import build as features_build
from ..features import fold as fold_mod
from ..inference import threshold as threshold_mod
from ..labeling import horizon_labels
from ..models import registry
from ..splits import forward


@dataclass
class Pipeline:
    data: dict
    preprocessing: dict
    labeling: dict
    split: dict
    features: dict
    evaluation: dict

    @classmethod
    def from_experiment(cls, experiment_cfg: dict) -> "Pipeline":
        uses = experiment_cfg["uses"]
        loaded = {key: cfg_mod.load_yaml(path) for key, path in uses.items()}
        overrides = experiment_cfg.get("overrides") or {}
        for dotted, value in overrides.items():
            section, _, rest = dotted.partition(".")
            if section in loaded and rest:
                cfg_mod.set_dotted(loaded[section], rest, value)
        return cls(
            data=loaded["data"],
            preprocessing=loaded["preprocessing"],
            labeling=loaded["labeling"],
            split=loaded["split"],
            features=loaded["features"],
            evaluation=loaded["evaluation"],
        )

    def as_dict(self) -> dict:
        return {
            "data": self.data,
            "preprocessing": self.preprocessing,
            "labeling": self.labeling,
            "split": self.split,
            "features": self.features,
            "evaluation": self.evaluation,
        }


@dataclass
class PreparedDrive:
    name: str
    canonical_path: Path
    labels_path: Path
    features_path: Path
    splits_path: Path
    folds: list[forward.Fold]


def prepare_drive(
    drive_name: str, pipeline: Pipeline, *, force: bool = False
) -> PreparedDrive:
    """드라이브 하나의 파생 레이어를 전부 갖춰 둔다. 이미 있으면 재사용한다."""
    drive = cfg_mod.find_drive(pipeline.data, drive_name)

    canonical_path = canonicalize.build(
        drive, pipeline.data, pipeline.preprocessing, force=force
    )
    features_path = features_build.build(
        drive_name, canonical_path, pipeline.features, pipeline.preprocessing, force=force
    )
    labels_path = horizon_labels.build(
        drive_name, canonical_path, pipeline.labeling, pipeline.preprocessing, force=force
    )
    splits_path = forward.build(
        drive_name, labels_path, pipeline.split, pipeline.labeling, force=force
    )
    return PreparedDrive(
        name=drive_name,
        canonical_path=canonical_path,
        labels_path=labels_path,
        features_path=features_path,
        splits_path=splits_path,
        folds=forward.load_folds(splits_path),
    )


def _load_part(
    prepared: PreparedDrive,
    fold: forward.Fold,
    part: str,
    family: str,
    features_cfg: dict,
    labeling_cfg: dict,
    threads: int,
):
    start, end = fold.window(part)
    is_train = part == "train"
    sampling = features_cfg.get("train_sampling") if is_train else None
    # match_val 은 "학습 표본의 불균형을 val 창의 불균형에 맞춘다"는 규칙이라
    # 목표 비율이 필요하다. manifest 통계는 적재 기준과 같은 값이다.
    target_rate = None
    if is_train and (sampling or {}).get("strategy") == "match_val":
        target_rate = fold.val_positives / fold.val_rows if fold.val_rows else 0.0
    # train 은 채점하지 않으므로 serial / failure_date 를 읽지 않는다.
    # 24M행에서 문자열 배열 하나가 수 GB다.
    common = {
        "sampling_cfg": sampling,
        "target_positive_rate": target_rate,
        "with_meta": not is_train,
        "horizon_days": int(labeling_cfg["horizon_days"]),
        "censoring_scope": labeling_cfg.get("censoring_scope", "global"),
        "threads": threads,
    }
    if family == "tabular":
        return fold_mod.load_tabular(
            prepared.features_path, prepared.labels_path, start, end, **common
        )
    return fold_mod.load_sequence(
        prepared.features_path,
        prepared.labels_path,
        start,
        end,
        lookback=int(features_cfg["sequence"]["lookback_days"]),
        short_history=features_cfg["sequence"].get("short_history", "pad"),
        **common,
    )


def run_fold(
    prepared: PreparedDrive,
    fold: forward.Fold,
    model_cfg: dict,
    seed: int,
    pipeline: Pipeline,
    experiment: str,
    *,
    force: bool = False,
    previous_model=None,
) -> tuple[Path, object | None]:
    """fold 하나를 학습하고 평가한다. (결과 디렉터리, 학습된 모델) 을 돌려준다.

    previous_model 이 있으면 warm start 를 시도한다. 모델이 지원하지 않으면
    새로 학습하고 그 사실을 fit_info 에 남긴다.
    """
    out_dir = paths.run_dir(experiment, prepared.name, model_cfg["name"], seed, fold.fold)
    if (out_dir / "fold_metrics.json").exists() and not force:
        print(f"  [run] 재사용: {out_dir}")
        # warm start 사슬을 이어야 하므로 저장된 모델을 되살린다.
        restored = None
        if previous_model is not None or (out_dir / "model").exists():
            try:
                restored = registry.resolve_class(model_cfg["class"]).load(out_dir / "model")
            except Exception as error:  # noqa: BLE001
                print(f"    [warm start] 저장된 모델을 못 읽었다: {error}")
        return out_dir, restored
    out_dir.mkdir(parents=True, exist_ok=True)

    family = model_cfg["family"]
    threads = int(pipeline.preprocessing.get("duckdb", {}).get("threads", 8))
    print(
        f"\n=== {prepared.name} | {model_cfg['name']} | seed {seed} | "
        f"fold {fold.fold:02d} (test {fold.test_month}) ==="
    )

    n_columns = len(
        features_build.tabular_columns(prepared.features_path)
        if family == "tabular"
        else features_build.sequence_columns(prepared.features_path)
    )
    sampling_cfg = pipeline.features.get("train_sampling") or {}
    stride = fold_mod.stride_from(sampling_cfg)
    estimated_rows = fold.train_rows if not stride else fold.train_rows // stride
    note = ""
    if stride:
        note = f" (stride {stride} 적용 후 근사)"
    elif sampling_cfg.get("strategy") == "match_val":
        note = " (음성 표집 전 상한)"
    print(
        f"  train 행렬 예상 "
        f"{fold_mod.estimate_memory_gb(estimated_rows, n_columns):.1f} GB "
        f"({estimated_rows:,}행 x {n_columns}피처){note}. 부족하면 "
        f"features.yaml 의 train_sampling.strategy 를 stride 로 두어라."
    )

    loaded = time.time()
    parts = (prepared, fold)
    shared = (family, pipeline.features, pipeline.labeling, threads)
    train = _load_part(*parts, "train", *shared)
    val = _load_part(*parts, "val", *shared)
    test = _load_part(*parts, "test", *shared)
    print(
        f"  train {len(train):,} (양성 {train.positive_rate:.4%}) | "
        f"val {len(val):,} | test {len(test):,} "
        f"({time.time() - loaded:.1f}s)"
    )
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(f"fold {fold.fold}: 비어 있는 구간이 있습니다.")
    # 한쪽 클래스만 있으면 라이브러리 내부에서 알아보기 힘든 오류가 난다.
    # split manifest 의 통계와 실제 적재가 어긋난 경우이므로 여기서 잡는다.
    for name, part in (("train", train), ("val", val)):
        classes = np.unique(part.y)
        if classes.size < 2:
            raise ValueError(
                f"fold {fold.fold} ({fold.test_month}): {name} 에 클래스가 "
                f"{classes.tolist()} 뿐이다. split manifest 의 통계와 적재 기준이 "
                "어긋났는지 확인하라 (labeling.censoring_scope)."
            )

    # 스케일러는 train 구간에서만 적합시킨다. 트리 모델은 스케일에 불변이라
    # 적용하지 않는다.
    scaler_state = None
    if family == "sequence":
        scaling = pipeline.features.get("scaling", {})
        scaler = fold_mod.Scaler(
            scaling.get("method", "standard"), scaling.get("clip_quantile")
        ).fit(train.matrix)
        for part in (train, val, test):
            part.matrix = scaler.transform(part.matrix, fill_nan=True)
        scaler_state = scaler.state_dict()

    model = registry.create(model_cfg, seed)
    warm_requested = previous_model is not None
    warm_accepted = model.warm_start(previous_model) if warm_requested else False
    if warm_requested and not warm_accepted:
        print(f"    [warm start] {model_cfg['name']} 은 지원하지 않아 새로 학습한다.")

    started = time.time()
    fit_info = model.fit(train, val)
    fit_seconds = time.time() - started
    fit_info.setdefault("warm_started", warm_accepted)

    val_score = model.predict_proba(val)
    test_score = model.predict_proba(test)

    val_frame = fold_mod.to_frame(val, val_score)
    test_frame = fold_mod.to_frame(test, test_score)

    horizon = int(pipeline.labeling["horizon_days"])
    val_spec = metrics_mod.DiskLevelSpec.from_config(
        pipeline.evaluation, horizon_days=horizon, period=fold.window("val")
    )

    def _val_disk_far(candidate: float) -> float:
        """validation 의 디스크 단위 FAR. fixed_disk_far 정책이 호출한다."""
        units = metrics_mod.collapse_to_disks(val_frame, candidate, val_spec)
        return metrics_mod.disk_metrics(units, val_spec)["far"]

    threshold_record = threshold_mod.select(
        val_frame["y"].to_numpy(),
        val_frame["score"].to_numpy(),
        pipeline.evaluation["threshold"],
        disk_far_fn=_val_disk_far,
    )
    chosen = threshold_record["threshold"]

    result = {
        "val": metrics_mod.evaluate(
            val_frame,
            chosen,
            pipeline.evaluation,
            horizon_days=horizon,
            period=fold.window("val"),
        ),
        "test": metrics_mod.evaluate(
            test_frame,
            chosen,
            pipeline.evaluation,
            horizon_days=horizon,
            period=fold.window("test"),
        ),
    }
    ci = metrics_mod.bootstrap_ci(
        test_frame,
        chosen,
        pipeline.evaluation,
        pipeline.evaluation.get("bootstrap", {}),
        horizon_days=horizon,
        period=fold.window("test"),
    )

    # --- artifact 저장 ---------------------------------------------------
    model.save(out_dir / "model")
    threshold_mod.save(threshold_record, out_dir)
    val_frame.to_parquet(out_dir / "predictions_val.parquet", index=False)
    test_frame.to_parquet(out_dir / "predictions_test.parquet", index=False)
    cfg_mod.dump_resolved(
        {"experiment": experiment, "model": model_cfg, "seed": seed, **pipeline.as_dict()},
        out_dir / "resolved_config.yaml",
    )

    record = {
        "experiment": experiment,
        "drive": prepared.name,
        "model": model_cfg["name"],
        "family": family,
        "seed": seed,
        "fold": {
            "fold": fold.fold,
            "test_month": fold.test_month,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "val_start": fold.val_start,
            "val_end": fold.val_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
        },
        "hashes": {
            "canonical": canonicalize.canonical_hash(
                pipeline.preprocessing, cfg_mod.find_drive(pipeline.data, prepared.name)
            ),
            "labels": horizon_labels.label_hash(
                pipeline.labeling,
                canonicalize.canonical_hash(
                    pipeline.preprocessing,
                    cfg_mod.find_drive(pipeline.data, prepared.name),
                ),
            ),
            "splits": forward.load_manifest(prepared.splits_path)["split_hash"],
        },
        "threshold": threshold_record,
        "metrics": result,
        "bootstrap_ci": ci,
        "fit_info": fit_info,
        "sampling": train.sampling,
        "complexity": model.complexity(),
        "scaler": scaler_state,
        "timing": {"fit_seconds": round(fit_seconds, 1)},
    }
    with (out_dir / "fold_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False, default=str)

    row = result["test"]["row_level"]
    disk = result["test"]["disk_level"]
    print(
        f"  test  row: PR-AUC {row['pr_auc']:.4f} P {row['precision']:.3f} "
        f"R {row['recall']:.3f} | disk: P {disk['precision']:.3f} "
        f"R {disk['recall']:.3f} FAR {disk['far']:.4f} "
        f"(fit {fit_seconds:.1f}s)"
    )
    return out_dir, model


def run_experiment(experiment_path: str | Path, *, force: bool = False) -> Path:
    """실험 설정 하나를 끝까지 실행하고 결과 디렉터리를 돌려준다."""
    experiment_cfg = cfg_mod.load_yaml(experiment_path)
    experiment = experiment_cfg["experiment"]
    pipeline = Pipeline.from_experiment(experiment_cfg)

    model_configs = [
        registry.load_model_config(path) for path in experiment_cfg["models"]
    ]
    for model_cfg in model_configs:
        override_key = f"models.{model_cfg['name']}."
        for dotted, value in (experiment_cfg.get("overrides") or {}).items():
            if dotted.startswith(override_key):
                cfg_mod.set_dotted(model_cfg, dotted[len(override_key) :], value)

    # 월 단위 파인튜닝: 직전 fold 의 가중치를 다음 fold 의 초기값으로 넘긴다.
    # fold 를 시간 순서대로 돌아야 의미가 있으므로 순서를 강제한다.
    warm_start = bool(experiment_cfg.get("warm_start", False))

    fold_filter = experiment_cfg.get("folds")
    for drive_name in experiment_cfg["drives"]:
        prepared = prepare_drive(drive_name, pipeline, force=force)
        folds = sorted(prepared.folds, key=lambda f: f.fold)
        if fold_filter is not None:
            wanted = set(fold_filter)
            folds = [f for f in folds if f.fold in wanted]

        for model_cfg in model_configs:
            for seed in experiment_cfg["seeds"]:
                previous = None
                for fold in folds:
                    _, fitted = run_fold(
                        prepared,
                        fold,
                        model_cfg,
                        int(seed),
                        pipeline,
                        experiment,
                        force=force,
                        previous_model=previous if warm_start else None,
                    )
                    if warm_start:
                        previous = fitted

    return paths.RUNS_ROOT / experiment
