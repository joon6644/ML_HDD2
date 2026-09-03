"""학습 표본에만 적용하는 인메모리 재표집.

DuckDB 단의 train_sampling(none/stride/ratio/match_val)은 "음성을 덜 읽는"
쪽이다. 2,500만 행을 통째로 메모리에 올리지 않으려는 장치라 감축만 할 수 있다.

이 모듈은 그 뒤에 온다. 이미 메모리에 올라온 (X, y) 를 받아 합성 표본을
만들거나(SMOTE 계열) 경계를 정리한다(Tomek/ENN). imbalanced-learn 이
scikit-learn 추정기와 같은 규약을 쓰므로 얇게 감싸기만 한다.

    [DuckDB 감축]  ->  [인메모리 재표집]  ->  model.fit

⚠ 평가 구간에는 절대 적용하지 않는다. val/test 의 유병률이 바뀌면 그 위에서
  고른 임곗값도, 보고하는 오탐률도 실제와 달라진다. 호출부에서 train 조각만
  넘긴다.

⚠ 근접이웃을 쓰는 방법(SMOTE/ADASYN/Tomek/ENN)은 표본 수에 대해 초선형이다.
  전 표본(2,500만 행)에 그대로 걸면 끝나지 않는다. 반드시 DuckDB 단에서
  ratio 로 줄인 뒤에 쓴다. 그 전제가 깨지면 max_rows 에서 막는다.
"""

from __future__ import annotations

import time

import numpy as np

# 표집기 이름 -> (모듈, 클래스). 임포트는 실제로 쓸 때만 한다.
_REGISTRY = {
    # 오버샘플링
    "random_over": ("imblearn.over_sampling", "RandomOverSampler"),
    "smote": ("imblearn.over_sampling", "SMOTE"),
    "borderline_smote": ("imblearn.over_sampling", "BorderlineSMOTE"),
    "svm_smote": ("imblearn.over_sampling", "SVMSMOTE"),
    "adasyn": ("imblearn.over_sampling", "ADASYN"),
    # 언더샘플링 (DuckDB 단에서 못 하는 것들 — 이웃 기반 정리)
    "random_under": ("imblearn.under_sampling", "RandomUnderSampler"),
    "tomek": ("imblearn.under_sampling", "TomekLinks"),
    "enn": ("imblearn.under_sampling", "EditedNearestNeighbours"),
    "nearmiss": ("imblearn.under_sampling", "NearMiss"),
    # 하이브리드
    "smote_tomek": ("imblearn.combine", "SMOTETomek"),
    "smote_enn": ("imblearn.combine", "SMOTEENN"),
}

# 이웃 탐색을 하는 방법. 표본 수 상한을 건다.
_NEIGHBOUR_BASED = {
    "smote", "borderline_smote", "svm_smote", "adasyn",
    "tomek", "enn", "nearmiss", "smote_tomek", "smote_enn",
}

DEFAULT_MAX_ROWS = 3_000_000


def _build(method: str, params: dict, seed: int):
    module_name, class_name = _REGISTRY[method]
    module = __import__(module_name, fromlist=[class_name])
    klass = getattr(module, class_name)
    kwargs = dict(params)
    # 재현성. random_state 를 받지 않는 표집기(Tomek/ENN)도 있다.
    if "random_state" in klass().get_params():
        kwargs.setdefault("random_state", seed)
    if "n_jobs" in klass().get_params():
        kwargs.setdefault("n_jobs", 8)
    return klass(**kwargs)


def apply(train, resample_cfg: dict | None, seed: int):
    """train 조각을 재표집한 새 조각으로 바꿔 돌려준다.

    method 가 none 이거나 설정이 없으면 원본을 그대로 준다. 재표집을 하면
    합성 행에는 대응하는 serial / record_date / failure_date 가 없으므로
    그 메타는 버린다 (학습에만 쓰는 조각이라 평가에서 참조되지 않는다).
    """
    cfg = resample_cfg or {}
    method = str(cfg.get("method", "none")).lower()
    if method in ("none", ""):
        return train, {"method": "none"}
    if method not in _REGISTRY:
        raise ValueError(
            f"알 수 없는 resample.method: {method!r} "
            f"(가능: none, {', '.join(sorted(_REGISTRY))})"
        )

    X, y = train.X, train.y
    n_before = len(y)
    pos_before = int((y == 1).sum())

    max_rows = int(cfg.get("max_rows", DEFAULT_MAX_ROWS))
    if method in _NEIGHBOUR_BASED and n_before > max_rows:
        raise ValueError(
            f"{method} 는 이웃 탐색을 하므로 {n_before:,}행에 걸 수 없다 "
            f"(상한 {max_rows:,}). features.train_sampling 을 ratio 로 두어 "
            "먼저 음성을 줄여라. 상한을 바꾸려면 resample.max_rows 를 올려라."
        )

    # imbalanced-learn 은 결측을 허용하지 않는다. 트리 모델은 NaN 을 그대로
    # 처리하지만 이웃 거리 계산에는 값이 있어야 한다. 학습 표본에만 적용되고
    # 평가 표본은 손대지 않으므로 모델이 보는 결측 규약이 달라지는 점을
    # stats 에 남긴다.
    n_nan = int(np.isnan(X).sum())
    if n_nan:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    sampler = _build(method, dict(cfg.get("params") or {}), seed)
    started = time.time()
    X_new, y_new = sampler.fit_resample(X, y)
    elapsed = time.time() - started

    X_new = np.ascontiguousarray(X_new, dtype=X.dtype)
    y_new = np.asarray(y_new, dtype=train.y.dtype)

    resampled = type(train)(
        X=X_new,
        y=y_new,
        record_date=np.zeros(len(y_new), dtype=train.record_date.dtype),
        serial=None,
        failure_date=None,
        columns=list(train.columns),
        sampling=dict(train.sampling),
        window=train.window,
    )
    stats = {
        "method": method,
        "params": dict(cfg.get("params") or {}),
        "rows_before": n_before,
        "rows_after": int(len(y_new)),
        "positives_before": pos_before,
        "positives_after": int((y_new == 1).sum()),
        "positive_rate_before": pos_before / n_before if n_before else 0.0,
        "positive_rate_after": float((y_new == 1).mean()) if len(y_new) else 0.0,
        "nan_filled": n_nan,
        "elapsed_seconds": round(elapsed, 1),
    }
    print(
        f"  [resample] {method}: {n_before:,} -> {stats['rows_after']:,}행, "
        f"양성 {stats['positive_rate_before']:.4%} -> "
        f"{stats['positive_rate_after']:.4%} ({elapsed:.1f}s)"
    )
    return resampled, stats
