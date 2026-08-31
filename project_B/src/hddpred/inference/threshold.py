"""임곗값 선정.

임곗값은 validation 예측에서만 고른다. test 예측은 임곗값 선정에 절대
사용하지 않는다. 이 규칙이 깨지면 test 성능이 낙관적으로 편향된다.

모델 artifact 와 threshold 는 별도로 저장한다. 같은 모델에 다른 운영 정책을
적용해 보려면 threshold 만 다시 고르면 되기 때문이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

THRESHOLD_FILE = "threshold.json"


def _f1(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    denominator = precision + recall
    return np.where(denominator > 0, 2 * precision * recall / np.maximum(denominator, 1e-12), 0.0)


def select(
    y_true: np.ndarray,
    score: np.ndarray,
    policy_cfg: dict,
    *,
    disk_far_fn=None,
) -> dict:
    """정책에 따라 임곗값 하나를 고르고 근거를 함께 돌려준다.

    disk_far_fn(threshold) -> 디스크 단위 FAR. fixed_disk_far 정책에만 쓴다.
    """
    policy = policy_cfg.get("policy", "max_f1")
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)

    if y_true.max() == y_true.min():
        raise ValueError(
            "validation 에 한쪽 클래스만 있습니다. split.min_val_failures 를 확인하세요."
        )

    if policy == "fixed_disk_far":
        # RODMAN 방식: 디스크 단위 오탐률을 고정하고 나머지를 읽는다.
        #   Alibaba FPR 0.1% / Backblaze FPR 4.0%
        # 행 단위 FPR 을 고정하는 것과 다르다. 분모가 미고장 디스크다.
        if disk_far_fn is None:
            raise ValueError("fixed_disk_far 정책에는 disk_far_fn 이 필요하다.")
        target = float(policy_cfg.get("fixed_disk_far", 0.04))
        chosen, achieved = _search_disk_far(
            score, disk_far_fn, target, int(policy_cfg.get("n_grid", 500))
        )
        return _package(
            policy,
            chosen,
            y_true,
            score,
            {"target_disk_far": target, "achieved_disk_far": achieved},
        )

    if policy == "fixed_fpr":
        target = float(policy_cfg.get("fixed_fpr", 0.001))
        fpr, tpr, thresholds = roc_curve(y_true, score)
        feasible = np.flatnonzero(fpr <= target)
        index = feasible[np.argmax(tpr[feasible])] if feasible.size else int(np.argmin(fpr))
        chosen = float(thresholds[index])
        rationale = {"target_fpr": target, "achieved_fpr": float(fpr[index]),
                     "achieved_tpr": float(tpr[index])}
        return _package(policy, chosen, y_true, score, rationale)

    precision, recall, thresholds = precision_recall_curve(y_true, score)
    # precision_recall_curve 는 마지막 점에 대응하는 임곗값이 없다.
    precision, recall = precision[:-1], recall[:-1]
    if thresholds.size == 0:
        raise ValueError("임곗값 후보가 없습니다.")

    if policy == "max_f1":
        index = int(np.argmax(_f1(precision, recall)))
        rationale = {"f1": float(_f1(precision, recall)[index])}
    elif policy == "precision_floor":
        floor = float(policy_cfg.get("precision_floor", 0.3))
        feasible = np.flatnonzero(precision >= floor)
        if feasible.size == 0:
            index = int(np.argmax(precision))
            rationale = {"precision_floor": floor, "floor_met": False}
        else:
            index = int(feasible[np.argmax(recall[feasible])])
            rationale = {"precision_floor": floor, "floor_met": True}
    elif policy == "recall_floor":
        floor = float(policy_cfg.get("recall_floor", 0.5))
        feasible = np.flatnonzero(recall >= floor)
        if feasible.size == 0:
            index = int(np.argmax(recall))
            rationale = {"recall_floor": floor, "floor_met": False}
        else:
            index = int(feasible[np.argmax(precision[feasible])])
            rationale = {"recall_floor": floor, "floor_met": True}
    else:
        raise ValueError(f"알 수 없는 threshold policy: {policy!r}")

    rationale |= {
        "val_precision": float(precision[index]),
        "val_recall": float(recall[index]),
    }
    return _package(policy, float(thresholds[index]), y_true, score, rationale)


def _search_disk_far(
    score: np.ndarray, disk_far_fn, target: float, n_grid: int
) -> tuple[float, float]:
    """디스크 단위 FAR <= target 을 만족하는 가장 낮은 임곗값을 찾는다.

    임곗값이 오르면 알람이 줄고 FAR 은 단조 감소하므로 이분 탐색이 성립한다.
    가장 낮은 임곗값을 고르는 이유는 제약 안에서 recall 을 최대로 하기 위해서다.
    """
    grid = np.unique(np.quantile(score, np.linspace(0.0, 1.0, n_grid)))
    if disk_far_fn(grid[-1]) > target:
        # 최댓값에서도 목표를 못 맞추면 가장 보수적인 지점을 준다.
        return float(grid[-1]), float(disk_far_fn(grid[-1]))

    low, high = 0, len(grid) - 1
    while low < high:
        mid = (low + high) // 2
        if disk_far_fn(grid[mid]) <= target:
            high = mid
        else:
            low = mid + 1
    return float(grid[low]), float(disk_far_fn(grid[low]))


def _package(policy: str, threshold: float, y_true, score, rationale: dict) -> dict:
    predicted = (score >= threshold).astype(int)
    return {
        "policy": policy,
        "threshold": threshold,
        "selected_on": "validation",
        "n_val": int(y_true.shape[0]),
        "val_positive_rate": float(y_true.mean()),
        "val_alarm_rate": float(predicted.mean()),
        "rationale": rationale,
    }


def save(record: dict, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / THRESHOLD_FILE
    with target.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    return target


def load(directory: Path) -> dict:
    with (Path(directory) / THRESHOLD_FILE).open("r", encoding="utf-8") as fh:
        return json.load(fh)
