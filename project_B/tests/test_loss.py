"""손실 함수 검증.

focal loss 는 쉬운 표본의 기여를 눌러야 하고, gamma=0 이면 가중된 BCE 와
같아져야 한다. 이 두 성질이 깨지면 학습이 조용히 달라진다.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from hddpred.models.sequence import FocalLoss, build_criterion


def _bce(logits, targets):
    return nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )


def test_gamma_zero_reduces_to_alpha_weighted_bce():
    logits = torch.tensor([-2.0, 0.5, 3.0, -0.1])
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0])
    alpha = 0.25

    got = FocalLoss(gamma=0.0, alpha=alpha)(logits, targets)
    weight = alpha * targets + (1 - alpha) * (1 - targets)
    expected = (weight * _bce(logits, targets)).mean()
    torch.testing.assert_close(got, expected)


def test_alpha_none_and_gamma_zero_is_plain_bce():
    logits = torch.tensor([-2.0, 0.5, 3.0])
    targets = torch.tensor([0.0, 1.0, 1.0])
    got = FocalLoss(gamma=0.0, alpha=None)(logits, targets)
    torch.testing.assert_close(got, _bce(logits, targets).mean())


def test_easy_examples_are_downweighted_more_than_hard_ones():
    """focal 의 핵심 성질. 잘 맞힌 표본일수록 손실 기여가 더 많이 줄어든다."""
    targets = torch.tensor([1.0])
    focal = FocalLoss(gamma=2.0, alpha=None)

    ratios = []
    for logit in [0.0, 2.0, 4.0, 6.0]:  # 점점 더 쉬운(확신하는) 양성
        x = torch.tensor([logit])
        ratios.append(float(focal(x, targets) / _bce(x, targets).mean()))

    # 쉬워질수록 BCE 대비 비율이 단조 감소해야 한다.
    assert all(a > b for a, b in zip(ratios, ratios[1:]))
    assert ratios[0] == pytest.approx(0.25, rel=1e-3)  # p=0.5 -> (1-0.5)^2


def test_higher_gamma_downweights_easy_examples_further():
    x = torch.tensor([3.0])
    targets = torch.tensor([1.0])
    losses = [float(FocalLoss(gamma=g, alpha=None)(x, targets)) for g in [0.0, 1.0, 2.0, 5.0]]
    assert all(a > b for a, b in zip(losses, losses[1:]))


def test_loss_is_finite_at_extreme_logits():
    logits = torch.tensor([-40.0, 40.0, -40.0, 40.0])
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0])
    value = FocalLoss(gamma=2.0, alpha=0.25)(logits, targets)
    assert torch.isfinite(value)


def test_gradient_flows():
    logits = torch.tensor([0.3, -1.2], requires_grad=True)
    targets = torch.tensor([1.0, 0.0])
    FocalLoss(gamma=2.0, alpha=0.25)(logits, targets).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


# --------------------------------------------------------------------------
# build_criterion
# --------------------------------------------------------------------------
Y = np.array([0] * 990 + [1] * 10, dtype=np.int8)
DEVICE = torch.device("cpu")


def test_default_is_bce_without_weighting():
    criterion, info = build_criterion({}, Y, DEVICE)
    assert isinstance(criterion, nn.BCEWithLogitsLoss)
    assert info == {"loss": "bce", "pos_weight": None}


def test_auto_pos_weight_uses_the_class_ratio():
    _, info = build_criterion({"auto_pos_weight": True}, Y, DEVICE)
    assert info["pos_weight"] == pytest.approx(99.0)


def test_focal_is_selected_and_records_its_parameters():
    criterion, info = build_criterion(
        {"loss": "focal", "focal_gamma": 3.0, "focal_alpha": 0.4}, Y, DEVICE
    )
    assert isinstance(criterion, FocalLoss)
    assert info["focal_gamma"] == 3.0
    assert info["focal_alpha"] == 0.4


def test_focal_alpha_auto_uses_one_minus_positive_rate():
    _, info = build_criterion({"loss": "focal", "focal_alpha": "auto"}, Y, DEVICE)
    assert info["focal_alpha"] == pytest.approx(0.99)


def test_focal_ignores_pos_weight_and_says_so():
    """불균형 보정이 두 번 걸리면 안 된다."""
    criterion, info = build_criterion(
        {"loss": "focal", "auto_pos_weight": True}, Y, DEVICE
    )
    assert isinstance(criterion, FocalLoss)
    assert "무시" in info["note"]


def test_unknown_loss_is_rejected():
    with pytest.raises(ValueError, match="training.loss"):
        build_criterion({"loss": "hinge"}, Y, DEVICE)


def test_all_sequence_model_configs_declare_focal():
    from hddpred import paths
    from hddpred.models import registry

    for name in ["lstm", "gru", "tcn", "transformer"]:
        cfg = registry.load_model_config(paths.CONFIG_DIR / "models" / f"{name}.yaml")
        assert cfg["training"]["loss"] == "focal", name
        assert cfg["training"]["auto_pos_weight"] is False, name
