import pytest
from filtering.gates import (
    GateThresholds,
    apply_gates,
    apply_soft_fusion,
    thresholds_from_quantile,
)


def test_apply_gates_keeps_when_all_above():
    scores = {"conf": 0.8, "itm": 0.7, "xcons": 0.9}
    keep, reason = apply_gates(scores, GateThresholds(0.5, 0.5, 0.5))
    assert keep is True
    assert reason == "kept"


def test_apply_gates_rejects_on_conf():
    scores = {"conf": 0.1, "itm": 0.9, "xcons": 0.9}
    keep, reason = apply_gates(scores, GateThresholds(0.5, 0.5, 0.5))
    assert keep is False
    assert reason == "conf"


def test_apply_gates_rejects_on_itm_even_if_conf_passes():
    scores = {"conf": 0.9, "itm": 0.1, "xcons": 0.9}
    keep, reason = apply_gates(scores, GateThresholds(0.5, 0.5, 0.5))
    assert (keep, reason) == (False, "itm")


def test_apply_gates_rejects_on_xcons_last():
    scores = {"conf": 0.9, "itm": 0.9, "xcons": 0.1}
    keep, reason = apply_gates(scores, GateThresholds(0.5, 0.5, 0.5))
    assert (keep, reason) == (False, "xcons")


def test_apply_gates_disabled_via_neg_infinity():
    scores = {"conf": -10.0, "itm": -10.0, "xcons": -10.0}
    keep, reason = apply_gates(
        scores, GateThresholds(float("-inf"), float("-inf"), float("-inf"))
    )
    assert keep is True


def test_thresholds_from_quantile_keep_top_75():
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    t = thresholds_from_quantile(values, keep_top=0.75)
    assert t == pytest.approx(0.325, abs=0.01)


def test_thresholds_from_quantile_keep_all():
    values = [0.1, 0.2, 0.3]
    t = thresholds_from_quantile(values, keep_top=1.0)
    assert t == float("-inf")


def test_thresholds_from_quantile_keep_top_1():
    values = [0.1, 0.2, 0.3]
    t = thresholds_from_quantile(values, keep_top=0.0)
    assert t == float("inf")


def test_apply_soft_fusion_stub_raises():
    with pytest.raises(NotImplementedError):
        apply_soft_fusion({"conf": 0.5, "itm": 0.5, "xcons": 0.5}, (1, 1, 1), 0.5)
