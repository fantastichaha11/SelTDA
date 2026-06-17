import pytest
from filtering.gates import (
    GateThresholds,
    apply_gates,
    apply_soft_fusion,
    fused_score,
    normalize_gate_scores,
    normalize_rank,
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


def test_normalize_rank_spreads_values():
    out = normalize_rank([10.0, 20.0, 30.0])
    assert out == pytest.approx([0.0, 0.5, 1.0])


def test_fused_score_weighted_average():
    norm = {"conf": 1.0, "itm": 0.0, "xcons": 0.5}
    weights = {"conf": 0.5, "itm": 0.25, "xcons": 0.25}
    assert fused_score(norm, weights, ["conf", "itm", "xcons"]) == pytest.approx(0.625)


def test_apply_soft_fusion_keeps_above_tau():
    norm = {"conf": 0.9, "itm": 0.8, "xcons": 0.7}
    weights = {"conf": 1.0, "itm": 1.0, "xcons": 1.0}
    keep, score = apply_soft_fusion(norm, weights, 0.7, ["conf", "itm", "xcons"])
    assert keep is True
    assert score == pytest.approx(0.8)


def test_apply_soft_fusion_rejects_below_tau():
    norm = {"conf": 0.2, "itm": 0.1, "xcons": 0.0}
    weights = {"conf": 1.0, "itm": 1.0, "xcons": 1.0}
    keep, score = apply_soft_fusion(norm, weights, 0.5, ["conf", "itm", "xcons"])
    assert keep is False
    assert score == pytest.approx(0.1)


def test_normalize_gate_scores_minmax():
    out = normalize_gate_scores([0.0, 0.5, 1.0], mode="minmax")
    assert out == pytest.approx([0.0, 0.5, 1.0])
