import pytest
from filtering.scorers import normalize_min_max, score_confidence


def test_score_confidence_returns_log_prob_when_present():
    record = {"gen_logprob": -1.5}
    assert score_confidence(record) == -1.5


def test_score_confidence_returns_none_when_missing():
    assert score_confidence({}) is None


def test_score_confidence_returns_none_when_explicit_none():
    assert score_confidence({"gen_logprob": None}) is None


def test_normalize_min_max_basic():
    out = normalize_min_max([-3.0, -1.0, 0.0])
    assert out == pytest.approx([0.0, 2.0 / 3.0, 1.0])


def test_normalize_min_max_handles_constant():
    out = normalize_min_max([-2.0, -2.0, -2.0])
    assert out == [0.5, 0.5, 0.5]


def test_normalize_min_max_ignores_nones():
    out = normalize_min_max([-3.0, None, 0.0])
    assert out[0] == pytest.approx(0.0)
    assert out[1] is None
    assert out[2] == pytest.approx(1.0)
