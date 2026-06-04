"""Unit tests for VQAScore adapter and reward integration (no t2v_metrics required)."""

from unittest.mock import MagicMock

import pytest

from filtering.reward import RewardConfig, RewardTerms, compose_reward, score_vqascore
from filtering.vqascore_adapter import (
    DEFAULT_TEMPLATE,
    VQAScoreAdapter,
    format_vqascore_text,
)


def test_format_vqascore_text():
    t = format_vqascore_text("How many dogs?", "three", DEFAULT_TEMPLATE)
    assert "How many dogs? three" in t
    assert "yes or no" in t.lower()


def test_vqascore_adapter_disabled_returns_zero():
    ad = VQAScoreAdapter(backend="none")
    assert ad.score("/tmp/x.jpg", "Q", "A") == 0.0


def test_vqascore_adapter_calls_scorer():
    ad = VQAScoreAdapter(backend="t2v_metrics", model="clip-flant5-xl")
    mock_scorer = MagicMock(return_value=[0.73])
    ad._scorer = mock_scorer
    s = ad.score("/tmp/img.jpg", "What color?", "red")
    assert abs(s - 0.73) < 1e-9
    mock_scorer.assert_called_once()
    args, kwargs = mock_scorer.call_args
    assert kwargs["images"] == ["/tmp/img.jpg"]
    assert "What color? red" in kwargs["texts"][0]


def test_score_vqascore_delegates():
    ad = MagicMock()
    ad.score.return_value = 0.42
    assert score_vqascore("p.jpg", "Q", "A", ad) == 0.42
    ad.score.assert_called_once_with("p.jpg", "Q", "A")


def test_compose_reward_includes_vqascore_term():
    cfg = RewardConfig(
        w_type=0.0,
        w_itm=0.5,
        w_grounding=1.0,
        w_learnability=0.5,
        w_kl=0.1,
        w_repetition=0.3,
        w_vqa=0.5,
    )
    terms = RewardTerms(
        itm=0.6,
        grounding=1.0,
        learnability=0.8,
        vqascore=0.9,
        kl=0.0,
        repetition=0.0,
    )
    # 0.5*0.6 + 1.0*1.0 + 0.5*0.8 + 0.5*0.9 = 2.15
    assert abs(compose_reward(terms, cfg) - 2.15) < 1e-9


def test_vqascore_adapter_import_error_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "t2v_metrics":
            raise ImportError("missing t2v_metrics")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    ad = VQAScoreAdapter(backend="t2v_metrics")
    with pytest.raises(ImportError, match="t2v-metrics"):
        ad._load_scorer()
