"""Unit tests for VQAScore adapter and reward integration (no t2v_metrics required)."""

from unittest.mock import MagicMock

import pytest

from filtering.reward import (
    RewardConfig,
    RewardTerms,
    compose_reward,
    score_teacher_conf,
    score_vqascore,
    score_vqascore_margin,
)
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
    mock_scorer = MagicMock()
    mock_scorer.forward.return_value = [0.73]
    ad._scorer = mock_scorer
    s = ad.score("/tmp/img.jpg", "What color?", "red")
    assert abs(s - 0.73) < 1e-9
    mock_scorer.forward.assert_called_once()
    args, kwargs = mock_scorer.forward.call_args
    assert args[0] == ["/tmp/img.jpg"]
    assert args[1] == ["What color? red"]
    assert "{}" in kwargs["question_template"]
    assert kwargs["answer_template"] == "Yes"


def test_vqascore_adapter_yes_no_margin():
    ad = VQAScoreAdapter(backend="t2v_metrics", model="clip-flant5-xl")
    mock_scorer = MagicMock()
    mock_scorer.forward.side_effect = [[0.8], [0.3]]
    ad._scorer = mock_scorer
    margin = ad.score_yes_no_margin("/tmp/img.jpg", "Is it red?", "yes")
    assert abs(margin - 0.5) < 1e-9
    assert mock_scorer.forward.call_count == 2
    assert mock_scorer.forward.call_args_list[0].kwargs["answer_template"] == "Yes"
    assert mock_scorer.forward.call_args_list[1].kwargs["answer_template"] == "No"


def test_score_vqascore_delegates():
    ad = MagicMock()
    ad.score.return_value = 0.42
    assert score_vqascore("p.jpg", "Q", "A", ad) == 0.42
    ad.score.assert_called_once_with("p.jpg", "Q", "A")


def test_score_vqascore_margin_delegates():
    ad = MagicMock()
    ad.score_yes_no_margin.return_value = 0.25
    assert score_vqascore_margin("p.jpg", "Q", "A", ad) == 0.25


def test_score_teacher_conf():
    assert score_teacher_conf(-1.5) == -1.5
    assert score_teacher_conf(None) == 0.0


def test_compose_reward_vqascore_conf_only():
    cfg = RewardConfig(
        w_type=0.0,
        w_itm=0.0,
        w_grounding=0.0,
        w_learnability=0.0,
        w_kl=0.0,
        w_repetition=0.0,
        w_vqa=1.0,
        w_conf=1.0,
    )
    terms = RewardTerms(vqascore=0.4, gen_logprob=-1.2)
    assert abs(compose_reward(terms, cfg) - (-0.8)) < 1e-9


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


def test_vqascore_adapter_local_checkpoint_requires_dir(tmp_path):
    ad = VQAScoreAdapter(backend="t2v_metrics", checkpoint=str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="vqascore_checkpoint"):
        ad._load_scorer()


def test_vqascore_adapter_import_error_message(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    ad = VQAScoreAdapter(backend="t2v_metrics")
    with pytest.raises(ImportError, match="t2v-metrics"):
        ad._load_scorer()
