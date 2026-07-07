import pytest
import torch

from filtering.adapters import (
    VQAScoreAdapter,
    _first_token_ids,
    _none_if_blank,
    _score_yes_no_logits,
    _tensor_scores_to_list,
)


def test_tensor_scores_to_list_extracts_diagonal():
    raw = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    assert _tensor_scores_to_list(raw, 2) == pytest.approx([0.9, 0.8])


def test_tensor_scores_to_list_handles_vector():
    raw = torch.tensor([0.5, 0.7])
    assert _tensor_scores_to_list(raw, 2) == pytest.approx([0.5, 0.7])


def test_tensor_scores_to_list_handles_scalar():
    raw = torch.tensor(0.42)
    assert _tensor_scores_to_list(raw, 3) == pytest.approx([0.42, 0.42, 0.42])


def test_first_token_ids_deduplicates_candidate_spellings():
    class Tokenizer:
        bos_token_id = 1

        def encode(self, text, add_special_tokens=False):
            table = {
                "Yes": [10],
                " yes": [10],
                "yes": [11],
                "": [],
            }
            return table[text]

    assert _first_token_ids(Tokenizer(), ["Yes", " yes", "yes", ""]) == [10, 11]


def test_none_if_blank_normalizes_shell_null_values():
    assert _none_if_blank("null") is None
    assert _none_if_blank("none") is None
    assert _none_if_blank("") is None
    assert _none_if_blank("auto") == "auto"


def test_score_yes_no_logits_can_normalize_yes_against_no():
    logits = torch.zeros((2, 4, 8))
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    logits[0, 2, 2] = 3.0
    logits[0, 2, 3] = 0.0
    logits[1, 3, 2] = 0.0
    logits[1, 3, 3] = 3.0

    scores = _score_yes_no_logits(
        logits,
        attention_mask=attention_mask,
        yes_token_ids=[2],
        no_token_ids=[3],
        score_mode="yes_no_probability",
    )

    assert scores[0] > 0.9
    assert scores[1] < 0.1


def test_vqascore_adapter_uses_registered_backend(monkeypatch):
    calls = {}

    class Backend:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def score_pairs(self, image_paths, texts):
            return [0.25] * len(image_paths)

    monkeypatch.setitem(VQAScoreAdapter.BACKENDS, "custom", Backend)

    scorer = VQAScoreAdapter(
        backend="custom",
        model="some-vlm",
        device="cpu",
        prompt_template="prompt: {text}",
        score_mode="yes_no_probability",
    )

    assert calls["kwargs"]["model"] == "some-vlm"
    assert calls["kwargs"]["device"] == "cpu"
    assert calls["kwargs"]["prompt_template"] == "prompt: {text}"
    assert calls["kwargs"]["score_mode"] == "yes_no_probability"
    assert scorer.score_pairs(["a.jpg", "b.jpg"], ["a", "b"]) == [0.25, 0.25]


def test_pallava_backend_defaults_to_llama3_chat_prompt(monkeypatch):
    calls = {}

    class Backend:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def score_pairs(self, image_paths, texts):
            return [0.5] * len(image_paths)

    monkeypatch.setitem(VQAScoreAdapter.BACKENDS, "pallava", Backend)

    VQAScoreAdapter(backend="pallava", model="merged-pallava", device="cpu")

    prompt = calls["kwargs"]["prompt_template"]
    assert "<|start_header_id|>user<|end_header_id|>" in prompt
    assert "{image}" in prompt
    assert "{text}" in prompt
