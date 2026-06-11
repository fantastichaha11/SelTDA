import pytest
import torch

from filtering.adapters import _tensor_scores_to_list


def test_tensor_scores_to_list_extracts_diagonal():
    raw = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    assert _tensor_scores_to_list(raw, 2) == pytest.approx([0.9, 0.8])


def test_tensor_scores_to_list_handles_vector():
    raw = torch.tensor([0.5, 0.7])
    assert _tensor_scores_to_list(raw, 2) == pytest.approx([0.5, 0.7])


def test_tensor_scores_to_list_handles_scalar():
    raw = torch.tensor(0.42)
    assert _tensor_scores_to_list(raw, 3) == pytest.approx([0.42, 0.42, 0.42])
