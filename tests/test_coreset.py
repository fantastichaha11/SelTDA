import numpy as np
from filtering.coreset import select_coreset


def test_select_coreset_respects_budget():
    emb = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    idx = select_coreset(emb, budget=2, diversity_weight=0.5)
    assert len(idx) == 2
    assert len(set(idx)) == 2
