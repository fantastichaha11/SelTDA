from unittest.mock import MagicMock

from filtering.scorers_knowledge import score_knowledge_consistency


def test_score_knowledge_max_entailment():
    nli = MagicMock()
    nli.predict.return_value = [[0.1, 0.9, 0.2]]
    s = score_knowledge_consistency("Q?", "A", ["p1", "p2", "p3"], nli)
    assert s == 0.9
