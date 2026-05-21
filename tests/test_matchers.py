import pytest
from filtering.matchers import exact_match, max_match, sbert_match


def test_exact_match_identical():
    assert exact_match("dog", "dog") == 1.0


def test_exact_match_case_insensitive_with_punct():
    assert exact_match("A Dog.", "a dog") == 1.0


def test_exact_match_mismatch():
    assert exact_match("dog", "cat") == 0.0


def test_exact_match_yes_no():
    assert exact_match("yes", "Yes") == 1.0
    assert exact_match("no", "yes") == 0.0


class DummySbert:
    def encode(self, texts, convert_to_numpy=True):
        import numpy as np

        out = []
        for t in texts:
            t = t.lower().strip()
            if "dog" in t:
                out.append(np.array([1.0, 0.0]))
            elif "cat" in t:
                out.append(np.array([-1.0, 0.0]))
            else:
                out.append(np.array([0.5, 0.5]))
        return np.stack(out)


def test_sbert_match_paraphrase():
    score = sbert_match("a dog", "the dog", model=DummySbert())
    assert score == pytest.approx(1.0, abs=1e-6)


def test_sbert_match_different():
    score = sbert_match("dog", "cat", model=DummySbert())
    assert score == pytest.approx(0.0, abs=1e-6)


def test_max_match_prefers_exact_when_high():
    score = max_match("dog", "dog", sbert_model=DummySbert())
    assert score == 1.0


def test_max_match_falls_back_to_sbert():
    score = max_match("a dog", "the dog", sbert_model=DummySbert())
    assert score == pytest.approx(1.0, abs=1e-6)
