from filtering.reward import grounding, learnability, lp_flip, repetition_penalty, type_match


def test_type_match_in_weak_set():
    assert type_match("How many dogs?", "3", {"how_many"}) == 1.0


def test_type_match_not_in_weak_set():
    assert type_match("Is it red?", "yes", {"how_many"}) == 0.0


def test_repetition_penalty_unique_is_zero():
    seen = ["how many dogs are there", "what color is the car"]
    assert repetition_penalty("is the sky blue", seen, threshold=0.9) == 0.0


def test_repetition_penalty_duplicate_is_one():
    seen = ["how many dogs are there"]
    assert repetition_penalty("how many dogs are there", seen, threshold=0.9) == 1.0


class _FakeStudentProb:
    def __init__(self, prob):
        self._p = prob

    def answer_prob(self, image, question):
        return self._p


def test_learnability_high_when_student_uncertain():
    assert abs(learnability("img", "Q", _FakeStudentProb(0.2)) - 0.8) < 1e-9


def test_learnability_low_when_student_confident():
    assert abs(learnability("img", "Q", _FakeStudentProb(0.95)) - 0.05) < 1e-9


class _FakeAnswerer:
    def __init__(self, mapping):
        self.mapping = mapping

    def answer_question(self, image, question):
        return self.mapping[image]


class _FakeSbert:
    """Fixed embeddings: same string -> cos=1; dog vs cat -> cos=-1 (max_match 0 vs 1)."""

    _TABLE = {
        "dog": [1.0, 0.0],
        "cat": [-1.0, 0.0],
    }

    def encode(self, texts, convert_to_numpy=True):
        import numpy as np

        return np.array([self._TABLE.get(t, [0.0, 1.0]) for t in texts])


def _corrupt(image):
    return image + "_blank"


def test_lp_flip_one_when_answer_changes():
    ans = _FakeAnswerer({"img": "dog", "img_blank": "cat"})
    assert lp_flip("img", "What animal?", ans, _corrupt, _FakeSbert()) == 1.0


def test_lp_flip_zero_when_answer_same():
    ans = _FakeAnswerer({"img": "dog", "img_blank": "dog"})
    assert lp_flip("img", "What animal?", ans, _corrupt, _FakeSbert()) == 0.0


def test_grounding_is_product_of_xcons_and_flip():
    ans = _FakeAnswerer({"img": "dog", "img_blank": "cat"})
    g = grounding("img", "What animal?", "dog", ans, _corrupt, _FakeSbert())
    assert g == 1.0
