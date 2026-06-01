from filtering.reward import learnability, repetition_penalty, type_match


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
