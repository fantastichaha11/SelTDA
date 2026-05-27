import pytest
from filtering.scorers import score_xcons


class StubStudent:
    def __init__(self, answer: str):
        self.answer = answer
        self.last_call = None

    def answer_question(self, image, question: str) -> str:
        self.last_call = (image, question)
        return self.answer


class StubSbert:
    def encode(self, texts, convert_to_numpy=True):
        import numpy as np

        out = []
        for t in texts:
            out.append(np.array([1.0, 0.0]) if "dog" in t.lower() else np.array([-1.0, 0.0]))
        return np.stack(out)


def test_score_xcons_exact_match():
    student = StubStudent("dog")
    s, pred = score_xcons(
        record={"question": "what is this?", "answer": ["dog"]},
        image=None,
        student=student,
        sbert=StubSbert(),
    )
    assert s == 1.0
    assert pred == "dog"


def test_score_xcons_falls_back_to_sbert_paraphrase():
    student = StubStudent("a dog")
    s, _ = score_xcons(
        record={"question": "what is this?", "answer": ["dog"]},
        image=None,
        student=student,
        sbert=StubSbert(),
    )
    assert s == pytest.approx(1.0, abs=1e-6)


def test_score_xcons_mismatch():
    student = StubStudent("cat")
    s, _ = score_xcons(
        record={"question": "what is this?", "answer": ["dog"]},
        image=None,
        student=student,
        sbert=StubSbert(),
    )
    assert s == 0.0


def test_score_xcons_passes_correct_question_to_student():
    student = StubStudent("dog")
    score_xcons(
        record={"question": "what is this?", "answer": ["dog"]},
        image="img_obj",
        student=student,
        sbert=StubSbert(),
    )
    assert student.last_call == ("img_obj", "what is this?")


def test_score_xcons_reuses_predicted_without_student_call():
    student = StubStudent("dog")
    s, pred = score_xcons(
        record={"question": "what is this?", "answer": ["dog"]},
        image=None,
        student=student,
        sbert=StubSbert(),
        predicted="dog",
    )
    assert s == 1.0
    assert pred == "dog"
    assert student.last_call is None
