import numpy as np
import pytest
from filtering.scorers import score_clip_itm


class StubClip:
    def __init__(self, image_emb, text_emb):
        self.image_emb = image_emb
        self.text_emb = text_emb
        self.last_text = None

    def embed_image(self, image):
        return self.image_emb

    def embed_text(self, text):
        self.last_text = text
        return self.text_emb


def test_score_clip_itm_perfect_alignment():
    clip = StubClip(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    s = score_clip_itm(
        record={"question": "is this a dog?", "answer": ["yes"]},
        image=None,
        clip=clip,
    )
    assert s == pytest.approx(1.0, abs=1e-6)


def test_score_clip_itm_orthogonal_is_half():
    clip = StubClip(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    s = score_clip_itm(
        record={"question": "q?", "answer": ["a"]},
        image=None,
        clip=clip,
    )
    assert s == pytest.approx(0.5, abs=1e-6)


def test_score_clip_itm_text_format_includes_question_and_answer():
    clip = StubClip(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    score_clip_itm(
        record={"question": "is this a dog?", "answer": ["yes"]},
        image=None,
        clip=clip,
    )
    assert clip.last_text == "is this a dog? yes"


def test_score_clip_itm_uses_first_answer_when_list():
    clip = StubClip(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    score_clip_itm(
        record={"question": "q?", "answer": ["a", "b", "c"]},
        image=None,
        clip=clip,
    )
    assert clip.last_text == "q? a"
