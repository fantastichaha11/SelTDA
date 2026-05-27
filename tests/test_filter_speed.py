"""Tests for filter speed helpers (image cache, batch adapters, LP reuse)."""

from unittest.mock import MagicMock

from PIL import Image

from filtering.image_cache import ImageCache
from filtering.scorers_language_prior import score_language_prior


def test_image_cache_returns_same_object(tmp_path):
    img_dir = tmp_path / "coco"
    img_dir.mkdir()
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(img_dir / "a.jpg")
    cache = ImageCache(img_dir)
    assert cache.get("a.jpg") is cache.get("a.jpg")


def test_score_language_prior_reuses_a_clean():
    student = MagicMock()
    student.answer_question.side_effect = ["no"]
    record = {"question": "Is it red?", "answer": ["yes"]}
    img = Image.new("RGB", (8, 8))
    s = score_language_prior(record, img, student, corruption="gray_box", a_clean="yes")
    assert s == 1.0
    assert student.answer_question.call_count == 1


def test_blip_adapter_batch_delegates_single_to_batch():
    from filtering.adapters import BlipStudentAdapter

    adapter = BlipStudentAdapter.__new__(BlipStudentAdapter)
    adapter._torch = __import__("torch")
    adapter.device = "cpu"
    adapter.preprocess = lambda img: adapter._torch.zeros(3, 4, 4)
    adapter.model = MagicMock(return_value=["one"])

    out = adapter.answer_question(None, "q1")
    assert out == "one"
    adapter.model.assert_called_once()
