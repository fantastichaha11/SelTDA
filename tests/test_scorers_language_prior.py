from PIL import Image
import numpy as np
from unittest.mock import MagicMock

from filtering.scorers_language_prior import corrupt_image, score_language_prior


def test_corrupt_gaussian_changes_pixels():
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    out = corrupt_image(img, "gaussian_noise")
    assert not np.array_equal(np.array(img), np.array(out))


def test_score_language_prior_grounded():
    student = MagicMock()
    student.answer_question.side_effect = ["yes", "no"]
    record = {"question": "Is it red?", "answer": ["yes"]}
    img = Image.new("RGB", (8, 8))
    s = score_language_prior(record, img, student, corruption="gray_box")
    assert s == 1.0
