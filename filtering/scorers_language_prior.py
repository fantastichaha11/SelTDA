"""LP-1: language-prior gate via image corruption."""

from __future__ import annotations

import numpy as np
from PIL import Image

from filtering.matchers import max_match


def corrupt_image(img: Image.Image, method: str) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    if method == "gaussian_noise":
        arr = np.clip(arr + np.random.randn(*arr.shape) * 64, 0, 255)
    elif method == "gray_box":
        arr[:] = arr.mean()
    elif method == "shuffle_patches":
        h, w = arr.shape[:2]
        ph, pw = max(1, h // 4), max(1, w // 4)
        patches = []
        coords = []
        for i in range(0, h, ph):
            for j in range(0, w, pw):
                patch = arr[i : i + ph, j : j + pw].copy()
                patches.append(patch)
                coords.append((i, j))
        order = np.random.permutation(len(patches))
        out = arr.copy()
        for idx, (i, j) in enumerate(coords):
            pi, pj = coords[order[idx]]
            out[i : i + ph, j : j + pw] = patches[order[idx]]
    else:
        raise ValueError(f"Unknown corruption method: {method}")
    return Image.fromarray(arr.astype(np.uint8))


def score_language_prior(
    record: dict,
    image,
    student,
    corruption: str = "gaussian_noise",
    sbert=None,
) -> float:
    answer = record["answer"]
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    q = record["question"]
    a_clean = student.answer_question(image, q)
    corrupt_img = corrupt_image(
        image if isinstance(image, Image.Image) else Image.open(image).convert("RGB"),
        corruption,
    )
    a_corrupt = student.answer_question(corrupt_img, q)
    clean_ok = max_match(a_clean, answer, sbert_model=sbert) >= 0.5
    corrupt_ok = max_match(a_corrupt, answer, sbert_model=sbert) >= 0.5
    if clean_ok and not corrupt_ok:
        return 1.0
    if clean_ok and corrupt_ok:
        return 0.0
    return 0.5
