"""Load and cache RGB images by dataset-relative path during filtering."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


class ImageCache:
    def __init__(self, image_root: Path):
        self.image_root = Path(image_root)
        self._cache: dict[str, Image.Image] = {}

    def get(self, image_field: str) -> Image.Image:
        if image_field not in self._cache:
            path = self.image_root / image_field
            self._cache[image_field] = Image.open(path).convert("RGB")
        return self._cache[image_field]

    def get_copy(self, image_field: str) -> Image.Image:
        return self.get(image_field).copy()
