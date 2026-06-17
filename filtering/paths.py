"""Resolve dataset paths: prefer /teamspace/uploads, fallback to repo datasets/."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

UPLOADS_ROOT = Path("/teamspace/uploads")
LOCAL_ROOT = Path(__file__).resolve().parents[1] / "datasets"


def _local_fallback(uploads_path: Path) -> Path | None:
    try:
        rel = uploads_path.relative_to(UPLOADS_ROOT)
    except ValueError:
        return None
    candidate = LOCAL_ROOT / rel
    return candidate if candidate.exists() else None


def resolve_dataset_path(path: str | Path) -> Path:
    """Return existing path, or repo-local datasets/ mirror of /teamspace/uploads."""
    p = Path(path)
    if p.exists():
        return p.resolve()

    if p.is_absolute():
        alt = _local_fallback(p)
        if alt is not None:
            logger.info("Using local fallback %s for %s", alt, p)
            return alt.resolve()

    return p
