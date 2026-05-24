import pytest
from pathlib import Path

from checkpoint_utils import resolve_resume_checkpoint


def test_resolve_resume_checkpoint_none():
    assert resolve_resume_checkpoint(None, "/tmp/out") is None


def test_resolve_resume_checkpoint_explicit_path(tmp_path):
    ckpt = tmp_path / "checkpoint_03.pth"
    ckpt.write_bytes(b"fake")
    assert resolve_resume_checkpoint(str(ckpt), tmp_path) == str(ckpt)


def test_resolve_resume_checkpoint_auto_picks_latest(tmp_path):
    (tmp_path / "checkpoint_01.pth").write_bytes(b"1")
    latest = tmp_path / "checkpoint_09.pth"
    latest.write_bytes(b"9")
    (tmp_path / "checkpoint_05.pth").write_bytes(b"5")

    assert resolve_resume_checkpoint("auto", tmp_path) == str(latest)


def test_resolve_resume_checkpoint_auto_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        resolve_resume_checkpoint("auto", tmp_path)


def test_resolve_resume_checkpoint_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_resume_checkpoint(str(tmp_path / "missing.pth"), tmp_path)
