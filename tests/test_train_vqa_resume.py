import pytest
from pathlib import Path

from checkpoint_utils import (
    prune_checkpoints,
    resolve_resume_checkpoint,
    resolve_training_resume,
)


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


def test_prune_checkpoints_keeps_latest_five(tmp_path):
    for epoch in range(10):
        (tmp_path / f"checkpoint_{epoch:02d}.pth").write_bytes(str(epoch).encode())

    removed = prune_checkpoints(tmp_path, 5)

    assert len(removed) == 5
    remaining = sorted(tmp_path.glob("checkpoint_*.pth"))
    assert [p.name for p in remaining] == [
        "checkpoint_05.pth",
        "checkpoint_06.pth",
        "checkpoint_07.pth",
        "checkpoint_08.pth",
        "checkpoint_09.pth",
    ]


def test_prune_checkpoints_noop_when_disabled(tmp_path):
    (tmp_path / "checkpoint_00.pth").write_bytes(b"0")
    assert prune_checkpoints(tmp_path, None) == []
    assert prune_checkpoints(tmp_path, 0) == []
    assert list(tmp_path.glob("checkpoint_*.pth"))


def test_resolve_training_resume_auto_picks_latest(tmp_path, capsys):
    (tmp_path / "checkpoint_01.pth").write_bytes(b"1")
    (tmp_path / "checkpoint_09.pth").write_bytes(b"9")

    path = resolve_training_resume(None, tmp_path, auto=True)

    assert path == str(tmp_path / "checkpoint_09.pth")
    assert "Auto-resume" in capsys.readouterr().out


def test_resolve_training_resume_none_when_empty(tmp_path):
    assert resolve_training_resume(None, tmp_path, auto=True) is None


def test_resolve_training_resume_disabled_when_no_resume(tmp_path):
    (tmp_path / "checkpoint_03.pth").write_bytes(b"3")
    assert resolve_training_resume(None, tmp_path, auto=False) is None


def test_resolve_training_resume_explicit_overrides_auto(tmp_path):
    ckpt = tmp_path / "checkpoint_02.pth"
    ckpt.write_bytes(b"2")
    assert resolve_training_resume(str(ckpt), tmp_path, auto=False) == str(ckpt)
