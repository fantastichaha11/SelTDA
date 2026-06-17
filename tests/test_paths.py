from pathlib import Path

from filtering.paths import resolve_dataset_path


def test_resolve_dataset_path_prefers_existing_uploads(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    target = uploads / "aokvqa" / "data.json"
    target.parent.mkdir(parents=True)
    target.write_text("[]")

    monkeypatch.setattr("filtering.paths.UPLOADS_ROOT", uploads)
    monkeypatch.setattr(
        "filtering.paths.LOCAL_ROOT", tmp_path / "repo" / "datasets"
    )

    resolved = resolve_dataset_path(uploads / "aokvqa" / "data.json")
    assert resolved == target.resolve()


def test_resolve_dataset_path_falls_back_to_local_datasets(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    local = tmp_path / "repo" / "datasets"
    local.mkdir(parents=True)
    target = local / "aokvqa" / "data.json"
    target.parent.mkdir(parents=True)
    target.write_text("[]")

    monkeypatch.setattr("filtering.paths.UPLOADS_ROOT", uploads)
    monkeypatch.setattr("filtering.paths.LOCAL_ROOT", local)

    resolved = resolve_dataset_path(uploads / "aokvqa" / "data.json")
    assert resolved == target.resolve()
