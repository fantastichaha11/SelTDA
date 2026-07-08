import io
import tarfile
import zipfile
from pathlib import Path

from scripts.download_pathvqa import (
    PATHVQA_SYNTHETIC_DATA_FILE_ID,
    PATHVQA_TEACHER_CHECKPOINT_FILE_ID,
    download_google_drive_file,
    resolve_auxiliary_asset_paths,
)
from scripts.download_daquar import extract_tar_images
from scripts.download_vizwiz import download_file, extract_zip_files


def test_extract_zip_files_flattens_vizwiz_archives(tmp_path):
    archive = tmp_path / "vizwiz.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("train/VizWiz_train_00000001.jpg", b"image")
        zf.writestr("nested/ignore.txt", b"ignore")

    out = tmp_path / "images" / "train"
    extracted = extract_zip_files(archive, out, suffixes={".jpg"})

    assert extracted == [out / "VizWiz_train_00000001.jpg"]
    assert (out / "VizWiz_train_00000001.jpg").read_bytes() == b"image"
    assert not (out / "ignore.txt").exists()


def test_extract_zip_files_flattens_vizwiz_annotations(tmp_path):
    archive = tmp_path / "annotations.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Annotations/train.json", "[]")
        zf.writestr("example_code/eval.py", "pass")

    out = tmp_path / "annotations"
    extracted = extract_zip_files(archive, out, suffixes={".json"})

    assert extracted == [out / "train.json"]
    assert (out / "train.json").read_text() == "[]"


def test_extract_tar_images_flattens_daquar_image_archive(tmp_path):
    archive = tmp_path / "nyu_depth_images.tar"
    with tarfile.open(archive, "w") as tf:
        payload = b"image-bytes"
        info = tarfile.TarInfo("nyu_depth_images/image3.png")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    out = tmp_path / "images"
    extracted = extract_tar_images(archive, out)

    assert extracted == [out / "image3.png"]
    assert (out / "image3.png").read_bytes() == b"image-bytes"


def test_download_file_skips_existing_file(tmp_path):
    source = tmp_path / "source.txt"
    target = tmp_path / "target.txt"
    source.write_text("new")
    target.write_text("old")

    downloaded = download_file(source.as_uri(), target, force=False)

    assert downloaded is False
    assert target.read_text() == "old"


def test_resolve_auxiliary_asset_paths_uses_expected_defaults(tmp_path):
    output_root = tmp_path / "datasets" / "pathvqa"

    paths = resolve_auxiliary_asset_paths(output_root)

    assert paths["teacher_checkpoint"] == Path("cache/pathvqa_teacher_weights/checkpoint_04.pth")
    assert paths["synthetic_data"] == output_root / "synthetic_data_raw.json"


def test_resolve_auxiliary_asset_paths_honors_explicit_overrides(tmp_path):
    output_root = tmp_path / "datasets" / "pathvqa"
    teacher_path = tmp_path / "cache" / "teacher.pth"
    synthetic_path = tmp_path / "exports" / "synthetic.json"

    paths = resolve_auxiliary_asset_paths(
        output_root,
        teacher_checkpoint_output=teacher_path,
        synthetic_data_output=synthetic_path,
    )

    assert paths["teacher_checkpoint"] == teacher_path
    assert paths["synthetic_data"] == synthetic_path


def test_download_google_drive_file_skips_existing_target(tmp_path):
    target = tmp_path / "existing.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"existing")
    calls = []

    def fake_runner(command, check):
        calls.append((command, check))

    downloaded = download_google_drive_file(
        PATHVQA_TEACHER_CHECKPOINT_FILE_ID,
        target,
        runner=fake_runner,
    )

    assert downloaded is False
    assert calls == []


def test_download_google_drive_file_invokes_gdown_with_expected_command(tmp_path):
    target = tmp_path / "downloads" / "synthetic_data_raw.json"
    calls = []

    def fake_runner(command, check):
        calls.append((command, check))

    downloaded = download_google_drive_file(
        PATHVQA_SYNTHETIC_DATA_FILE_ID,
        target,
        runner=fake_runner,
    )

    assert downloaded is True
    assert calls == [
        (
            [
                "gdown",
                f"https://drive.google.com/uc?id={PATHVQA_SYNTHETIC_DATA_FILE_ID}",
                "-O",
                str(target),
            ],
            True,
        )
    ]
