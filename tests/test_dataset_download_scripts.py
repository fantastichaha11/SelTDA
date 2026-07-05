import io
import tarfile
import zipfile
from pathlib import Path

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
