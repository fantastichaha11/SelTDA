#!/usr/bin/env python3
"""Download VizWiz-VQA files into the layout expected by SelTDA converters."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile


DEFAULT_TRAIN_IMAGES_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip"
DEFAULT_VAL_IMAGES_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip"
DEFAULT_TEST_IMAGES_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip"
DEFAULT_ANNOTATIONS_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip"
DEFAULT_TEST_ANNOTATIONS_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/VQA_test.json"

USER_AGENT = "SelTDA dataset downloader"


def download_file(url: str, output_path: str | Path, *, force: bool = False) -> bool:
    """Download `url` to `output_path`; return True if bytes were written."""
    output_path = Path(output_path)
    if output_path.exists() and not force:
        print(f"[skip] {output_path} already exists")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    print(f"[download] {url} -> {output_path}")
    with urlopen(request) as response, tmp_path.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    tmp_path.replace(output_path)
    return True


def extract_zip_files(
    zip_path: str | Path,
    output_dir: str | Path,
    *,
    suffixes: set[str] | None = None,
    force: bool = False,
) -> list[Path]:
    """Extract matching files from a zip archive, flattened by filename."""
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[Path] = []
    with ZipFile(zip_path) as zf:
        for member in sorted(zf.infolist(), key=lambda item: item.filename):
            if member.is_dir():
                continue
            source_name = Path(member.filename)
            if suffixes is not None and source_name.suffix.lower() not in suffixes:
                continue
            target = output_dir / source_name.name
            if target.exists() and not force:
                extracted.append(target)
                continue
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)
    print(f"[extract] {zip_path} -> {output_dir} ({len(extracted)} files)")
    return extracted


def prepare_vizwiz(
    output_root: str | Path,
    *,
    train_images_url: str = DEFAULT_TRAIN_IMAGES_URL,
    val_images_url: str = DEFAULT_VAL_IMAGES_URL,
    test_images_url: str = DEFAULT_TEST_IMAGES_URL,
    annotations_url: str = DEFAULT_ANNOTATIONS_URL,
    test_annotations_url: str = DEFAULT_TEST_ANNOTATIONS_URL,
    include_test: bool = False,
    force: bool = False,
) -> None:
    root = Path(output_root)
    downloads = root / "downloads"

    train_zip = downloads / "vizwiz_train.zip"
    val_zip = downloads / "vizwiz_val.zip"
    ann_zip = downloads / "vizwiz_annotations.zip"

    download_file(train_images_url, train_zip, force=force)
    download_file(val_images_url, val_zip, force=force)
    download_file(annotations_url, ann_zip, force=force)

    extract_zip_files(train_zip, root / "images" / "train", suffixes={".jpg", ".jpeg", ".png"}, force=force)
    extract_zip_files(val_zip, root / "images" / "val", suffixes={".jpg", ".jpeg", ".png"}, force=force)
    extract_zip_files(ann_zip, root / "annotations", suffixes={".json"}, force=force)

    if include_test:
        test_zip = downloads / "vizwiz_test.zip"
        download_file(test_images_url, test_zip, force=force)
        download_file(test_annotations_url, root / "annotations" / "test.json", force=force)
        extract_zip_files(test_zip, root / "images" / "test", suffixes={".jpg", ".jpeg", ".png"}, force=force)

    print(f"[done] VizWiz files are under {root}")
    print("       Next: python convert_vizwiz.py --vizwiz-root", root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("datasets/vizwiz"))
    parser.add_argument("--include-test", action="store_true", help="Also download test images/annotations.")
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite extracted files.")
    parser.add_argument("--train-images-url", default=DEFAULT_TRAIN_IMAGES_URL)
    parser.add_argument("--val-images-url", default=DEFAULT_VAL_IMAGES_URL)
    parser.add_argument("--test-images-url", default=DEFAULT_TEST_IMAGES_URL)
    parser.add_argument("--annotations-url", default=DEFAULT_ANNOTATIONS_URL)
    parser.add_argument("--test-annotations-url", default=DEFAULT_TEST_ANNOTATIONS_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_vizwiz(
        args.output_root,
        train_images_url=args.train_images_url,
        val_images_url=args.val_images_url,
        test_images_url=args.test_images_url,
        annotations_url=args.annotations_url,
        test_annotations_url=args.test_annotations_url,
        include_test=args.include_test,
        force=args.force,
    )


if __name__ == "__main__":
    main()
