#!/usr/bin/env python3
"""Download DAQUAR reduced QA files and NYU Depth images for SelTDA."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_IMAGES_URL = "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/nyu_depth_images.tar"
DEFAULT_TRAIN_QA_URL = "https://raw.githubusercontent.com/jayantk/lsp/master/data/daquar/reduced/qa.37.raw.train.txt"
DEFAULT_TEST_QA_URL = "https://raw.githubusercontent.com/jayantk/lsp/master/data/daquar/reduced/qa.37.raw.reduced.test.txt"

USER_AGENT = "SelTDA dataset downloader"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


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


def extract_tar_images(
    tar_path: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> list[Path]:
    """Extract image files from a tar archive, flattened by filename."""
    tar_path = Path(tar_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[Path] = []
    with tarfile.open(tar_path) as tf:
        members = [member for member in tf.getmembers() if member.isfile()]
        for member in sorted(members, key=lambda item: item.name):
            source_name = Path(member.name)
            if source_name.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            target = output_dir / source_name.name
            if target.exists() and not force:
                extracted.append(target)
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            with src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)
    print(f"[extract] {tar_path} -> {output_dir} ({len(extracted)} images)")
    return extracted


def prepare_daquar(
    output_root: str | Path,
    *,
    images_url: str = DEFAULT_IMAGES_URL,
    train_qa_url: str = DEFAULT_TRAIN_QA_URL,
    test_qa_url: str = DEFAULT_TEST_QA_URL,
    force: bool = False,
) -> None:
    root = Path(output_root)
    downloads = root / "downloads"
    images_tar = downloads / "nyu_depth_images.tar"

    download_file(train_qa_url, root / "qa.37.raw.train.txt", force=force)
    download_file(test_qa_url, root / "qa.37.raw.reduced.test.txt", force=force)
    download_file(images_url, images_tar, force=force)
    extract_tar_images(images_tar, root / "images", force=force)

    print(f"[done] DAQUAR files are under {root}")
    print(
        "       Next: python convert_daquar.py --daquar-root",
        root,
        "--train-qa",
        root / "qa.37.raw.train.txt",
        "--test-qa",
        root / "qa.37.raw.reduced.test.txt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("datasets/daquar"))
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite extracted files.")
    parser.add_argument("--images-url", default=DEFAULT_IMAGES_URL)
    parser.add_argument("--train-qa-url", default=DEFAULT_TRAIN_QA_URL)
    parser.add_argument("--test-qa-url", default=DEFAULT_TEST_QA_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_daquar(
        args.output_root,
        images_url=args.images_url,
        train_qa_url=args.train_qa_url,
        test_qa_url=args.test_qa_url,
        force=args.force,
    )


if __name__ == "__main__":
    main()
