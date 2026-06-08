"""Convert SelTDA A-OKVQA JSON to CLIP-FlanT5 / LLaVA supervised JSON."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cli
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

IMAGE_TOKEN = "<image>"


def pick_answer(answer: Any) -> str:
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer.strip()
    if isinstance(answer, list):
        items = [str(a).strip() for a in answer if str(a).strip()]
        if not items:
            return ""
        return Counter(items).most_common(1)[0][0]
    return str(answer).strip()


def coco_image_relpath(image_name: str, *, image_prefix: str) -> str:
    name = Path(image_name).name
    prefix = image_prefix.strip("/")
    return f"{prefix}/{name}" if prefix else name


def resolve_image_path(vqa_root: Path, rel_image: str) -> Path | None:
    """Resolve CLIP-FlanT5 relative path to an on-disk COCO file."""
    name = Path(rel_image).name
    candidates = [
        vqa_root / rel_image,
        vqa_root / name,
        vqa_root / "train2017" / name,
        vqa_root / "val2017" / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def record_to_llava(
    record: dict,
    *,
    mode: str,
    image_prefix: str,
    vqascore_template: str,
) -> dict | None:
    question = str(record.get("question", "")).strip()
    if not question:
        return None
    answer = pick_answer(record.get("answer"))
    if not answer:
        return None
    image_name = record.get("image")
    if not image_name:
        return None
    rel_image = coco_image_relpath(str(image_name), image_prefix=image_prefix)

    if mode == "vqascore":
        qa = f"{question} {answer}".strip()
        prompt = vqascore_template.replace("{qa}", qa)
        human = f"{IMAGE_TOKEN}\n{prompt}"
        gpt = "Yes"
    elif mode == "vqa":
        human = f"{IMAGE_TOKEN}\n{question}"
        gpt = answer
    else:
        raise ValueError(f"Unknown mode {mode!r}; use 'vqa' or 'vqascore'")

    out: dict[str, Any] = {
        "id": str(record.get("question_id", "")),
        "image": rel_image,
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": gpt},
        ],
    }
    if record.get("dataset"):
        out["dataset"] = record["dataset"]
    return out


def load_seltda_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list")
    return data


def convert_records(
    records: Iterable[dict],
    *,
    mode: str,
    image_prefix: str,
    vqascore_template: str,
    vqa_root: Path,
    skip_missing_images: bool,
) -> tuple[list[dict], int]:
    out: list[dict] = []
    skipped = 0
    for record in records:
        rel = record_to_llava(
            record,
            mode=mode,
            image_prefix=image_prefix,
            vqascore_template=vqascore_template,
        )
        if rel is None:
            skipped += 1
            continue
        if skip_missing_images and resolve_image_path(vqa_root, rel["image"]) is None:
            skipped += 1
            continue
        out.append(rel)
    return out, skipped


def main(config: DictConfig) -> None:
    ann_root = Path(config.ann_root)
    vqa_root = Path(config.vqa_root)
    mode = str(config.mode).lower()
    image_prefix = str(config.image_prefix)
    output = Path(config.output)
    skip_missing = bool(config.get("skip_missing_images", True))
    template = str(config.get("vqascore_template", ""))

    all_records: list[dict] = []
    for split in list(config.splits):
        path = ann_root / f"{split}.json"
        if not path.is_file():
            logger.warning("Skip missing split file %s", path)
            continue
        records = load_seltda_json(path)
        logger.info("Loaded %d records from %s", len(records), path)
        all_records.extend(records)

    synthetic = config.get("synthetic")
    if synthetic:
        syn_path = Path(synthetic)
        if syn_path.is_file():
            syn_records = load_seltda_json(syn_path)
            logger.info("Loaded %d synthetic records from %s", len(syn_records), syn_path)
            all_records.extend(syn_records)

    converted, skipped = convert_records(
        all_records,
        mode=mode,
        image_prefix=image_prefix,
        vqascore_template=template,
        vqa_root=vqa_root,
        skip_missing_images=skip_missing,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(converted, ensure_ascii=False))
    logger.info(
        "Wrote %d samples to %s (skipped %d, mode=%s)",
        len(converted),
        output,
        skipped,
        mode,
    )


if __name__ == "__main__":
    args, cfg = cli.parse_args(default_config_path="./configs/convert_clip_flant5.yaml")
    main(cfg)
