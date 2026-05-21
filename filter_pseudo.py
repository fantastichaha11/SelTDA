"""Entry point: load raw synthetic JSON, score, gate, write filtered + report."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import cli
from filtering.adapters import BlipStudentAdapter, OpenClipAdapter
from filtering.gates import GateThresholds, apply_gates, thresholds_from_quantile
from filtering.io import dump_records, load_records
from filtering.report import build_filter_report
from filtering.scorers import (
    normalize_min_max,
    score_clip_itm,
    score_confidence,
    score_xcons,
)

logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_image(image_root: Path, image_field: str) -> Path:
    return image_root / image_field


def _run_gate_conf(records, config) -> float:
    if not config.gates.conf.enabled:
        for r in records:
            r.setdefault("scores", {})["conf"] = 1.0
        return float("-inf")

    raw = [score_confidence(r) for r in records]
    normalized = normalize_min_max(raw)
    for r, v in zip(records, normalized):
        r.setdefault("scores", {})["conf"] = 0.0 if v is None else float(v)

    return thresholds_from_quantile(
        [r["scores"]["conf"] for r in records],
        keep_top=config.gates.conf.keep_top,
    )


def _run_gate_itm(records, image_root: Path, config) -> float:
    if not config.gates.itm.enabled:
        for r in records:
            r.setdefault("scores", {})["itm"] = 1.0
        return float("-inf")

    clip = OpenClipAdapter(
        model_name=config.gates.itm.clip_model,
        pretrained=config.gates.itm.clip_pretrained,
        device=config.device,
    )
    for r in tqdm(records, desc="Gate ITM"):
        try:
            image = Image.open(_resolve_image(image_root, r["image"])).convert("RGB")
            s = score_clip_itm(r, image, clip)
        except Exception as e:
            logger.warning("ITM scoring failed for %s: %s", r.get("image"), e)
            s = 0.0
        r.setdefault("scores", {})["itm"] = float(s)

    return thresholds_from_quantile(
        [r["scores"]["itm"] for r in records],
        keep_top=config.gates.itm.keep_top,
    )


def _run_gate_xcons(records, image_root: Path, config) -> float:
    if not config.gates.xcons.enabled:
        for r in records:
            r.setdefault("scores", {})["xcons"] = 1.0
        return float("-inf")

    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer(config.gates.xcons.sbert_model, device=config.device)
    student = BlipStudentAdapter(
        checkpoint=config.gates.xcons.student_ckpt,
        med_config=config.gates.xcons.get("med_config", "configs/med_config.json"),
        image_size=config.gates.xcons.image_size,
        device=config.device,
    )
    for r in tqdm(records, desc="Gate X-cons"):
        try:
            image = Image.open(_resolve_image(image_root, r["image"])).convert("RGB")
            s = score_xcons(r, image, student, sbert)
        except Exception as e:
            logger.warning("X-cons scoring failed for %s: %s", r.get("image"), e)
            s = 0.0
        r.setdefault("scores", {})["xcons"] = float(s)

    return thresholds_from_quantile(
        [r["scores"]["xcons"] for r in records],
        keep_top=config.gates.xcons.keep_top,
    )


def main(args, config):
    _seed_everything(config.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    image_root = Path(config.image_root)
    records = load_records(config.input)
    logger.info("Loaded %d raw records from %s", len(records), config.input)

    tau_conf = _run_gate_conf(records, config)
    tau_itm = _run_gate_itm(records, image_root, config)
    tau_xcons = _run_gate_xcons(records, image_root, config)
    thresholds = GateThresholds(tau_conf, tau_itm, tau_xcons)
    logger.info("Thresholds: %s", thresholds)

    decisions = []
    kept_records = []
    for r in records:
        if config.scoring_only:
            decisions.append((r, "kept"))
            kept_records.append(r)
            continue
        keep, reason = apply_gates(r["scores"], thresholds)
        decisions.append((r, reason))
        if keep:
            kept_records.append(r)

    dump_records(kept_records, config.output)
    logger.info("Wrote %d kept records to %s", len(kept_records), config.output)

    report = build_filter_report(decisions, max_examples=config.report_max_examples)
    report["thresholds"] = {"conf": tau_conf, "itm": tau_itm, "xcons": tau_xcons}
    report["config"] = dict(config)
    Path(config.report).parent.mkdir(parents=True, exist_ok=True)
    with open(config.report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Wrote report to %s", config.report)


if __name__ == "__main__":
    args, config = cli.parse_args(default_config_path="./configs/filter_pseudo.yaml")
    cli.setup(args, config)
    main(args, config)
