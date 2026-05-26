"""Entry point: load raw synthetic JSON, score, gate, write filtered + report."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import cli
from filtering.adapters import BlipStudentAdapter, OpenClipAdapter
from filtering.coreset import select_coreset
from filtering.gate_registry import apply_cascade, enabled_gate_names
from filtering.gates import thresholds_from_quantile
from filtering.io import dump_records, load_records
from filtering.report import build_filter_report
from filtering.scorers import (
    _format_qa_for_clip,
    normalize_min_max,
    score_clip_itm,
    score_confidence,
    score_xcons,
)
from filtering.strata import assign_types, thresholds_per_stratum

logger = logging.getLogger(__name__)

ThresholdMap = Union[float, dict[str, float]]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_image(image_root: Path, image_field: str) -> Path:
    return image_root / image_field


def _scores(record: dict) -> dict:
    if record.get("scores") is None:
        record["scores"] = {}
    return record["scores"]


def _stratify_cfg(config) -> dict | None:
    if not hasattr(config, "stratify"):
        return None
    return dict(config.stratify) if config.stratify is not None else None


def _coreset_cfg(config) -> dict | None:
    if not hasattr(config, "coreset"):
        return None
    return dict(config.coreset) if config.coreset is not None else None


def _run_gate_conf(records, config) -> float:
    if not config.gates.conf.enabled:
        for r in records:
            _scores(r)["conf"] = 1.0
        return float("-inf")

    raw = [score_confidence(r) for r in records]
    normalized = normalize_min_max(raw)
    for r, v in zip(records, normalized):
        _scores(r)["conf"] = 0.0 if v is None else float(v)

    return thresholds_from_quantile(
        [r["scores"]["conf"] for r in records],
        keep_top=config.gates.conf.keep_top,
    )


def _run_gate_itm(records, image_root: Path, config) -> float:
    if not config.gates.itm.enabled:
        for r in records:
            _scores(r)["itm"] = 1.0
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
        _scores(r)["itm"] = float(s)

    return thresholds_from_quantile(
        [r["scores"]["itm"] for r in records],
        keep_top=config.gates.itm.keep_top,
    )


def _run_gate_xcons(records, image_root: Path, config) -> float:
    if not config.gates.xcons.enabled:
        for r in records:
            _scores(r)["xcons"] = 1.0
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
        _scores(r)["xcons"] = float(s)

    return thresholds_from_quantile(
        [r["scores"]["xcons"] for r in records],
        keep_top=config.gates.xcons.keep_top,
    )


def _compute_thresholds(records, config) -> dict[str, ThresholdMap]:
    stratify = _stratify_cfg(config)
    stratify_on = bool(stratify and stratify.get("enabled", False))
    if stratify_on:
        assign_types(records)

    gate_runners = {
        "conf": lambda: _run_gate_conf(records, config),
        "itm": lambda: _run_gate_itm(records, Path(config.image_root), config),
        "xcons": lambda: _run_gate_xcons(records, Path(config.image_root), config),
    }
    thresholds: dict[str, ThresholdMap] = {}
    global_fallback: float | None = None

    for gate in enabled_gate_names(config):
        gate_runners[gate]()
        gate_cfg = getattr(config.gates, gate)
        if stratify_on and global_fallback is None and gate == "conf":
            conf_vals = [r["scores"]["conf"] for r in records if "conf" in r.get("scores", {})]
            if conf_vals:
                global_fallback = thresholds_from_quantile(
                    conf_vals,
                    keep_top=stratify.get("global_keep_top_fallback", 0.75),
                )
        if stratify_on:
            thresholds[gate] = thresholds_per_stratum(
                records,
                gate=gate,
                keep_top=gate_cfg.keep_top,
                min_stratum_size=stratify.get("min_stratum_size", 50),
                global_fallback=global_fallback,
            )
        else:
            thresholds[gate] = thresholds_from_quantile(
                [r["scores"][gate] for r in records],
                keep_top=gate_cfg.keep_top,
            )
    return thresholds


def _record_thresholds(
    record: dict, thresholds: dict[str, ThresholdMap], gate_order: list[str]
) -> dict[str, float]:
    qtype = record.get("question_type", "visual_reasoning")
    out: dict[str, float] = {}
    for gate in gate_order:
        tau = thresholds[gate]
        if isinstance(tau, dict):
            out[gate] = tau.get(qtype, tau.get("visual_reasoning", float("-inf")))
        else:
            out[gate] = tau
    return out


def _apply_coreset(
    kept_records: list[dict],
    image_root: Path,
    config,
) -> list[dict]:
    coreset = _coreset_cfg(config)
    if not coreset or not coreset.get("enabled", False):
        return kept_records

    clip = OpenClipAdapter(
        model_name=config.gates.itm.clip_model,
        pretrained=config.gates.itm.clip_pretrained,
        device=config.device,
    )
    keep_ratio = float(coreset.get("keep_ratio", 0.9))
    diversity_weight = float(coreset.get("diversity_weight", 0.5))

    assign_types(kept_records)
    by_type: dict[str, list[dict]] = {}
    for r in kept_records:
        by_type.setdefault(r["question_type"], []).append(r)

    selected: list[dict] = []
    for stratum_records in by_type.values():
        if not stratum_records:
            continue
        embeddings = []
        for r in stratum_records:
            image = Image.open(_resolve_image(image_root, r["image"])).convert("RGB")
            text = _format_qa_for_clip(r)
            img_emb = np.asarray(clip.embed_image(image), dtype=np.float32).reshape(-1)
            txt_emb = np.asarray(clip.embed_text(text), dtype=np.float32).reshape(-1)
            embeddings.append(np.concatenate([img_emb, txt_emb]))
        emb_matrix = np.stack(embeddings, axis=0)
        budget = max(1, int(len(stratum_records) * keep_ratio))
        indices = select_coreset(emb_matrix, budget=budget, diversity_weight=diversity_weight)
        selected.extend(stratum_records[i] for i in indices)
    return selected


def main(args, config):
    _seed_everything(config.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    image_root = Path(config.image_root)
    records = load_records(config.input)
    logger.info("Loaded %d raw records from %s", len(records), config.input)

    thresholds = _compute_thresholds(records, config)
    gate_order = enabled_gate_names(config)
    logger.info("Gate order: %s; thresholds: %s", gate_order, thresholds)

    decisions = []
    kept_records = []
    for r in records:
        if config.scoring_only or not gate_order:
            decisions.append((r, "kept"))
            kept_records.append(r)
            continue
        _scores(r)
        th = _record_thresholds(r, thresholds, gate_order)
        keep, reason = apply_cascade(r["scores"], th, gate_order)
        decisions.append((r, reason))
        if keep:
            kept_records.append(r)

    kept_records = _apply_coreset(kept_records, image_root, config)

    dump_records(kept_records, config.output)
    logger.info("Wrote %d kept records to %s", len(kept_records), config.output)

    report = build_filter_report(decisions, max_examples=config.report_max_examples)
    report["thresholds"] = thresholds
    report["config"] = dict(config)
    Path(config.report).parent.mkdir(parents=True, exist_ok=True)
    with open(config.report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Wrote report to %s", config.report)


if __name__ == "__main__":
    args, config = cli.parse_args(default_config_path="./configs/filter_pseudo.yaml")
    cli.setup(args, config)
    main(args, config)
