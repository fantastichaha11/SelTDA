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
from filtering.image_cache import ImageCache
from filtering.gate_registry import apply_cascade, enabled_gate_names
from filtering.gates import thresholds_from_quantile
from filtering.io import dump_records, load_records
from filtering.report import build_filter_report
from filtering.scorers import (
    _format_qa_for_clip,
    normalize_min_max,
    score_clip_itm_from_image_emb,
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


def _gate_batch_size(config, gate: str, default: int) -> int:
    gate_cfg = getattr(config.gates, gate, None)
    if gate_cfg is None:
        return default
    raw = gate_cfg.get("batch_size", default) if hasattr(gate_cfg, "get") else getattr(
        gate_cfg, "batch_size", default
    )
    try:
        size = int(raw)
    except (TypeError, ValueError):
        return default
    return max(1, size)


def _strip_filter_internals(records: list[dict]) -> None:
    for r in records:
        r.pop("_student_answer", None)


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


def _run_gate_itm(records, image_cache: ImageCache, config) -> float:
    if not config.gates.itm.enabled:
        for r in records:
            _scores(r)["itm"] = 1.0
        return float("-inf")

    clip = OpenClipAdapter(
        model_name=config.gates.itm.clip_model,
        pretrained=config.gates.itm.clip_pretrained,
        device=config.device,
    )
    batch_size = _gate_batch_size(config, "itm", 32)
    for start in tqdm(range(0, len(records), batch_size), desc="Gate ITM"):
        batch = records[start : start + batch_size]
        try:
            images = [image_cache.get(r["image"]) for r in batch]
            img_embs = clip.embed_images_batch(images)
            for r, img_emb in zip(batch, img_embs):
                s = score_clip_itm_from_image_emb(r, img_emb, clip)
                _scores(r)["itm"] = float(s)
        except Exception as e:
            logger.warning("ITM batch failed at %d: %s", start, e)
            for r in batch:
                try:
                    image = image_cache.get(r["image"])
                    s = score_clip_itm_from_image_emb(
                        r, clip.embed_image(image), clip
                    )
                except Exception as inner:
                    logger.warning("ITM scoring failed for %s: %s", r.get("image"), inner)
                    s = 0.0
                _scores(r)["itm"] = float(s)

    return thresholds_from_quantile(
        [r["scores"]["itm"] for r in records],
        keep_top=config.gates.itm.keep_top,
    )


def _run_gate_xcons(records, image_cache: ImageCache, config) -> float:
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
    batch_size = _gate_batch_size(config, "xcons", 8)
    for start in tqdm(range(0, len(records), batch_size), desc="Gate X-cons"):
        batch = records[start : start + batch_size]
        try:
            images = [image_cache.get(r["image"]) for r in batch]
            questions = [r["question"] for r in batch]
            preds = student.answer_questions_batch(images, questions)
            for r, image, pred in zip(batch, images, preds):
                s, predicted = score_xcons(r, image, student, sbert, predicted=pred)
                r["_student_answer"] = predicted
                _scores(r)["xcons"] = float(s)
        except Exception as e:
            logger.warning("X-cons batch failed at %d: %s", start, e)
            for r in batch:
                try:
                    image = image_cache.get(r["image"])
                    s, predicted = score_xcons(r, image, student, sbert)
                    r["_student_answer"] = predicted
                except Exception as inner:
                    logger.warning("X-cons scoring failed for %s: %s", r.get("image"), inner)
                    s, predicted = 0.0, ""
                _scores(r)["xcons"] = float(s)

    return thresholds_from_quantile(
        [r["scores"]["xcons"] for r in records],
        keep_top=config.gates.xcons.keep_top,
    )


def _run_gate_lp(records, image_cache: ImageCache, config) -> float:
    from filtering.scorers_language_prior import score_language_prior

    if not config.gates.lp.enabled:
        for r in records:
            _scores(r)["lp"] = 1.0
        return float("-inf")

    lp_cfg = config.gates.lp
    xcons_cfg = getattr(config.gates, "xcons", None)
    student = BlipStudentAdapter(
        checkpoint=lp_cfg.get("student_ckpt") or getattr(
            xcons_cfg, "student_ckpt", "cache/student_weights/checkpoint_09.pth"
        ),
        med_config=lp_cfg.get("med_config")
        or getattr(xcons_cfg, "med_config", "configs/med_config.json"),
        image_size=lp_cfg.get("image_size") or getattr(xcons_cfg, "image_size", 384),
        device=config.device,
    )
    corruption = config.gates.lp.get("corruption", "gaussian_noise")
    for r in tqdm(records, desc="Gate LP"):
        try:
            a_clean = r.get("_student_answer")
            image = image_cache.get_copy(r["image"]) if a_clean is None else image_cache.get(
                r["image"]
            )
            s = score_language_prior(
                r,
                image,
                student,
                corruption=corruption,
                a_clean=a_clean,
            )
        except Exception as e:
            logger.warning("LP scoring failed for %s: %s", r.get("image"), e)
            s = 0.0
        _scores(r)["lp"] = float(s)

    return thresholds_from_quantile(
        [r["scores"]["lp"] for r in records],
        keep_top=config.gates.lp.keep_top,
    )


def _run_gate_kcons(records, config) -> float:
    from filtering.retrieval.wikipedia import retrieve_passages
    from filtering.scorers_knowledge import score_knowledge_consistency

    if not config.gates.kcons.enabled:
        for r in records:
            _scores(r)["kcons"] = 1.0
        return float("-inf")

    strata_filter = list(config.gates.kcons.get("strata_filter", []) or [])
    if strata_filter:
        assign_types(records)

    nli_model = None
    nli_name = config.gates.kcons.get("nli_model")
    if nli_name:
        from sentence_transformers import CrossEncoder

        nli_model = CrossEncoder(nli_name, device=config.device)

    cache_dir = None
    retrieval = config.gates.kcons.get("retrieval")
    if retrieval and retrieval.get("cache_dir"):
        cache_dir = Path(retrieval["cache_dir"])

    k = 3
    if retrieval and retrieval.get("k"):
        k = int(retrieval["k"])

    for r in tqdm(records, desc="Gate K-cons"):
        if strata_filter and r.get("question_type") not in strata_filter:
            _scores(r)["kcons"] = 1.0
            continue
        try:
            ans = r["answer"][0] if isinstance(r["answer"], list) else r["answer"]
            passages = retrieve_passages(
                r["question"], ans, k=k, cache_dir=cache_dir
            )
            if nli_model is None:
                s = 0.5
            else:
                s = score_knowledge_consistency(
                    r["question"], ans, passages, nli_model
                )
        except Exception as e:
            logger.warning("K-cons scoring failed for %s: %s", r.get("image"), e)
            s = 0.0
        _scores(r)["kcons"] = float(s)

    return thresholds_from_quantile(
        [r["scores"]["kcons"] for r in records],
        keep_top=config.gates.kcons.keep_top,
    )


def _compute_thresholds(records, config) -> dict[str, ThresholdMap]:
    stratify = _stratify_cfg(config)
    stratify_on = bool(stratify and stratify.get("enabled", False))
    if stratify_on:
        assign_types(records)

    image_root = Path(config.image_root)
    image_cache = ImageCache(image_root)
    gate_runners = {
        "conf": lambda: _run_gate_conf(records, config),
        "itm": lambda: _run_gate_itm(records, image_cache, config),
        "xcons": lambda: _run_gate_xcons(records, image_cache, config),
        "lp": lambda: _run_gate_lp(records, image_cache, config),
        "kcons": lambda: _run_gate_kcons(records, config),
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

    _strip_filter_internals(records)
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
