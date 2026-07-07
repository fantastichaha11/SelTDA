"""Entry point: load raw synthetic JSON, score, gate, write filtered + report."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Union

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

import cli
from filtering.adapters import BlipStudentAdapter, OpenClipAdapter, VQAScoreAdapter
from filtering.coreset import select_coreset
from filtering.image_cache import ImageCache
from filtering.gate_registry import apply_cascade, enabled_gate_names
from filtering.gates import (
    fused_score,
    normalize_gate_scores,
    thresholds_from_quantile,
)
from filtering.io import dump_records, load_records
from filtering.paths import resolve_dataset_path
from filtering.report import build_filter_report
from filtering.score_cache import (
    apply_gate_cache,
    load_gate_cache,
    record_key,
    records_missing_gate,
    save_gate_cache,
)
from filtering.scorers import (
    _format_qa_for_clip,
    normalize_min_max,
    score_clip_itm_from_image_emb,
    score_confidence,
    score_xcons,
)
from filtering.scorers_vqascore import score_vqascore
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


def _plain_cfg_value(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _fusion_cfg(config) -> dict | None:
    if not hasattr(config, "fusion"):
        return None
    return dict(config.fusion) if config.fusion is not None else None


def _fusion_enabled(config) -> bool:
    fusion = _fusion_cfg(config)
    return bool(fusion and fusion.get("enabled", False))


def _score_cache_cfg(config) -> dict | None:
    if not hasattr(config, "score_cache"):
        return None
    if config.score_cache is None:
        return None
    return dict(config.score_cache)


def _score_cache_dir(config) -> Path | None:
    cache = _score_cache_cfg(config)
    if not cache:
        return None
    raw = cache.get("dir")
    if not raw:
        return None
    return Path(raw)


def _score_cache_save_enabled(config) -> bool:
    cache = _score_cache_cfg(config)
    if not cache:
        return False
    return bool(cache.get("save", True))


def _persist_gate_cache(
    records: list[dict],
    gate: str,
    config,
    input_path: Path,
) -> None:
    if not _score_cache_save_enabled(config):
        return
    cache_dir = _score_cache_dir(config)
    extras = None
    if gate == "xcons":
        extras = {
            record_key(r): {"_student_answer": r["_student_answer"]}
            for r in records
            if r.get("_student_answer") is not None
        }
    save_gate_cache(cache_dir, input_path, gate, records, extras=extras)


def _hydrate_gate_cache(
    records: list[dict],
    gate: str,
    config,
    input_path: Path,
) -> int:
    cache_dir = _score_cache_dir(config)
    payload = load_gate_cache(
        cache_dir, input_path, gate, expected_n=len(records)
    )
    if not payload:
        return 0
    return apply_gate_cache(records, gate, payload)


def _resolve_fusion_weights(config, gate_order: list[str]) -> dict[str, float]:
    fusion = _fusion_cfg(config) or {}
    raw = dict(fusion.get("weights") or {})
    weights: dict[str, float] = {}
    if raw:
        for gate in gate_order:
            if gate in raw:
                weights[gate] = float(raw[gate])
    else:
        weights = {gate: 1.0 for gate in gate_order}
    total = sum(weights.values())
    if total <= 0:
        return {gate: 1.0 / len(gate_order) for gate in gate_order} if gate_order else {}
    return {gate: w / total for gate, w in weights.items()}


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

    to_score = records_missing_gate(records, "itm")
    if not to_score:
        logger.info("[skip] ITM scoring — all records already have scores")
        return thresholds_from_quantile(
            [r["scores"]["itm"] for r in records],
            keep_top=config.gates.itm.keep_top,
        )

    clip = OpenClipAdapter(
        model_name=config.gates.itm.clip_model,
        pretrained=config.gates.itm.clip_pretrained,
        device=config.device,
    )
    batch_size = _gate_batch_size(config, "itm", 32)
    for start in tqdm(range(0, len(to_score), batch_size), desc="Gate ITM"):
        batch = to_score[start : start + batch_size]
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


def _run_gate_vqascore(
    records, image_root: Path, config
) -> float:
    if not config.gates.vqascore.enabled:
        for r in records:
            _scores(r)["vqascore"] = 1.0
        return float("-inf")

    to_score = records_missing_gate(records, "vqascore")
    if not to_score:
        logger.info("[skip] VQAScore scoring — all records already have scores")
        return thresholds_from_quantile(
            [r["scores"]["vqascore"] for r in records],
            keep_top=config.gates.vqascore.keep_top,
        )

    vqascore_cfg = config.gates.vqascore
    scorer_kwargs = {
        "backend": vqascore_cfg.get("backend", "t2v"),
        "model": vqascore_cfg.get("model", "clip-flant5-xl"),
        "device": config.device,
    }
    for key in (
        "processor",
        "model_class",
        "prompt_template",
        "image_token",
        "score_mode",
        "yes_tokens",
        "no_tokens",
        "trust_remote_code",
        "torch_dtype",
        "device_map",
        "model_kwargs",
        "processor_kwargs",
    ):
        if key in vqascore_cfg and vqascore_cfg.get(key) is not None:
            scorer_kwargs[key] = _plain_cfg_value(vqascore_cfg.get(key))
    scorer = VQAScoreAdapter(**scorer_kwargs)
    batch_size = _gate_batch_size(config, "vqascore", 8)
    for start in tqdm(range(0, len(to_score), batch_size), desc="Gate VQAScore"):
        batch = to_score[start : start + batch_size]
        try:
            image_paths = [
                str(_resolve_image(image_root, r["image"])) for r in batch
            ]
            texts = [_format_qa_for_clip(r) for r in batch]
            batch_scores = scorer.score_pairs(image_paths, texts)
            for r, s in zip(batch, batch_scores):
                _scores(r)["vqascore"] = float(s)
        except Exception as e:
            logger.warning("VQAScore batch failed at %d: %s", start, e)
            for r in batch:
                try:
                    image_path = _resolve_image(image_root, r["image"])
                    s = score_vqascore(r, image_path, scorer)
                except Exception as inner:
                    logger.warning(
                        "VQAScore scoring failed for %s: %s", r.get("image"), inner
                    )
                    s = 0.0
                _scores(r)["vqascore"] = float(s)

    return thresholds_from_quantile(
        [r["scores"]["vqascore"] for r in records],
        keep_top=config.gates.vqascore.keep_top,
    )


def _run_gate_xcons(records, image_cache: ImageCache, config) -> float:
    if not config.gates.xcons.enabled:
        for r in records:
            _scores(r)["xcons"] = 1.0
        return float("-inf")

    to_score = records_missing_gate(records, "xcons")
    if not to_score:
        logger.info("[skip] X-cons scoring — all records already have scores")
        return thresholds_from_quantile(
            [r["scores"]["xcons"] for r in records],
            keep_top=config.gates.xcons.keep_top,
        )

    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer(config.gates.xcons.sbert_model, device=config.device)
    student = BlipStudentAdapter(
        checkpoint=config.gates.xcons.student_ckpt,
        med_config=config.gates.xcons.get("med_config", "configs/med_config.json"),
        image_size=config.gates.xcons.image_size,
        device=config.device,
    )
    batch_size = _gate_batch_size(config, "xcons", 8)
    for start in tqdm(range(0, len(to_score), batch_size), desc="Gate X-cons"):
        batch = to_score[start : start + batch_size]
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
    corruption = config.gates.lp.get("corruption", "gaussian_noise")
    to_score = records_missing_gate(records, "lp")
    if not to_score:
        logger.info("[skip] LP scoring — all records already have scores")
        return thresholds_from_quantile(
            [r["scores"]["lp"] for r in records],
            keep_top=config.gates.lp.keep_top,
        )

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
    for r in tqdm(to_score, desc="Gate LP"):
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

    to_score = [
        r
        for r in records
        if "kcons" not in (r.get("scores") or {})
        and (not strata_filter or r.get("question_type") in strata_filter)
    ]
    if not to_score:
        for r in records:
            if strata_filter and r.get("question_type") not in strata_filter:
                _scores(r)["kcons"] = 1.0
        logger.info("[skip] K-cons scoring — all records already have scores")
        return thresholds_from_quantile(
            [r["scores"]["kcons"] for r in records],
            keep_top=config.gates.kcons.keep_top,
        )

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
        if "kcons" in _scores(r):
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


def _compute_thresholds(records, config, input_path: Path) -> dict[str, ThresholdMap]:
    stratify = _stratify_cfg(config)
    stratify_on = bool(stratify and stratify.get("enabled", False))
    if stratify_on:
        assign_types(records)

    image_root = Path(config.image_root)
    image_cache = ImageCache(image_root)
    gate_runners = {
        "conf": lambda: _run_gate_conf(records, config),
        "itm": lambda: _run_gate_itm(records, image_cache, config),
        "vqascore": lambda: _run_gate_vqascore(records, image_root, config),
        "xcons": lambda: _run_gate_xcons(records, image_cache, config),
        "lp": lambda: _run_gate_lp(records, image_cache, config),
        "kcons": lambda: _run_gate_kcons(records, config),
    }
    thresholds: dict[str, ThresholdMap] = {}
    global_fallback: float | None = None

    for gate in enabled_gate_names(config):
        _hydrate_gate_cache(records, gate, config, input_path)
        gate_runners[gate]()
        _persist_gate_cache(records, gate, config, input_path)
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
        elif not _fusion_enabled(config):
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


def _prepare_fusion(
    records: list[dict],
    gate_order: list[str],
    config,
) -> tuple[float, dict[str, float], str]:
    """Normalize per-gate scores on the pool and attach fused score to each record."""
    fusion = _fusion_cfg(config) or {}
    mode = str(fusion.get("normalize", "minmax"))
    keep_top = float(fusion.get("keep_top", 0.75))
    weights = _resolve_fusion_weights(config, gate_order)

    normed_by_gate: dict[str, list[float]] = {}
    for gate in gate_order:
        vals = [float(r["scores"].get(gate, 0.0)) for r in records]
        normed_by_gate[gate] = normalize_gate_scores(vals, mode)

    fused_vals: list[float] = []
    for i, record in enumerate(records):
        normalized = {gate: normed_by_gate[gate][i] for gate in gate_order}
        score = fused_score(normalized, weights, gate_order)
        record["scores"]["fusion"] = score
        fused_vals.append(score)

    tau = thresholds_from_quantile(fused_vals, keep_top)
    return tau, weights, mode


def _resolve_config_paths(config) -> None:
    for key in ("input", "image_root", "output", "report"):
        if hasattr(config, key) and config[key] is not None:
            config[key] = str(resolve_dataset_path(config[key]))
    cache_cfg = _score_cache_cfg(config)
    if cache_cfg and cache_cfg.get("dir"):
        cache_cfg["dir"] = str(resolve_dataset_path(cache_cfg["dir"]))
    scored_pool = (_score_cache_cfg(config) or {}).get("scored_pool")
    if scored_pool:
        if hasattr(config, "score_cache") and config.score_cache is not None:
            config.score_cache.scored_pool = str(resolve_dataset_path(scored_pool))


def main(args, config):
    _seed_everything(config.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    _resolve_config_paths(config)

    image_root = Path(config.image_root)
    input_path = Path(config.input)
    records = load_records(input_path)
    logger.info("Loaded %d raw records from %s", len(records), input_path)

    cache_cfg = _score_cache_cfg(config)
    if cache_cfg and cache_cfg.get("merge_input_scores", True):
        prefilled = sum(1 for r in records if r.get("scores"))
        if prefilled:
            logger.info(
                "Input already contains scores on %d/%d records",
                prefilled,
                len(records),
            )

    thresholds = _compute_thresholds(records, config, input_path)
    gate_order = enabled_gate_names(config)
    fusion_enabled = _fusion_enabled(config)
    if fusion_enabled and _stratify_cfg(config) and _stratify_cfg(config).get("enabled"):
        logger.warning(
            "fusion.enabled=true ignores per-stratum cascade thresholds; using global fusion only"
        )

    fusion_meta: dict | None = None
    tau_fusion: float | None = None
    if fusion_enabled and gate_order:
        tau_fusion, fusion_weights, fusion_norm = _prepare_fusion(
            records, gate_order, config
        )
        fusion_meta = {
            "tau": tau_fusion,
            "weights": fusion_weights,
            "normalize": fusion_norm,
            "keep_top": float(_fusion_cfg(config).get("keep_top", 0.75)),
        }
        logger.info("Fusion enabled: %s", fusion_meta)
    else:
        logger.info("Gate order: %s; thresholds: %s", gate_order, thresholds)

    scored_pool = (_score_cache_cfg(config) or {}).get("scored_pool")
    if scored_pool:
        dump_records(records, scored_pool)
        logger.info("Wrote scored pool (%d records) to %s", len(records), scored_pool)

    decisions = []
    kept_records = []
    for r in records:
        if config.scoring_only or not gate_order:
            decisions.append((r, "kept"))
            kept_records.append(r)
            continue
        _scores(r)
        if fusion_enabled and tau_fusion is not None:
            fused = float(r["scores"].get("fusion", 0.0))
            keep = fused >= tau_fusion
            reason = "kept" if keep else "fusion"
        else:
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
    if fusion_meta is not None:
        report["fusion"] = fusion_meta
    cache_dir = _score_cache_dir(config)
    if cache_dir is not None:
        report["score_cache"] = {"dir": str(cache_dir)}
    report["config"] = dict(config)
    Path(config.report).parent.mkdir(parents=True, exist_ok=True)
    with open(config.report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Wrote report to %s", config.report)


if __name__ == "__main__":
    args, config = cli.parse_args(default_config_path="./configs/filter_pseudo.yaml")
    cli.setup(args, config)
    main(args, config)
