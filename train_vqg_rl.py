"""GRPO trainer for the BLIP decoder teacher (RL-1).

Only the math helpers (group_advantages, grpo_loss) are unit-tested. `main`
wires the BLIP teacher + reference model and is exercised by the smoke test.
Does NOT modify train_vqa.py / train_vqg.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Group-relative advantage: (r - mean) / (std + eps) within the group."""
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + eps)


def grpo_loss(
    logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    kl_beta: float,
) -> torch.Tensor:
    """GRPO objective: -E[A * logp] + beta * KL(policy || ref)."""
    pg = -(advantages.detach() * logp).mean()
    log_ratio = logp - ref_logp
    kl = (torch.exp(log_ratio) - 1.0 - log_ratio).mean()
    return pg + kl_beta * kl


def _parse_qa(model_output: str) -> tuple[str, str]:
    """Reuse generate_questions parsing; fall back to naive split."""
    from generate_questions import VQARecord, VQADatasetOrigin

    try:
        rec = VQARecord.build_from_raw_model_output(
            model_output,
            "x",
            dataset_origin=VQADatasetOrigin.vqa,
            parse_rationale=False,
        )
        ans = rec.answer[0] if isinstance(rec.answer, list) else rec.answer
        return rec.question, ans
    except Exception:
        return model_output, ""


@dataclass
class StepStats:
    mean_reward: float
    steps: int
    peak_vram_mib: int = 0


def _decode_captions(policy, outputs) -> list[str]:
    captions = []
    for output in outputs:
        caption = policy.tokenizer.decode(output, skip_special_tokens=True)
        captions.append(caption[len(policy.prompt) :])
    return captions


@torch.no_grad()
def generate_caption(
    policy,
    image: torch.Tensor,
    *,
    device: str,
    top_p: float,
    max_length: int,
    min_length: int,
) -> str:
    """Sample one caption for audit / eval (no gradients)."""
    policy.eval()
    img = image.unsqueeze(0).to(device)
    image_embeds = policy.visual_encoder(img)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
    model_kwargs = {
        "encoder_hidden_states": image_embeds,
        "encoder_attention_mask": image_atts,
    }
    prompt = [policy.prompt]
    input_ids = policy.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    input_ids[:, 0] = policy.tokenizer.bos_token_id
    input_ids = input_ids[:, :-1]
    outputs = policy.text_decoder.generate(
        input_ids=input_ids,
        max_length=max_length,
        min_length=min_length,
        do_sample=True,
        top_p=top_p,
        num_return_sequences=1,
        eos_token_id=policy.tokenizer.sep_token_id,
        pad_token_id=policy.tokenizer.pad_token_id,
        repetition_penalty=1.1,
        **model_kwargs,
    )
    return _decode_captions(policy, outputs)[0]


def sample_with_logprob(
    policy,
    image: torch.Tensor,
    *,
    top_p: float,
    max_length: int,
    min_length: int,
) -> tuple[list[str], torch.Tensor]:
    """Sample captions and return differentiable mean log-probs (batch,)."""
    image_embeds = policy.visual_encoder(image)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
    model_kwargs = {
        "encoder_hidden_states": image_embeds,
        "encoder_attention_mask": image_atts,
    }
    prompt = [policy.prompt] * image.size(0)
    input_ids = policy.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
    input_ids[:, 0] = policy.tokenizer.bos_token_id
    input_ids = input_ids[:, :-1]

    outputs = policy.text_decoder.generate(
        input_ids=input_ids,
        max_length=max_length,
        min_length=min_length,
        do_sample=True,
        top_p=top_p,
        num_return_sequences=1,
        eos_token_id=policy.tokenizer.sep_token_id,
        pad_token_id=policy.tokenizer.pad_token_id,
        repetition_penalty=1.1,
        **model_kwargs,
    )
    captions = _decode_captions(policy, outputs)
    attn_mask = (outputs != policy.tokenizer.pad_token_id).long()
    labels = outputs.masked_fill(outputs == policy.tokenizer.pad_token_id, -100)
    decoder_out = policy.text_decoder(
        input_ids=outputs,
        attention_mask=attn_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        labels=labels,
        return_dict=True,
    )
    shift_logits = decoder_out.logits[:, :-1, :].contiguous()
    shift_labels = outputs[:, 1:].contiguous()
    shift_mask = attn_mask[:, 1:].contiguous().float()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    mean_lp = (gathered * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1.0)
    return captions, mean_lp


def ref_logprob(
    ref,
    image: torch.Tensor,
    *,
    top_p: float,
    max_length: int,
    min_length: int,
) -> torch.Tensor:
    """Frozen reference mean log-prob under the same sampling setup as policy."""
    with torch.no_grad():
        _, lp = sample_with_logprob(
            ref,
            image,
            top_p=top_p,
            max_length=max_length,
            min_length=min_length,
        )
    return lp


def group_advantages_batched(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-image group normalization. `rewards` shape (B, G)."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + eps)


def _expand_groups(images: torch.Tensor, group_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B*G, C, H, W) with each image repeated G times."""
    b = images.size(0)
    return (
        images.unsqueeze(1)
        .expand(b, group_size, -1, -1, -1)
        .reshape(b * group_size, *images.shape[1:])
    )


def _grpo_step_batch(
    policy,
    ref,
    images,
    image_paths,
    reward_fn,
    optimizer,
    cfg,
    device,
) -> list[float]:
    """One GRPO update over B images × G samples each."""
    ref_device = next(ref.parameters()).device
    b, g = images.size(0), cfg.group_size
    torch.cuda.reset_peak_memory_stats(device)
    img = _expand_groups(images, g)
    outputs, logp = sample_with_logprob(
        policy,
        img,
        top_p=cfg.top_p,
        max_length=cfg.max_length,
        min_length=cfg.min_length,
    )
    ref_logp = ref_logprob(
        ref,
        img.to(ref_device),
        top_p=cfg.top_p,
        max_length=cfg.max_length,
        min_length=cfg.min_length,
    ).to(device)

    reward_rows: list[list[float]] = []
    for bi in range(b):
        row = []
        for gi in range(g):
            idx = bi * g + gi
            out = outputs[idx]
            lp = float(logp[idx].detach().cpu())
            row.append(
                float(
                    reward_fn(
                        image_paths[bi],
                        *_parse_qa(out),
                        gen_logprob=lp,
                    )
                )
            )
        reward_rows.append(row)
    rewards = torch.tensor(reward_rows, device=device, dtype=torch.float32)
    adv = group_advantages_batched(rewards).reshape(-1)
    loss = grpo_loss(logp, ref_logp, adv, kl_beta=cfg.kl_beta)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    step_vram_mib = torch.cuda.max_memory_allocated(device) // (1024**2)
    rewards_out = [r for row in reward_rows for r in row]
    del outputs, logp, ref_logp, rewards, adv, loss, img
    torch.cuda.empty_cache()
    return rewards_out, step_vram_mib


def train_one_epoch(
    policy,
    ref,
    loader,
    reward_fn,
    optimizer,
    cfg,
    device,
    *,
    max_steps: int | None = None,
) -> StepStats:
    """One GRPO epoch. `reward_fn(image_path, question, answer) -> float`."""
    policy.train()
    epoch_rewards: list[float] = []
    steps = 0
    peak_vram_mib = 0

    for images, image_paths in loader:
        if max_steps is not None and steps >= max_steps:
            break
        images = images.to(device)
        batch_rewards, step_vram_mib = _grpo_step_batch(
            policy, ref, images, image_paths, reward_fn, optimizer, cfg, device
        )
        epoch_rewards.extend(batch_rewards)
        steps += 1
        peak_vram_mib = max(peak_vram_mib, step_vram_mib)
        if steps == 1 or steps % 10 == 0:
            logger.info(
                "GRPO step %d mean_reward=%.4f peak_vram~%d MiB (post-step free %d MiB)",
                steps,
                sum(batch_rewards) / max(1, len(batch_rewards)),
                step_vram_mib,
                torch.cuda.mem_get_info(device)[0] // (1024**2),
            )
    mean_r = sum(epoch_rewards) / max(1, len(epoch_rewards))
    return StepStats(mean_reward=mean_r, steps=steps, peak_vram_mib=peak_vram_mib)


@dataclass
class ProbeResult:
    batch_size: int
    group_size: int
    peak_vram_mib: int


def probe_grpo_batch(
    policy,
    device: str,
    *,
    image_size: int,
    max_images_per_step: int = 4,
    group_candidates: tuple[int, ...] = (32, 28, 24, 20, 16, 14, 12, 10, 8, 4, 2),
) -> ProbeResult:
    """Find largest (images_per_step, group_size) that fits policy+backward on GPU."""
    policy.train()
    adam_mib = (
        sum(p.numel() for p in policy.parameters() if p.requires_grad) * 8 // (1024**2)
    )
    _, total = torch.cuda.mem_get_info(device)
    budget_mib = int((total // (1024**2)) * 0.97)
    for g in group_candidates:
        for b in range(max_images_per_step, 0, -1):
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                dummy = torch.randn(b, 3, image_size, image_size, device=device)
                img = _expand_groups(dummy, g)
                _, logp = sample_with_logprob(
                    policy, img, top_p=0.92, max_length=30, min_length=5
                )
                adv = group_advantages_batched(
                    torch.ones(b, g, device=device)
                ).reshape(-1)
                loss = grpo_loss(logp, logp.detach(), adv, kl_beta=0.1)
                loss.backward()
                for p in policy.parameters():
                    if p.grad is not None:
                        p.grad = None
                peak_mib = torch.cuda.max_memory_allocated(device) // (1024**2)
                free, total = torch.cuda.mem_get_info(device)
                total_mib = total // (1024**2)
                if peak_mib + adam_mib > budget_mib:
                    del dummy, img, logp, adv, loss
                    torch.cuda.empty_cache()
                    logger.info(
                        "probe images_per_step=%d group_size=%d tight (peak %d + adam %d MiB)",
                        b,
                        g,
                        peak_mib,
                        adam_mib,
                    )
                    continue
                logger.info(
                    "probe images_per_step=%d group_size=%d OK (peak %d MiB, adam %d MiB, free %d/%d MiB)",
                    b,
                    g,
                    peak_mib,
                    adam_mib,
                    free // (1024**2),
                    total_mib,
                )
                del dummy, img, logp, adv, loss
                torch.cuda.empty_cache()
                return ProbeResult(batch_size=b, group_size=g, peak_vram_mib=peak_mib)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                logger.info("probe images_per_step=%d group_size=%d OOM", b, g)
    return ProbeResult(batch_size=1, group_size=group_candidates[-1], peak_vram_mib=0)


def _corrupt_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    return Image.new("RGB", img.size, (128, 128, 128))


class _ExactOnlySbert:
    def encode(self, texts, convert_to_numpy=True):
        import numpy as np

        return np.array([[1.0, 0.0] if t else [0.0, 1.0] for t in texts])


def _resolve_vqascore_device(config, reward_device: str) -> str | None:
    """Pick device for CLIP-FlanT5 VQAScore (default cuda when available)."""
    raw = str(config.reward.get("vqascore_device", "cuda"))
    if raw == "auto":
        if reward_device.startswith("cuda"):
            return reward_device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw if raw not in ("", "none") else None


def _resolve_reward_device(
    config, device: str, *, probe_peak_mib: int
) -> str:
    """Pick reward device; student+CLIP need headroom alongside policy autograd graph."""
    reward_device = str(config.reward.get("device", "cpu"))
    if reward_device != "auto":
        return reward_device
    if device == "cpu" or not torch.cuda.is_available():
        return "cpu"
    torch.cuda.empty_cache()
    _, total = torch.cuda.mem_get_info(device)
    total_mib = total // (1024**2)
    reserve_mib = int(config.reward.get("gpu_reserve_mib", 5500))
    slack_mib = int(config.reward.get("gpu_slack_mib", 800))
    if probe_peak_mib + reserve_mib + slack_mib <= total_mib:
        return device
    logger.info(
        "Reward on CPU (policy peak %d MiB + reserve %d MiB exceeds %d MiB GPU)",
        probe_peak_mib,
        reserve_mib,
        total_mib,
    )
    return "cpu"


def _vqascore_checkpoint(config) -> str | None:
    raw = config.reward.get("vqascore_checkpoint")
    if raw in (None, "", "null"):
        return None
    return str(raw)


def _build_vqascore_conf_reward_fn(config, *, device: str, probe_peak_mib: int = 0):
    """VQAScore-only GRPO: P(yes)-P(no) + teacher gen_logprob (gate 1 conf)."""
    from filtering.reward import (
        RewardConfig,
        RewardTerms,
        compose_reward,
        score_teacher_conf,
        score_vqascore_margin,
    )
    from filtering.vqascore_adapter import DEFAULT_TEMPLATE, VQAScoreAdapter

    reward_device = _resolve_reward_device(
        config, device, probe_peak_mib=probe_peak_mib
    )
    logger.info("Reward mode=vqascore_conf on %s", reward_device)

    reward_cfg = RewardConfig(
        w_type=0.0,
        w_itm=0.0,
        w_grounding=0.0,
        w_learnability=0.0,
        w_kl=0.0,
        w_repetition=0.0,
        w_vqa=float(config.reward.get("w_vqa", 1.0)),
        w_conf=float(config.reward.get("w_conf", 1.0)),
    )

    backend = str(config.reward.get("vqascore_backend", "t2v_metrics"))
    vqascore_adapter = VQAScoreAdapter(
        model=str(config.reward.get("vqascore_model", "clip-flant5-xl")),
        backend=backend,
        device=_resolve_vqascore_device(config, reward_device),
        template=str(config.reward.get("vqascore_template", DEFAULT_TEMPLATE)),
        checkpoint=_vqascore_checkpoint(config),
    )
    logger.info(
        "VQAScore conf reward: margin=yes-no w_vqa=%.3f w_conf=%.3f model=%s",
        reward_cfg.w_vqa,
        reward_cfg.w_conf,
        vqascore_adapter.model,
    )

    image_root = Path(config.data.image_folder)

    def _resolve(path: str):
        p = Path(path)
        if p.is_file():
            return str(p)
        return str(image_root / path)

    def _score_terms(
        image_path: str,
        question: str,
        answer: str,
        *,
        gen_logprob: float | None = None,
    ) -> tuple[float, dict]:
        img_path = _resolve(image_path)
        vqa = score_vqascore_margin(img_path, question, answer, vqascore_adapter)
        conf = score_teacher_conf(gen_logprob)
        terms = RewardTerms(vqascore=vqa, gen_logprob=conf)
        breakdown = {"vqascore_margin": vqa, "gen_logprob": conf}
        return compose_reward(terms, reward_cfg), breakdown

    def reward_fn(
        image_path: str,
        question: str,
        answer: str,
        *,
        gen_logprob: float | None = None,
    ) -> float:
        total, _ = _score_terms(
            image_path, question, answer, gen_logprob=gen_logprob
        )
        return total

    def audit_terms_fn(image_path: str, question: str, answer: str) -> tuple[float, dict]:
        return _score_terms(image_path, question, answer, gen_logprob=None)

    return reward_fn, reward_cfg, audit_terms_fn


def build_reward_fn(config, *, device: str, probe_peak_mib: int = 0):
    mode = str(config.reward.get("mode", "full")).lower().replace("-", "_")
    if mode in ("vqascore_conf", "vqascore_only"):
        return _build_vqascore_conf_reward_fn(
            config, device=device, probe_peak_mib=probe_peak_mib
        )

    from filtering.adapters import BlipStudentAdapter, OpenClipAdapter
    from filtering.reward import RewardConfig, RewardTerms, compose_reward
    from filtering.reward import (
        grounding,
        learnability,
        repetition_penalty,
        score_vqascore,
        type_match,
    )
    from filtering.scorers import score_clip_itm
    from filtering.vqascore_adapter import DEFAULT_TEMPLATE, VQAScoreAdapter

    reward_device = _resolve_reward_device(
        config, device, probe_peak_mib=probe_peak_mib
    )
    logger.info("Reward models on %s", reward_device)

    w_vqa = float(config.reward.get("w_vqa", 0.0))
    reward_cfg = RewardConfig(
        w_type=float(config.reward.w_type),
        w_itm=float(config.reward.w_itm),
        w_grounding=float(config.reward.w_grounding),
        w_learnability=float(config.reward.w_learnability),
        w_kl=float(config.reward.w_kl),
        w_repetition=float(config.reward.w_repetition),
        w_vqa=w_vqa,
    )

    vqascore_adapter = None
    if w_vqa > 0.0:
        backend = str(config.reward.get("vqascore_backend", "t2v_metrics"))
        if backend.lower() not in ("none", "disabled", ""):
            vqascore_adapter = VQAScoreAdapter(
                model=str(config.reward.get("vqascore_model", "clip-flant5-xl")),
                backend=backend,
                device=_resolve_vqascore_device(config, reward_device),
                template=str(
                    config.reward.get("vqascore_template", DEFAULT_TEMPLATE)
                ),
                checkpoint=_vqascore_checkpoint(config),
            )
            logger.info(
                "VQAScore reward: model=%s backend=%s w_vqa=%.3f",
                vqascore_adapter.model,
                vqascore_adapter.backend,
                w_vqa,
            )
        else:
            logger.warning("w_vqa=%.3f but vqascore_backend=none; VQAScore skipped", w_vqa)
    weak_path = Path(config.weak_types.out)
    if weak_path.exists():
        weak_types = set(json.loads(weak_path.read_text()).get("weak_types", []))
    else:
        weak_types = set(
            config.weak_types.get("default", ["how_many", "external_knowledge", "yes_no"])
        )

    clip = OpenClipAdapter(device=reward_device)
    student = BlipStudentAdapter(
        checkpoint=str(config.student.checkpoint),
        med_config=str(config.student.med_config),
        image_size=int(config.student.image_size),
        device=reward_device,
    )
    sbert = _ExactOnlySbert()
    seen_questions: list[str] = []
    image_root = Path(config.data.image_folder)

    def _resolve(path: str):
        p = Path(path)
        if p.is_file():
            return str(p)
        return str(image_root / path)

    def _score_terms(
        image_path: str, question: str, answer: str, seen: list[str]
    ) -> tuple[float, dict]:
        img_path = _resolve(image_path)
        record = {"question": question, "answer": answer}
        itm = score_clip_itm(record, img_path, clip)
        g = grounding(img_path, question, answer, student, _corrupt_image, sbert)
        lrn = learnability(img_path, question, student)
        rep = repetition_penalty(question, seen)
        vqa = 0.0
        if vqascore_adapter is not None:
            vqa = score_vqascore(img_path, question, answer, vqascore_adapter)
        terms = RewardTerms(
            type_match=type_match(question, answer, weak_types),
            itm=itm,
            grounding=g,
            learnability=lrn,
            kl=0.0,
            repetition=rep,
            vqascore=vqa,
        )
        breakdown = {
            "type_match": terms.type_match,
            "itm": terms.itm,
            "grounding": terms.grounding,
            "learnability": terms.learnability,
            "repetition": terms.repetition,
            "vqascore": terms.vqascore,
        }
        return compose_reward(terms, reward_cfg), breakdown

    def reward_fn(
        image_path: str,
        question: str,
        answer: str,
        *,
        gen_logprob: float | None = None,
    ) -> float:
        total, _ = _score_terms(image_path, question, answer, seen_questions)
        seen_questions.append(question)
        return total

    audit_seen: list[str] = []

    def audit_terms_fn(image_path: str, question: str, answer: str) -> tuple[float, dict]:
        total, breakdown = _score_terms(image_path, question, answer, audit_seen)
        audit_seen.append(question)
        return total, breakdown

    return reward_fn, reward_cfg, audit_terms_fn


def _cfg_namespace(config) -> SimpleNamespace:
    g = config.grpo
    return SimpleNamespace(
        group_size=int(g.group_size),
        batch_size=int(g.get("batch_size", 1)),
        top_p=float(g.top_p),
        max_length=int(g.max_length),
        min_length=int(g.min_length),
        kl_beta=float(g.kl_beta),
        lr=float(g.lr),
        epochs_per_round=int(g.epochs_per_round),
        steps_per_epoch=int(g.steps_per_epoch),
    )


def _dataset_annotations(config) -> str | None:
    """None → glob unlabeled (or all) images under image_folder; path → JSON list."""
    ann = config.data.get("annotations", None)
    if ann is None:
        return None
    s = str(ann).strip()
    if not s or s.lower() in ("none", "null"):
        return None
    return s


def build_loader(config, batch_size: int) -> DataLoader:
    from generate_questions import build_dataset_from_config, collate_safe

    from omegaconf import OmegaConf

    ds_cfg = OmegaConf.create(
        {
            "image_folder": str(config.data.image_folder),
            "annotations": _dataset_annotations(config),
            "truncate_to": config.data.get("truncate_to"),
            "image_size": int(config.teacher.image_size),
        }
    )
    ds = build_dataset_from_config(ds_cfg)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config.data.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_safe,
    )


def build_teacher(
    config, *, device: str, trainable: bool = True, pretrained: str | None = None
):
    from generate_questions import build_model_from_config

    from omegaconf import OmegaConf

    model_cfg = OmegaConf.create(
        {
            "pretrained": str(pretrained or config.teacher.pretrained),
            "multimodal_encoder_decoder_config": str(config.teacher.med_config),
            "image_size": int(config.teacher.image_size),
            "vit": str(config.teacher.vit),
            "prompt": str(config.teacher.prompt),
        }
    )
    model = build_model_from_config(model_cfg)
    model.to(device)
    if trainable:
        model.train()
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def _read_epoch_rows(
    log_path: Path, round_id: int, *, min_steps: int = 200
) -> list[dict]:
    if not log_path.exists():
        return []
    by_epoch: dict[int, dict] = {}
    for line in log_path.read_text().strip().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(row.get("round", -1)) != round_id:
            continue
        if int(row.get("steps", 0)) < min_steps:
            continue
        ep = int(row.get("epoch", 0))
        if ep not in by_epoch or int(row.get("steps", 0)) >= int(by_epoch[ep].get("steps", 0)):
            by_epoch[ep] = row
    return [by_epoch[e] for e in sorted(by_epoch)]


def _load_resume(
    policy,
    optimizer,
    ckpt_path: Path,
    *,
    device: str,
    start_epoch: int,
) -> int:
    if not ckpt_path.is_file():
        return start_epoch
    ckpt = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt and start_epoch > 0:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as exc:
            logger.warning("Could not load optimizer state: %s", exc)
    resumed = int(ckpt.get("epoch", -1)) + 1
    logger.info("Resumed policy from %s (checkpoint epoch %d)", ckpt_path, ckpt.get("epoch", -1))
    return max(start_epoch, resumed)


def _resolve_backfill_ckpt(
    epoch: int,
    round_id: int,
    out_dir: Path,
    config,
) -> Path | None:
    """Checkpoint for auditing a past epoch (may be approximate for epoch 0)."""
    epoch_ckpt = out_dir / f"teacher_{round_id}_epoch{epoch}.pth"
    if epoch_ckpt.is_file():
        return epoch_ckpt
    if epoch == 0:
        fb = Path(
            str(
                config.audit.get(
                    "backfill_epoch0_ckpt", config.teacher.pretrained
                )
            )
        )
        return fb if fb.is_file() else None
    if epoch == 1:
        legacy = out_dir / f"teacher_{round_id}.pth"
        if legacy.is_file():
            ckpt = torch.load(legacy, map_location="cpu")
            if int(ckpt.get("epoch", -1)) == 1:
                return legacy
    return None


def _backfill_missing_audits(
    policy,
    config,
    *,
    round_id: int,
    out_dir: Path,
    epoch_rows: list[dict],
    reward_terms_fn,
    device: str,
    cfg,
    hrp_state,
    reward_cfg,
) -> None:
    """Run epoch audits for past epochs missing from the audit log (e.g. 0, 1 after epoch 2)."""
    from filtering.audit import load_audit_history

    audit_log = out_dir / f"round_{round_id}_audit.jsonl"
    done = {int(a["epoch"]) for a in load_audit_history(audit_log)}
    targets = [int(e) for e in config.audit.get("backfill_epochs", [])]
    missing = sorted(e for e in targets if e not in done)
    if not missing:
        return

    mean_by_epoch = {int(r["epoch"]): float(r["mean_reward"]) for r in epoch_rows}
    policy_backup = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    prev_audit = load_audit_history(audit_log)
    prev_audit = prev_audit[-1] if prev_audit else None

    try:
        for ep in missing:
            ckpt_path = _resolve_backfill_ckpt(ep, round_id, out_dir, config)
            if ckpt_path is None:
                logger.warning("Backfill audit epoch %d: no checkpoint found", ep)
                continue
            ckpt = torch.load(ckpt_path, map_location=device)
            policy.load_state_dict(ckpt["model"])
            policy.eval()
            mean_r = mean_by_epoch.get(ep, 0.0)
            approximate = ep == 0 and ckpt_path.name != f"teacher_{round_id}_epoch0.pth"
            logger.info(
                "Backfill audit epoch %d from %s (mean_reward=%.4f, approximate=%s)",
                ep,
                ckpt_path,
                mean_r,
                approximate,
            )
            hrp_state, reward_cfg, cfg, _ = _run_post_epoch_audit(
                policy,
                config,
                epoch=ep,
                round_id=round_id,
                mean_reward=mean_r,
                reward_terms_fn=reward_terms_fn,
                device=device,
                cfg=cfg,
                out_dir=out_dir,
                prev_audit=prev_audit,
                hrp_state=hrp_state,
                reward_cfg=reward_cfg,
                backfill=True,
                backfill_ckpt=str(ckpt_path),
                approximate=approximate,
            )
            prev_audit = load_audit_history(audit_log)[-1]
    finally:
        policy.load_state_dict({k: v.to(device) for k, v in policy_backup.items()})
        policy.train()


def _run_post_epoch_audit(
    policy,
    config,
    *,
    epoch: int,
    round_id: int,
    mean_reward: float,
    reward_terms_fn,
    device: str,
    cfg,
    out_dir: Path,
    prev_audit: dict | None,
    hrp_state,
    reward_cfg,
    backfill: bool = False,
    backfill_ckpt: str | None = None,
    approximate: bool = False,
):
    from filtering.audit import (
        detect_hacking,
        load_audit_history,
        run_epoch_audit,
        term_gain_shares,
    )
    from orchestration.hrp import respond

    from generate_questions import build_dataset_from_config

    from omegaconf import OmegaConf

    audit_cfg = config.audit
    if not bool(audit_cfg.get("every_epoch", True)):
        return hrp_state, reward_cfg, cfg, False

    ds_cfg = OmegaConf.create(
        {
            "image_folder": str(config.data.image_folder),
            "annotations": _dataset_annotations(config),
            "truncate_to": config.data.get("truncate_to"),
            "image_size": int(config.teacher.image_size),
        }
    )
    dataset = build_dataset_from_config(ds_cfg)
    seed = int(audit_cfg.get("seed", 42)) + epoch

    policy_device = str(next(policy.parameters()).device)

    def _gen(pol, image_tensor, _dev):
        return generate_caption(
            pol,
            image_tensor,
            device=policy_device,
            top_p=cfg.top_p,
            max_length=cfg.max_length,
            min_length=cfg.min_length,
        )

    audit_device = str(audit_cfg.get("device", "cpu"))
    if audit_device == "auto":
        audit_device = "cpu"

    result = run_epoch_audit(
        policy,
        dataset,
        device=audit_device,
        n_audit=int(audit_cfg.get("n_audit", 200)),
        seed=seed,
        generate_fn=_gen,
        reward_terms_fn=reward_terms_fn,
        config=config,
    )

    weights = {
        "type_match": float(config.reward.w_type),
        "itm": float(config.reward.w_itm),
        "grounding": float(config.reward.w_grounding),
        "learnability": float(config.reward.w_learnability),
        "repetition": float(config.reward.w_repetition),
        "vqascore": float(config.reward.get("w_vqa", 0.0)),
        "gen_logprob": float(config.reward.get("w_conf", 0.0)),
    }
    term_share: dict[str, float] = {}
    d_reward = d_judge = 0.0
    judge_dropping = False
    if prev_audit:
        d_reward = mean_reward - float(prev_audit.get("mean_reward", mean_reward))
        d_judge = result.mean_judge - float(prev_audit.get("mean_judge", result.mean_judge))
        judge_dropping = d_judge < -float(audit_cfg.get("hacking_eps", 0.02))
        term_share = term_gain_shares(
            prev_audit.get("term_means", {}),
            result.term_means,
            weights,
        )

    hacking = detect_hacking(
        d_reward,
        d_judge,
        float(audit_cfg.get("hacking_eps", 0.02)),
        term_share,
        float(audit_cfg.get("itm_dominance_tau", 0.6)),
    )
    dominant = None
    if term_share:
        term, share = max(term_share.items(), key=lambda kv: kv[1])
        if share > float(audit_cfg.get("itm_dominance_tau", 0.6)):
            dominant = term

    if backfill:
        action = "backfill"
    else:
        hrp_state, action = respond(
            hrp_state,
            hacking=hacking,
            dominant_term=dominant if dominant == "itm" else None,
            judge_dropping=judge_dropping,
            auto_levels=list(config.hrp.get("auto_levels", [1, 2, 3])),
            kl_c=float(config.hrp.get("adaptive_kl_c", 1.5)),
            downweight_d=float(config.hrp.get("downweight_d", 0.5)),
        )
        cfg.kl_beta = hrp_state.kl_beta
        reward_cfg.w_itm = hrp_state.w_itm
        if prev_audit:
            judge_drop_eps = float(audit_cfg.get("stop_on_judge_drop_eps", 0.01))
            if (
                bool(audit_cfg.get("stop_on_judge_drop", False))
                and d_judge < -judge_drop_eps
            ):
                action = "early_stop_judge_drop"
            elif bool(audit_cfg.get("stop_on_reward_plateau", False)):
                reward_drop_eps = float(
                    audit_cfg.get("stop_on_reward_plateau_eps", 0.05)
                )
                if d_reward < -reward_drop_eps:
                    action = "early_stop_reward_plateau"

    audit_row = {
        "round": round_id,
        "epoch": epoch,
        "mean_reward": mean_reward,
        "mean_judge": result.mean_judge,
        "mean_itm_judge": result.mean_itm_judge,
        "mean_vqa_match": result.mean_vqa_match,
        "audit_mean_reward": result.mean_reward,
        "n_audit": result.n_samples,
        "term_means": result.term_means,
        "d_reward": d_reward,
        "d_judge": d_judge,
        "hacking": hacking,
        "hrp_action": action,
    }
    if backfill:
        audit_row["backfill"] = True
        if backfill_ckpt:
            audit_row["backfill_ckpt"] = backfill_ckpt
        if approximate:
            audit_row["approximate"] = True
    audit_log = out_dir / f"round_{round_id}_audit.jsonl"
    with audit_log.open("a") as f:
        f.write(json.dumps(audit_row) + "\n")
    logger.info(
        "Audit epoch %d: judge=%.4f (itm=%.4f vqa=%.4f) reward=%.4f hacking=%s hrp=%s",
        epoch,
        result.mean_judge,
        result.mean_itm_judge,
        result.mean_vqa_match,
        result.mean_reward,
        hacking,
        action,
    )
    early_stop = action in {
        "early_stop_rollback",
        "early_stop_judge_drop",
        "early_stop_reward_plateau",
    }
    return hrp_state, reward_cfg, cfg, early_stop


def ensure_weak_types(config) -> None:
    out = Path(config.weak_types.out)
    if out.exists():
        return
    eval_path = Path(config.weak_types.student_base_eval)
    out.parent.mkdir(parents=True, exist_ok=True)
    if eval_path.exists():
        from orchestration.weak_types import compute_and_save

        weak = compute_and_save(eval_path, out, k=int(config.weak_types.k))
        logger.info("Computed weak types from eval: %s", weak)
        return
    default = list(
        config.weak_types.get("default", ["how_many", "external_knowledge", "yes_no"])
    )
    out.write_text(json.dumps({"weak_types": default}, indent=2))
    logger.info("Using default weak types: %s", default)


def train_grpo_round(config, args) -> Path:
    ensure_weak_types(config)
    device = args.device
    cfg = _cfg_namespace(config)
    round_id = int(config.grpo.get("round", 1))
    from orchestration.hrp import HrpState
    from orchestration.rl_teacher_loop import kl_beta_for_round

    cfg.kl_beta = kl_beta_for_round(
        float(config.grpo.kl_beta),
        float(config.grpo.kl_beta_decay_gamma),
        round_id,
    )

    out_dir = Path(config.grpo.get("output_dir", "orchestration/state"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"teacher_{round_id}.pth"
    log_path = out_dir / f"round_{round_id}_epochs.jsonl"
    audit_log = out_dir / f"round_{round_id}_audit.jsonl"

    completed_epochs = _read_epoch_rows(
        log_path, round_id, min_steps=cfg.steps_per_epoch
    )
    start_epoch = len(completed_epochs)
    resume_ckpt = getattr(args, "resume", None)
    if resume_ckpt:
        resume_path = Path(resume_ckpt) if resume_ckpt != "auto" else ckpt_path
        if resume_path.is_file():
            ckpt_path = resume_path
    elif start_epoch > 0 and ckpt_path.is_file():
        logger.info("Resuming from %d completed epochs in %s", start_epoch, log_path)

    policy_pretrained = str(config.teacher.pretrained)
    if start_epoch > 0 and ckpt_path.is_file():
        policy_pretrained = str(ckpt_path)
        logger.info(
            "Continuing from epoch %d checkpoint %s (next epoch %d)",
            start_epoch - 1,
            ckpt_path,
            start_epoch,
        )
    else:
        logger.info("Loading policy teacher from %s", policy_pretrained)
    policy = build_teacher(
        config, device=device, trainable=True, pretrained=policy_pretrained
    )
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=cfg.lr
    )

    if start_epoch > 0:
        start_epoch = _load_resume(
            policy, optimizer, ckpt_path, device=device, start_epoch=start_epoch
        )
        if completed_epochs:
            last = completed_epochs[-1]
            cfg.batch_size = int(last.get("batch_size", cfg.batch_size))
            cfg.group_size = int(last.get("group_size", cfg.group_size))
            cfg.kl_beta = float(last.get("kl_beta", cfg.kl_beta))
            logger.info(
                "Resume: optimizer restored, batch=%d group=%d kl_beta=%.3f start_epoch=%d",
                cfg.batch_size,
                cfg.group_size,
                cfg.kl_beta,
                start_epoch,
            )

    probe_peak_mib = 0
    if start_epoch == 0 and config.grpo.get("auto_tune", True):
        max_b = int(config.grpo.get("max_images_per_step", config.grpo.get("batch_size", 4)))
        probe = probe_grpo_batch(
            policy,
            device,
            image_size=int(config.teacher.image_size),
            max_images_per_step=max_b,
        )
        cfg.batch_size = probe.batch_size
        cfg.group_size = probe.group_size
        probe_peak_mib = probe.peak_vram_mib
        logger.info(
            "Auto-tuned images_per_step=%d group_size=%d (samples/step=%d, peak_vram=%d MiB)",
            cfg.batch_size,
            cfg.group_size,
            cfg.batch_size * cfg.group_size,
            probe_peak_mib,
        )
    elif start_epoch == 0:
        cfg.batch_size = int(config.grpo.get("batch_size", 1))
        cfg.group_size = int(config.grpo.get("group_size", 8))
    elif probe_peak_mib == 0:
        probe_peak_mib = int(config.grpo.get("resume_probe_peak_mib", 0))
        if probe_peak_mib == 0:
            for row in reversed(completed_epochs):
                if int(row.get("steps", 0)) >= int(cfg.steps_per_epoch):
                    probe_peak_mib = int(row.get("peak_vram_mib", 0))
                    if probe_peak_mib:
                        break
        if probe_peak_mib == 0:
            probe_peak_mib = 20515
        logger.info(
            "Resume: probe_peak_mib=%d for reward device (skip probe)",
            probe_peak_mib,
        )

    use_ref_cpu = bool(config.grpo.get("ref_on_cpu", True))
    ref_device = "cpu" if use_ref_cpu else device
    logger.info("Loading frozen reference teacher on %s", ref_device)
    ref = build_teacher(config, device=ref_device, trainable=False)

    reward_fn, reward_cfg, audit_terms_fn = build_reward_fn(
        config, device=device, probe_peak_mib=probe_peak_mib
    )
    loader = build_loader(config, batch_size=cfg.batch_size)

    from filtering.audit import load_audit_history

    audit_history = load_audit_history(audit_log)
    prev_audit = audit_history[-1] if audit_history else None
    hrp_state = HrpState(kl_beta=cfg.kl_beta, w_itm=float(reward_cfg.w_itm))
    best_judge = -1.0
    if audit_history:
        best_judge = max(float(r.get("mean_judge", -1.0)) for r in audit_history)
    best_epoch_ckpt = out_dir / f"teacher_{round_id}_best.pth"

    if start_epoch >= cfg.epochs_per_round:
        logger.info("Round %d already complete (%d epochs)", round_id, start_epoch)
        return ckpt_path

    for epoch in range(start_epoch, cfg.epochs_per_round):
        stats = train_one_epoch(
            policy,
            ref,
            loader,
            reward_fn,
            optimizer,
            cfg,
            device,
            max_steps=cfg.steps_per_epoch,
        )
        epoch_ckpt = out_dir / f"teacher_{round_id}_epoch{epoch}.pth"
        torch.save(
            {
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "round": round_id,
            },
            epoch_ckpt,
        )
        torch.save(
            {
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "round": round_id,
            },
            ckpt_path,
        )

        audit_row = None
        early_stop = False
        if bool(config.audit.get("every_epoch", True)):
            hrp_state, reward_cfg, cfg, early_stop = _run_post_epoch_audit(
                policy,
                config,
                epoch=epoch,
                round_id=round_id,
                mean_reward=stats.mean_reward,
                reward_terms_fn=audit_terms_fn,
                device=device,
                cfg=cfg,
                out_dir=out_dir,
                prev_audit=prev_audit,
                hrp_state=hrp_state,
                reward_cfg=reward_cfg,
            )
            from filtering.audit import load_audit_history

            audit_history = load_audit_history(audit_log)
            prev_audit = audit_history[-1] if audit_history else None
            mj = float(prev_audit.get("mean_judge", -1)) if prev_audit else -1.0
            if mj > best_judge:
                best_judge = mj
                torch.save(
                    torch.load(epoch_ckpt, map_location="cpu"),
                    best_epoch_ckpt,
                )

        row = {
            "round": round_id,
            "epoch": epoch,
            "mean_reward": stats.mean_reward,
            "steps": stats.steps,
            "group_size": cfg.group_size,
            "batch_size": cfg.batch_size,
            "samples_per_step": cfg.batch_size * cfg.group_size,
            "kl_beta": cfg.kl_beta,
            "peak_vram_mib": stats.peak_vram_mib,
        }
        if prev_audit:
            row["mean_judge"] = prev_audit.get("mean_judge")
            row["hacking"] = prev_audit.get("hacking")
            row["hrp_action"] = prev_audit.get("hrp_action")
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        logger.info(
            "Round %d epoch %d: mean_reward=%.4f steps=%d group_size=%d",
            round_id,
            epoch,
            stats.mean_reward,
            stats.steps,
            cfg.group_size,
        )

        backfill_after = config.audit.get("backfill_after_epoch")
        if backfill_after is not None and epoch == int(backfill_after):
            _backfill_missing_audits(
                policy,
                config,
                round_id=round_id,
                out_dir=out_dir,
                epoch_rows=_read_epoch_rows(
                    log_path, round_id, min_steps=cfg.steps_per_epoch
                ),
                reward_terms_fn=audit_terms_fn,
                device=device,
                cfg=cfg,
                hrp_state=hrp_state,
                reward_cfg=reward_cfg,
            )
            from filtering.audit import load_audit_history

            audit_history = load_audit_history(audit_log)
            prev_audit = audit_history[-1] if audit_history else prev_audit

        if early_stop and bool(config.audit.get("rollback_to_best_judge", True)):
            rollback = best_epoch_ckpt if best_epoch_ckpt.is_file() else epoch_ckpt
            logger.info("HRP early stop: rolling back to %s", rollback)
            ckpt = torch.load(rollback, map_location=device)
            policy.load_state_dict(ckpt["model"])
            torch.save(
                {
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "round": round_id,
                },
                ckpt_path,
            )
            break

    return ckpt_path


def main(args, config) -> None:
    ckpt = train_grpo_round(config, args)
    logger.info("Saved RL teacher checkpoint to %s", ckpt)


if __name__ == "__main__":
    import cli

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args, config = cli.parse_args(default_config_path="./configs/rl_teacher_aokvqa.yaml")
    cli.setup(args, config)
    if args.resume and str(config.grpo.get("output_dir", "")):
        out = Path(config.grpo.get("output_dir", "orchestration/state"))
        if args.resume == "auto":
            args.resume = str(out / f"teacher_{int(config.grpo.get('round', 1))}.pth")
    main(args, config)
