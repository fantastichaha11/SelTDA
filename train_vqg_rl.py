"""GRPO trainer for the BLIP decoder teacher (RL-1).

Only the math helpers (group_advantages, grpo_loss) are unit-tested. `main`
wires the BLIP teacher + reference model and is exercised by the smoke test.
Does NOT modify train_vqa.py / train_vqg.py.
"""

from __future__ import annotations

import torch


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
            dataset_origin=VQADatasetOrigin.visual_genome,
            parse_rationale=False,
        )
        ans = rec.answer[0] if isinstance(rec.answer, list) else rec.answer
        return rec.question, ans
    except Exception:
        return model_output, ""


def train_one_epoch(policy, ref, loader, reward_fn, optimizer, cfg, device):
    """One GRPO epoch. `reward_fn(image_path, question, answer) -> float`."""
    policy.train()
    epoch_rewards = []
    for images, image_paths in loader:
        images = images.to(device)
        for i in range(images.size(0)):
            img = images[i : i + 1].repeat(cfg.group_size, 1, 1, 1)
            outputs, logp = policy.generate(
                img,
                sample=True,
                top_p=cfg.top_p,
                max_length=cfg.max_length,
                min_length=cfg.min_length,
                return_logprob=True,
            )
            with torch.no_grad():
                _, ref_logp = ref.generate(
                    img,
                    sample=False,
                    max_length=cfg.max_length,
                    min_length=cfg.min_length,
                    return_logprob=True,
                )
            rewards = torch.tensor(
                [reward_fn(image_paths[i], *_parse_qa(o)) for o in outputs],
                device=device,
                dtype=torch.float32,
            )
            adv = group_advantages(rewards)
            loss = grpo_loss(
                torch.as_tensor(logp, device=device),
                torch.as_tensor(ref_logp, device=device),
                adv,
                kl_beta=cfg.kl_beta,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_rewards.extend(rewards.tolist())
    return sum(epoch_rewards) / max(1, len(epoch_rewards))
