"""AUD-1: independent held-out judges + hacking detection (epoch-level)."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

BLIP_ITM_LARGE_URL = (
    "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/"
    "model_large.pth"
)


def judge_score(itm_large: float, vqa_match: float) -> float:
    """Aggregate the two independent judges (mean)."""
    return (float(itm_large) + float(vqa_match)) / 2.0


def detect_hacking(
    d_reward: float,
    d_judge: float,
    eps: float,
    term_share: dict[str, float],
    tau: float,
) -> bool:
    """Hacking if reward rises while judge falls past eps, OR a cheap term dominates gain."""
    if d_reward > 0 and d_judge < -eps:
        return True
    if any(share > tau for share in term_share.values()):
        return True
    return False


def term_gain_shares(
    prev_terms: dict[str, float],
    curr_terms: dict[str, float],
    weights: dict[str, float],
) -> dict[str, float]:
    """Fraction of positive weighted term gains (for L2 ITM dominance)."""
    gains = {
        k: weights.get(k, 0.0) * (curr_terms.get(k, 0.0) - prev_terms.get(k, 0.0))
        for k in curr_terms
    }
    pos = sum(max(0.0, g) for g in gains.values())
    if pos <= 0:
        return {k: 0.0 for k in gains}
    return {k: max(0.0, gains[k]) / pos for k in gains}


def resolve_judge_checkpoints(config, repo_root: Path | None = None) -> tuple[Path, Path]:
    """Return (itm_ckpt, vqa_ckpt), downloading ITM-large if needed."""
    root = repo_root or Path(__file__).resolve().parents[1]
    audit = config.audit
    itm_path = Path(str(audit.get("itm_large_ckpt", "cache/judges/blip_itm_large.pth")))
    vqa_path = Path(str(audit.get("vqa_judge_ckpt", "cache/judges/blip_vqa_published.pth")))
    if not vqa_path.is_file():
        fallback = Path(str(audit.get("vqa_judge_fallback", "cache/student_weights/checkpoint_09.pth")))
        if fallback.is_file():
            logger.info("VQA judge: using fallback %s", fallback)
            vqa_path = fallback
    if not itm_path.is_file():
        itm_path.parent.mkdir(parents=True, exist_ok=True)
        url = str(audit.get("itm_download_url", BLIP_ITM_LARGE_URL))
        logger.info("Downloading BLIP ITM-large to %s", itm_path)
        import torch

        torch.hub.download_url_to_file(url, str(itm_path), progress=True)
    return itm_path, vqa_path


@dataclass
class EpochAuditResult:
    mean_judge: float
    mean_reward: float
    mean_itm_judge: float
    mean_vqa_match: float
    term_means: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0


def build_audit_judges(config, *, device: str, repo_root: Path | None = None):
    """Load ITM-large + published VQA judge (independent of reward student)."""
    from filtering.adapters import BlipItmJudgeAdapter, BlipStudentAdapter

    itm_ckpt, vqa_ckpt = resolve_judge_checkpoints(config, repo_root=repo_root)
    itm = BlipItmJudgeAdapter(
        checkpoint=str(itm_ckpt),
        med_config=str(config.teacher.med_config),
        image_size=int(config.teacher.image_size),
        vit=str(config.audit.get("itm_vit", "large")),
        device=device,
    )
    vqa = BlipStudentAdapter(
        checkpoint=str(vqa_ckpt),
        med_config=str(config.student.med_config),
        image_size=int(config.student.image_size),
        device=device,
    )
    return itm, vqa


def run_epoch_audit(
    policy,
    dataset,
    *,
    device: str,
    n_audit: int,
    seed: int,
    generate_fn: Callable,
    reward_terms_fn: Callable[[str, str, str], tuple[float, dict[str, float]]],
    config,
    itm_judge=None,
    vqa_judge=None,
) -> EpochAuditResult:
    """Sample n_audit images, generate Q/A, score with judges + reward terms."""
    from filtering.matchers import max_match

    if itm_judge is None or vqa_judge is None:
        itm_judge, vqa_judge = build_audit_judges(config, device=device)

    n = min(n_audit, len(dataset))
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), n)

    itm_scores: list[float] = []
    vqa_scores: list[float] = []
    judge_scores: list[float] = []
    rewards: list[float] = []
    term_acc: dict[str, list[float]] = {}

    class _ExactOnlySbert:
        def encode(self, texts, convert_to_numpy=True):
            import numpy as np

            return np.array([[1.0, 0.0] if t else [0.0, 1.0] for t in texts])

    sbert = _ExactOnlySbert()

    for idx in indices:
        item = dataset[idx]
        if item is None:
            continue
        image_tensor, image_path = item
        caption = generate_fn(policy, image_tensor, device)
        question, answer = _parse_qa_caption(caption)
        if not question:
            continue

        itm_s = itm_judge.match_prob(image_path, question, answer)
        pred = vqa_judge.answer_question(image_path, question)
        vqa_s = float(max_match(pred, answer, sbert))
        j = judge_score(itm_s, vqa_s)
        r, terms = reward_terms_fn(image_path, question, answer)

        itm_scores.append(itm_s)
        vqa_scores.append(vqa_s)
        judge_scores.append(j)
        rewards.append(r)
        for k, v in terms.items():
            term_acc.setdefault(k, []).append(float(v))

    if not judge_scores:
        return EpochAuditResult(0.0, 0.0, 0.0, 0.0, {}, 0)

    term_means = {k: sum(v) / len(v) for k, v in term_acc.items()}
    return EpochAuditResult(
        mean_judge=sum(judge_scores) / len(judge_scores),
        mean_reward=sum(rewards) / len(rewards),
        mean_itm_judge=sum(itm_scores) / len(itm_scores),
        mean_vqa_match=sum(vqa_scores) / len(vqa_scores),
        term_means=term_means,
        n_samples=len(judge_scores),
    )


def _parse_qa_caption(caption: str) -> tuple[str, str]:
    from generate_questions import VQARecord, VQADatasetOrigin

    try:
        rec = VQARecord.build_from_raw_model_output(
            caption,
            "x",
            dataset_origin=VQADatasetOrigin.vqa,
            parse_rationale=False,
        )
        ans = rec.answer[0] if isinstance(rec.answer, list) else rec.answer
        return rec.question, ans
    except Exception:
        return caption, ""


def load_audit_history(audit_log: Path) -> list[dict]:
    if not audit_log.exists():
        return []
    rows = []
    for line in audit_log.read_text().strip().splitlines():
        if line:
            rows.append(json.loads(line))
    return rows
