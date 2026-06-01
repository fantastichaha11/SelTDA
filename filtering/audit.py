"""AUD-1: independent held-out judges + hacking detection (epoch-level)."""

from __future__ import annotations


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
