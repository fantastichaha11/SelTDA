"""Gate logic (cascade), quantile thresholds, and soft score fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np

from filtering.gate_registry import apply_cascade
from filtering.scorers import normalize_min_max


@dataclass
class GateConfig:
    enabled: bool = True
    keep_top: float = 0.75


@dataclass
class GateThresholds:
    tau_conf: float
    tau_itm: float
    tau_xcons: float


def apply_gates(scores: dict, thresholds: GateThresholds) -> Tuple[bool, str]:
    """Cascade: conf -> itm -> xcons. Return (keep, reason)."""
    th = {
        "conf": thresholds.tau_conf,
        "itm": thresholds.tau_itm,
        "xcons": thresholds.tau_xcons,
    }
    return apply_cascade(scores, th, ["conf", "itm", "xcons"])


def thresholds_from_quantile(values: Sequence[float], keep_top: float) -> float:
    """Convert keep_top fraction to absolute threshold on values."""
    if keep_top >= 1.0:
        return float("-inf")
    if keep_top <= 0.0:
        return float("inf")
    cutoff = 1.0 - keep_top
    return float(np.quantile(np.asarray(values, dtype=np.float64), cutoff))


def normalize_rank(values: Sequence[float]) -> list[float]:
    """Map values to [0, 1] by rank (higher raw value -> higher rank)."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    for rank, idx in enumerate(order):
        ranks[idx] = rank / (n - 1)
    return ranks.tolist()


def normalize_gate_scores(values: Sequence[float], mode: str = "minmax") -> list[float]:
    """Normalize a gate's score vector for fusion."""
    if mode == "rank":
        return normalize_rank(values)
    if mode == "none":
        return [float(v) for v in values]
    normed = normalize_min_max(values)
    return [0.0 if v is None else float(v) for v in normed]


def fused_score(
    normalized_scores: Mapping[str, float],
    weights: Mapping[str, float],
    gate_order: Sequence[str],
) -> float:
    """Weighted average of per-gate normalized scores."""
    total_w = 0.0
    total = 0.0
    for gate in gate_order:
        w = float(weights.get(gate, 0.0))
        if w <= 0:
            continue
        total += w * float(normalized_scores.get(gate, 0.0))
        total_w += w
    if total_w <= 0:
        return 0.0
    return total / total_w


def apply_soft_fusion(
    normalized_scores: Mapping[str, float],
    weights: Mapping[str, float],
    tau: float,
    gate_order: Sequence[str],
) -> Tuple[bool, float]:
    """Keep sample when weighted fusion score >= tau. Returns (keep, fused_score)."""
    score = fused_score(normalized_scores, weights, gate_order)
    return score >= tau, score
