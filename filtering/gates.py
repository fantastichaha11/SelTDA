"""Gate logic (cascade) and quantile-based threshold helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from filtering.gate_registry import apply_cascade


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


def apply_soft_fusion(scores: dict, weights, tau: float):
    """Phase 2 (spec §6) — intentionally not implemented in this plan."""
    raise NotImplementedError("Soft fusion is Phase 2; out of scope.")
