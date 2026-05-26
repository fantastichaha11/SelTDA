from __future__ import annotations

from filtering.scorers import score_clip_itm, score_confidence, score_xcons

GATE_SCORERS = {
    "conf": score_confidence,
    "itm": score_clip_itm,
    "xcons": score_xcons,
}

# lp/kcons scored in filter_pseudo (need student/NLI context); order preserved here.
GATE_ORDER = ["conf", "itm", "xcons", "lp", "kcons"]


def enabled_gate_names(config) -> list[str]:
    enabled = []
    for name in GATE_ORDER:
        if not hasattr(config.gates, name):
            continue
        if getattr(config.gates, name).enabled:
            enabled.append(name)
    return enabled


def apply_cascade(scores: dict, thresholds: dict, gate_order: list[str]) -> tuple[bool, str]:
    for gate in gate_order:
        if scores[gate] < thresholds[gate]:
            return False, gate
    return True, "kept"
