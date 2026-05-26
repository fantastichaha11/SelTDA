from __future__ import annotations

from filtering.scorers import score_clip_itm, score_confidence, score_xcons

GATE_SCORERS = {
    "conf": score_confidence,
    "itm": score_clip_itm,
    "xcons": score_xcons,
}


def enabled_gate_names(config) -> list[str]:
    return [name for name in GATE_SCORERS if getattr(config.gates, name).enabled]


def apply_cascade(scores: dict, thresholds: dict, gate_order: list[str]) -> tuple[bool, str]:
    for gate in gate_order:
        if scores[gate] < thresholds[gate]:
            return False, gate
    return True, "kept"
