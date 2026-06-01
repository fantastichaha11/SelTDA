"""HRP-1: tiered hacking-response controller. L1-L3 auto; L4-L5 flag for human."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class HrpState:
    kl_beta: float
    w_itm: float
    persistent_rounds: int = 0


def respond(
    state: HrpState,
    *,
    hacking: bool,
    dominant_term: str | None,
    judge_dropping: bool,
    auto_levels: list[int],
    kl_c: float = 1.5,
    downweight_d: float = 0.5,
) -> tuple[HrpState, str]:
    """Return (new_state, action). Escalates L4 when hacking persists >=2 rounds."""
    if not hacking:
        return state, "continue"

    if state.persistent_rounds >= 2 or (judge_dropping and 3 not in auto_levels):
        return state, "flag_human"

    if judge_dropping and 3 in auto_levels:
        return state, "early_stop_rollback"

    if dominant_term == "itm" and 2 in auto_levels:
        return replace(state, w_itm=state.w_itm * downweight_d), "downweight"

    if 1 in auto_levels:
        return replace(state, kl_beta=state.kl_beta * kl_c), "adaptive_kl"

    return state, "flag_human"
