"""TC-1: question-type schedule for generation."""

from __future__ import annotations

QUESTION_TYPES = [
    "yes_no",
    "how_many",
    "color",
    "external_knowledge",
    "visual_reasoning",
]


def next_question_type(
    index: int,
    schedule: str,
    weak_types: list[str] | None = None,
) -> str:
    if schedule == "round_robin":
        return QUESTION_TYPES[index % len(QUESTION_TYPES)]
    if schedule == "target_weak" and weak_types:
        pool = weak_types * 3 + QUESTION_TYPES
        return pool[index % len(pool)]
    return QUESTION_TYPES[index % len(QUESTION_TYPES)]


def type_prompt_prefix(qtype: str) -> str:
    return f"[TYPE={qtype}] "
