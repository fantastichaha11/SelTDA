import json
from pathlib import Path

from orchestration.skill_gap import compute_skill_gap_from_results


def test_compute_skill_gap_top3_weak(tmp_path):
    results = [
        {"question": "Is it red?", "answer": "yes", "prediction": "no", "correct": False},
        {"question": "Is it blue?", "answer": "no", "prediction": "yes", "correct": False},
        {"question": "How many?", "answer": "2", "prediction": "2", "correct": True},
        {"question": "What kind of sport?", "answer": "soccer", "prediction": "tennis", "correct": False},
    ]
    p = tmp_path / "results.json"
    p.write_text(json.dumps(results))
    report = compute_skill_gap_from_results(p)
    assert len(report.weak_types) == 3
    assert "yes_no" in report.weak_types or "external_knowledge" in report.weak_types
