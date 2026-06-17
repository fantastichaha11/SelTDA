import json

import pytest

from filtering.score_cache import (
    apply_gate_cache,
    cache_file_path,
    load_gate_cache,
    record_key,
    records_missing_gate,
    save_gate_cache,
)


def test_record_key_prefers_question_id():
    assert record_key({"question_id": 42, "image": "a.jpg", "question": "q"}) == "42"


def test_record_key_fallback_image_question():
    r = {"image": "vg/x.jpg", "question": "what?"}
    assert record_key(r) == "vg/x.jpg\x00what?"


def test_save_and_load_gate_cache_roundtrip(tmp_path):
    input_path = tmp_path / "pool.json"
    input_path.write_text("[]")
    records = [
        {"question_id": 0, "scores": {"vqascore": 0.9}},
        {"question_id": 1, "scores": {"vqascore": 0.1}},
    ]
    save_gate_cache(tmp_path, input_path, "vqascore", records)
    path = cache_file_path(tmp_path, input_path, "vqascore")
    assert path.is_file()

    payload = load_gate_cache(tmp_path, input_path, "vqascore", expected_n=2)
    assert payload is not None
    assert payload["scores"]["0"] == pytest.approx(0.9)


def test_apply_gate_cache_attaches_scores_and_extras():
    records = [{"question_id": 0, "question": "q", "image": "a.jpg"}]
    payload = {
        "scores": {"0": 0.75},
        "extras": {"0": {"_student_answer": "yes"}},
    }
    n = apply_gate_cache(records, "xcons", payload)
    assert n == 1
    assert records[0]["scores"]["xcons"] == pytest.approx(0.75)
    assert records[0]["_student_answer"] == "yes"


def test_records_missing_gate():
    records = [{"scores": {"itm": 0.5}}, {}]
    assert len(records_missing_gate(records, "itm")) == 1
    assert len(records_missing_gate(records, "vqascore")) == 2


def test_load_gate_cache_rejects_count_mismatch(tmp_path):
    input_path = tmp_path / "pool.json"
    cache_path = cache_file_path(tmp_path, input_path, "itm")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "version": 1,
                "input": str(input_path),
                "n_records": 99,
                "gate": "itm",
                "scores": {"0": 0.5},
            }
        )
    )
    assert load_gate_cache(tmp_path, input_path, "itm", expected_n=5) is None
