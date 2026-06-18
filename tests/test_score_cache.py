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


def test_record_key_uses_image_and_question():
    assert (
        record_key({"question_id": 42, "image": "a.jpg", "question": "q"})
        == "a.jpg\x00q"
    )


def test_record_key_distinguishes_same_question_id():
    a = {"question_id": 0, "image": "unlabeled2017/1.jpg", "question": "what?"}
    b = {"question_id": 0, "image": "unlabeled2017/2.jpg", "question": "what?"}
    assert record_key(a) != record_key(b)


def test_save_and_load_gate_cache_roundtrip(tmp_path):
    input_path = tmp_path / "pool.json"
    input_path.write_text("[]")
    records = [
        {"question_id": 0, "image": "a.jpg", "question": "q0", "scores": {"vqascore": 0.9}},
        {"question_id": 1, "image": "b.jpg", "question": "q1", "scores": {"vqascore": 0.1}},
    ]
    save_gate_cache(tmp_path, input_path, "vqascore", records)
    path = cache_file_path(tmp_path, input_path, "vqascore")
    assert path.is_file()

    payload = load_gate_cache(tmp_path, input_path, "vqascore", expected_n=2)
    assert payload is not None
    assert payload["scores"]["a.jpg\x00q0"] == pytest.approx(0.9)


def test_apply_gate_cache_attaches_scores_and_extras():
    records = [{"question_id": 0, "question": "q", "image": "a.jpg"}]
    payload = {
        "scores": {"a.jpg\x00q": 0.75},
        "extras": {"a.jpg\x00q": {"_student_answer": "yes"}},
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
                "version": 2,
                "input": str(input_path),
                "n_records": 99,
                "gate": "itm",
                "scores": {"0": 0.5},
            }
        )
    )
    assert load_gate_cache(tmp_path, input_path, "itm", expected_n=5) is None
