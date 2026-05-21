import json
from pathlib import Path

from filtering.io import dump_records, load_records

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_data_raw_tiny.json"


def test_load_records_returns_list_of_dicts():
    records = load_records(FIXTURE)
    assert isinstance(records, list)
    assert len(records) == 5
    assert all(isinstance(r, dict) for r in records)


def test_load_records_preserves_all_fields():
    records = load_records(FIXTURE)
    assert set(records[0].keys()) >= {
        "question_id",
        "question",
        "answer",
        "image",
        "dataset",
        "gen_logprob",
    }


def test_dump_records_roundtrip(tmp_path):
    records = load_records(FIXTURE)
    out = tmp_path / "out.json"
    dump_records(records, out)
    loaded = json.loads(out.read_text())
    assert loaded == records


def test_dump_records_creates_parent_dirs(tmp_path):
    records = [{"a": 1}]
    out = tmp_path / "deep" / "nested" / "out.json"
    dump_records(records, out)
    assert out.exists()
