import json
from pathlib import Path

from scripts.random_subsample_control import main
import sys


def test_random_subsample_matches_reference_size(tmp_path, monkeypatch):
    raw = [{"question_id": i, "question": f"Q{i}?", "answer": ["a"], "image": "x.jpg", "dataset": "aokvqa"} for i in range(10)]
    ref = raw[:4]
    inp = tmp_path / "raw.json"
    refp = tmp_path / "ref.json"
    out_base = tmp_path / "out.json"
    inp.write_text(json.dumps(raw))
    refp.write_text(json.dumps(ref))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "random_subsample_control.py",
            "--input",
            str(inp),
            "--reference",
            str(refp),
            "--output",
            str(out_base),
            "--seeds",
            "42",
        ],
    )
    main()
    out = json.loads(out_base.with_name("out_seed42.json").read_text())
    assert len(out) == 4
