import json
from pathlib import Path

import pytest


@pytest.fixture
def tiny_setup(tmp_path):
    from PIL import Image

    fixture_src = Path(__file__).parent / "fixtures" / "synthetic_data_raw_tiny.json"
    records = json.loads(fixture_src.read_text())

    image_root = tmp_path / "images"
    (image_root / "vg").mkdir(parents=True)
    for i in range(5):
        Image.new("RGB", (32, 32), color=(i * 50, 100, 100)).save(
            image_root / "vg" / f"img_{i}.jpg"
        )

    raw_path = tmp_path / "synthetic_data_raw.json"
    raw_path.write_text(json.dumps(records))
    return {
        "image_root": str(image_root),
        "input": str(raw_path),
        "output": str(tmp_path / "synthetic_data.json"),
        "report": str(tmp_path / "filter_report.json"),
    }


def test_orchestrator_with_all_gates_disabled_keeps_all(tiny_setup):
    from omegaconf import OmegaConf
    import filter_pseudo

    config = OmegaConf.create(
        {
            "input": tiny_setup["input"],
            "image_root": tiny_setup["image_root"],
            "output": tiny_setup["output"],
            "report": tiny_setup["report"],
            "gates": {
                "conf": {"enabled": False, "keep_top": 1.0},
                "itm": {
                    "enabled": False,
                    "keep_top": 1.0,
                    "clip_model": "x",
                    "clip_pretrained": "y",
                },
                "xcons": {
                    "enabled": False,
                    "keep_top": 1.0,
                    "student_ckpt": "x",
                    "sbert_model": "y",
                    "image_size": 384,
                },
            },
            "scoring_only": False,
            "seed": 0,
            "device": "cpu",
            "report_max_examples": 10,
            "log_every": 100,
            "torch_home": None,
        }
    )

    class Args:
        output_dir = str(Path(tiny_setup["output"]).parent)
        result_dir = str(Path(tiny_setup["output"]).parent)
        device = "cpu"
        seed = 0

    filter_pseudo.main(Args(), config)

    out = json.loads(Path(tiny_setup["output"]).read_text())
    assert len(out) == 5

    report = json.loads(Path(tiny_setup["report"]).read_text())
    assert report["counts"]["total_in"] == 5
    assert report["counts"]["kept"] == 5


def test_orchestrator_confidence_only_drops_low_logprob(tiny_setup):
    from omegaconf import OmegaConf
    import filter_pseudo

    config = OmegaConf.create(
        {
            "input": tiny_setup["input"],
            "image_root": tiny_setup["image_root"],
            "output": tiny_setup["output"],
            "report": tiny_setup["report"],
            "gates": {
                "conf": {"enabled": True, "keep_top": 0.6},
                "itm": {
                    "enabled": False,
                    "keep_top": 1.0,
                    "clip_model": "x",
                    "clip_pretrained": "y",
                },
                "xcons": {
                    "enabled": False,
                    "keep_top": 1.0,
                    "student_ckpt": "x",
                    "sbert_model": "y",
                    "image_size": 384,
                },
            },
            "scoring_only": False,
            "seed": 0,
            "device": "cpu",
            "report_max_examples": 10,
            "log_every": 100,
            "torch_home": None,
        }
    )

    class Args:
        output_dir = str(Path(tiny_setup["output"]).parent)
        result_dir = str(Path(tiny_setup["output"]).parent)
        device = "cpu"
        seed = 0

    filter_pseudo.main(Args(), config)
    out = json.loads(Path(tiny_setup["output"]).read_text())
    assert len(out) == 3
    kept_qids = sorted(r["question_id"] for r in out)
    assert kept_qids == [0, 2, 4]
