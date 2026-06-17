import json
from pathlib import Path

import pytest

from filtering.score_cache import save_gate_cache


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
                "vqascore": {
                    "enabled": False,
                    "keep_top": 1.0,
                    "model": "clip-flant5-xl",
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
            "stratify": {"enabled": False},
            "coreset": {"enabled": False},
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
                "vqascore": {
                    "enabled": False,
                    "keep_top": 1.0,
                    "model": "clip-flant5-xl",
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
            "stratify": {"enabled": False},
            "coreset": {"enabled": False},
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


def test_orchestrator_fusion_confidence_only_keeps_top_fraction(tiny_setup):
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
                "vqascore": {
                    "enabled": False,
                    "keep_top": 1.0,
                    "model": "clip-flant5-xl",
                },
                "xcons": {
                    "enabled": False,
                    "keep_top": 1.0,
                    "student_ckpt": "x",
                    "sbert_model": "y",
                    "image_size": 384,
                },
            },
            "fusion": {
                "enabled": True,
                "keep_top": 0.6,
                "normalize": "minmax",
                "weights": {"conf": 1.0},
            },
            "scoring_only": False,
            "seed": 0,
            "device": "cpu",
            "report_max_examples": 10,
            "log_every": 100,
            "torch_home": None,
            "stratify": {"enabled": False},
            "coreset": {"enabled": False},
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
    report = json.loads(Path(tiny_setup["report"]).read_text())
    assert "fusion" in report
    assert report["fusion"]["weights"] == {"conf": 1.0}


def test_orchestrator_reuses_cached_vqascore_without_rescoring(tiny_setup, tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    import filter_pseudo

    input_path = Path(tiny_setup["input"])
    cache_dir = tmp_path / "score_cache"
    cached_records = json.loads(input_path.read_text())
    for i, r in enumerate(cached_records):
        r["scores"] = {"vqascore": 0.9 - i * 0.1}
    save_gate_cache(cache_dir, input_path, "vqascore", cached_records)

    calls = {"adapter": 0}

    class TrackingAdapter:
        def __init__(self, **kwargs):
            calls["adapter"] += 1

        def score_pairs(self, image_paths, texts):
            raise AssertionError("VQAScore should not run when cache is complete")

    monkeypatch.setattr(filter_pseudo, "VQAScoreAdapter", TrackingAdapter)

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
                "vqascore": {
                    "enabled": True,
                    "keep_top": 0.6,
                    "model": "clip-flant5-xl",
                },
                "xcons": {
                    "enabled": False,
                    "keep_top": 1.0,
                    "student_ckpt": "x",
                    "sbert_model": "y",
                    "image_size": 384,
                },
            },
            "score_cache": {
                "dir": str(cache_dir),
                "save": True,
                "merge_input_scores": True,
            },
            "scoring_only": False,
            "seed": 0,
            "device": "cpu",
            "report_max_examples": 10,
            "log_every": 100,
            "torch_home": None,
            "stratify": {"enabled": False},
            "coreset": {"enabled": False},
        }
    )

    class Args:
        output_dir = str(Path(tiny_setup["output"]).parent)
        result_dir = str(Path(tiny_setup["output"]).parent)
        device = "cpu"
        seed = 0

    filter_pseudo.main(Args(), config)
    assert calls["adapter"] == 0
    out = json.loads(Path(tiny_setup["output"]).read_text())
    assert all("vqascore" in r.get("scores", {}) for r in out)


def test_orchestrator_stratify_enabled_keeps_subset(tiny_setup):
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
                "vqascore": {
                    "enabled": False,
                    "keep_top": 1.0,
                    "model": "clip-flant5-xl",
                },
                "xcons": {
                    "enabled": False,
                    "keep_top": 1.0,
                    "student_ckpt": "x",
                    "sbert_model": "y",
                    "image_size": 384,
                },
            },
            "stratify": {
                "enabled": True,
                "min_stratum_size": 1,
                "global_keep_top_fallback": 0.75,
            },
            "coreset": {"enabled": False},
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
    assert 0 < len(out) <= 5
