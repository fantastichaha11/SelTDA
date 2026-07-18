import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_NOTEBOOK = ROOT / "download_vizwiz_kaggle.ipynb"
EVAL_NOTEBOOK = ROOT / "eval_vizwiz_checkpoints_kaggle.ipynb"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _source(cell):
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def _notebook_source(notebook):
    return "\n".join(_source(cell) for cell in notebook["cells"])


def _tags(notebook):
    return {
        tag
        for cell in notebook["cells"]
        for tag in cell.get("metadata", {}).get("tags", [])
    }


def _tagged_source(notebook, tag):
    matches = [
        _source(cell)
        for cell in notebook["cells"]
        if tag in cell.get("metadata", {}).get("tags", [])
    ]
    assert len(matches) == 1
    return matches[0]


def _assert_valid_notebook(path):
    notebook = _load(path)
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(_source(cell), filename=f"{path.name}:cell-{index}")
    return notebook


def test_download_notebook_downloads_only_val_and_test():
    notebook = _assert_valid_notebook(DOWNLOAD_NOTEBOOK)
    source = _notebook_source(notebook)

    assert {"config", "setup", "download", "convert", "validate", "preview", "archive"} <= _tags(notebook)
    assert "scripts/download_vizwiz.py" in source
    assert "VAL_IMAGES_URL" in source
    assert "TEST_IMAGES_URL" in source
    assert "VAL_ANNOTATIONS_URL" in source
    assert "TEST_ANNOTATIONS_URL" in source
    assert "VizWiz_all_answers/VQA_test.json" in source
    assert "val.json" in source
    assert "test.json" in source
    assert "answer_list.json" in source
    assert "vizwiz_val_metadata.json" in source
    assert "vizwiz_test_metadata.json" in source
    assert "images/val" in source
    assert "images/test" in source
    assert "for split_name in ('val', 'test')" in source


def test_download_notebook_converts_both_splits_without_test_answer_leakage(tmp_path):
    notebook = _assert_valid_notebook(DOWNLOAD_NOTEBOOK)
    for split_name, answer in [("val", "validation answer"), ("test", "test-only answer")]:
        image_name = f"VizWiz_{split_name}_00000001.jpg"
        image_path = tmp_path / "images" / split_name / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"image")
        annotation_path = tmp_path / "annotations" / f"{split_name}.json"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(
            json.dumps(
                [
                    {
                        "image": image_name,
                        "question": f"Question for {split_name}?",
                        "answers": [{"answer": answer}] * 10,
                        "answerable": 1,
                        "answer_type": "other",
                    }
                ]
            ),
            encoding="utf-8",
        )

    namespace = {
        "OUTPUT_ROOT": tmp_path,
        "INCLUDE_UNANSWERABLE": True,
    }
    convert_source = _tagged_source(notebook, "convert")
    exec(compile(convert_source, DOWNLOAD_NOTEBOOK.name, "exec"), namespace)

    assert not (tmp_path / "train.json").exists()
    assert json.loads((tmp_path / "answer_list.json").read_text()) == [
        "validation answer"
    ]
    test_records = json.loads((tmp_path / "test.json").read_text())
    assert test_records[0]["image"].startswith("test/")
    assert test_records[0]["question_id"] == 2_000_000
    test_metadata = json.loads((tmp_path / "vizwiz_test_metadata.json").read_text())
    assert test_metadata["2000000"]["answer_type"] == "other"


def test_eval_notebook_accepts_checkpoint_list_and_both_eval_splits():
    notebook = _assert_valid_notebook(EVAL_NOTEBOOK)
    source = _notebook_source(notebook)

    required_tags = {
        "config",
        "setup",
        "data",
        "checkpoints",
        "eval_config",
        "gpu",
        "inference",
        "metrics",
        "summary",
    }
    assert required_tags <= _tags(notebook)
    assert "CHECKPOINT_PATHS = [" in source
    assert "CHECKPOINT_SEARCH_ROOTS" in source
    assert "checkpoint_*.pth" in source
    assert "EVAL_SPLITS = ['val', 'test']" in source
    assert "'dataset_name': 'generic_vqa'" in source
    assert "'train_files': ['val']" in source
    assert "'val_file': eval_split" in source
    assert "vizwiz_val_metadata.json" in source
    assert "vizwiz_test_metadata.json" in source
    assert "(path / 'images/train').is_dir()" not in source


def test_eval_notebook_writes_one_config_per_split(tmp_path):
    notebook = _assert_valid_notebook(EVAL_NOTEBOOK)
    namespace = {
        "vizwiz_root": tmp_path / "vizwiz",
        "EVAL_SPLITS": ["val", "test"],
        "REPO_DIR": tmp_path / "repo",
        "checkpoint_paths": {"checkpoint": tmp_path / "checkpoint.pth"},
        "BATCH_SIZE_TEST": 4,
        "K_TEST": 128,
        "INFERENCE": "rank",
    }
    exec(
        compile(_tagged_source(notebook, "eval_config"), EVAL_NOTEBOOK.name, "exec"),
        namespace,
    )

    config_paths = namespace["config_paths"]
    assert set(config_paths) == {"val", "test"}
    assert "val_file: val" in config_paths["val"].read_text()
    assert "val_file: test" in config_paths["test"].read_text()
    assert namespace["config_args"]["val"].startswith("configs/")


def test_eval_notebook_scores_every_checkpoint_and_writes_summaries():
    notebook = _assert_valid_notebook(EVAL_NOTEBOOK)
    source = _notebook_source(notebook)

    assert "for eval_split in EVAL_SPLITS:" in source
    assert "for label, checkpoint in checkpoint_paths.items():" in source
    assert "train_vqa.py" in source
    assert "--evaluate" in source
    assert "--no-resume" in source
    assert "pretrained=" in source
    assert "vizwiz_eval.py" in source
    assert "--annotation-file" in source
    assert "--metadata-file" in source
    assert "vizwiz_eval_summary.json" in source
    assert "vizwiz_eval_summary.csv" in source
    assert "vizwiz_{eval_split}_summary.json" in source
    assert "vizwiz_{eval_split}_summary.csv" in source
