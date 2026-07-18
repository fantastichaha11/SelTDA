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


def _assert_valid_notebook(path):
    notebook = _load(path)
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(_source(cell), filename=f"{path.name}:cell-{index}")
    return notebook


def test_download_notebook_downloads_converts_and_validates_vizwiz():
    notebook = _assert_valid_notebook(DOWNLOAD_NOTEBOOK)
    source = _notebook_source(notebook)

    assert {"config", "setup", "download", "convert", "validate", "preview", "archive"} <= _tags(notebook)
    assert "scripts/download_vizwiz.py" in source
    assert "convert_vizwiz.py" in source
    assert "INCLUDE_TEST" in source
    assert "train.json" in source
    assert "val.json" in source
    assert "answer_list.json" in source
    assert "vizwiz_val_metadata.json" in source
    assert "images/train" in source
    assert "images/val" in source


def test_eval_notebook_accepts_checkpoint_list_and_uses_validation_labels():
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
    assert "'dataset_name': 'generic_vqa'" in source
    assert "'val_file': 'val'" in source
    assert "vizwiz_val_metadata.json" in source


def test_eval_notebook_scores_every_checkpoint_and_writes_summaries():
    notebook = _assert_valid_notebook(EVAL_NOTEBOOK)
    source = _notebook_source(notebook)

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
