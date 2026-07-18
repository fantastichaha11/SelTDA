import ast
import json
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "eval_pathvqa_checkpoints_kaggle.ipynb"
)
REQUIRED_TAGS = {
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


def _load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _source(cell):
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def test_notebook_json_and_code_cells_are_valid():
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(_source(cell), filename=f"cell_{index}")


def test_notebook_exposes_checkpoint_list_and_pathvqa_test_config():
    notebook = _load_notebook()
    source = "\n".join(_source(cell) for cell in notebook["cells"])
    tags = {
        tag
        for cell in notebook["cells"]
        for tag in cell.get("metadata", {}).get("tags", [])
    }
    assert REQUIRED_TAGS <= tags
    assert "CHECKPOINT_PATHS = [" in source
    assert "CHECKPOINT_SEARCH_ROOTS" in source
    assert "checkpoint_*.pth" in source
    assert "dataset_name': 'pathvqa'" in source
    assert "train.json" in source
    assert "test.json" in source
    assert "answer_list.json" in source


def test_notebook_evaluates_every_checkpoint_and_writes_summary():
    notebook = _load_notebook()
    source = "\n".join(_source(cell) for cell in notebook["cells"])
    assert "for label, checkpoint in checkpoint_paths.items():" in source
    assert "train_vqa.py" in source
    assert "--evaluate" in source
    assert "--no-resume" in source
    assert "pretrained=" in source
    assert "pathvqa_eval.py" in source
    assert "--annotation-file" in source
    assert "pathvqa_eval_summary.json" in source
    assert "pathvqa_eval_summary.csv" in source
