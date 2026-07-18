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


def _tagged_source(notebook, tag):
    matches = [
        _source(cell)
        for cell in notebook["cells"]
        if tag in cell.get("metadata", {}).get("tags", [])
    ]
    assert len(matches) == 1
    return matches[0]


def test_notebook_json_and_code_cells_are_valid():
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(_source(cell), filename=f"cell_{index}")


def test_notebook_exposes_checkpoint_list_and_both_pathvqa_eval_splits():
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
    assert "EVAL_SPLITS = ['val', 'test']" in source
    assert "dataset_name': 'pathvqa'" in source
    assert "train.json" in source
    assert "val.json" in source
    assert "test.json" in source
    assert "answer_list.json" in source
    assert "path / 'images' / split_name" in source


def test_eval_config_stages_val_and_test_without_modifying_source_data(tmp_path):
    notebook = _load_notebook()
    data_root = tmp_path / "pathvqa"
    data_root.mkdir()
    train_records = [{"question_id": 1, "answer": ["train answer"]}]
    val_records = [{"question_id": 2, "answer": "val answer"}]
    test_records = [{"question_id": 3, "answer": "test answer"}]
    for split_name, records in [
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ]:
        (data_root / f"{split_name}.json").write_text(json.dumps(records))

    namespace = {
        "pathvqa_root": data_root,
        "EVAL_SPLITS": ["val", "test"],
        "OUTPUT_ROOT": tmp_path / "output",
        "REPO_DIR": tmp_path / "repo",
        "checkpoint_paths": {"checkpoint": tmp_path / "checkpoint.pth"},
        "BATCH_SIZE_TEST": 4,
        "K_TEST": 128,
        "INFERENCE": "rank",
    }
    exec(
        compile(_tagged_source(notebook, "eval_config"), NOTEBOOK_PATH.name, "exec"),
        namespace,
    )

    annotation_roots = namespace["annotation_roots"]
    assert json.loads((annotation_roots["val"] / "test.json").read_text()) == val_records
    assert json.loads((annotation_roots["test"] / "test.json").read_text()) == test_records
    assert json.loads((annotation_roots["val"] / "answer_list.json").read_text()) == [
        "val answer"
    ]
    assert json.loads((data_root / "val.json").read_text()) == val_records


def test_notebook_evaluates_every_checkpoint_and_writes_summary():
    notebook = _load_notebook()
    source = "\n".join(_source(cell) for cell in notebook["cells"])
    assert "for eval_split in EVAL_SPLITS:" in source
    assert "for label, checkpoint in checkpoint_paths.items():" in source
    assert "train_vqa.py" in source
    assert "--evaluate" in source
    assert "--no-resume" in source
    assert "pretrained=" in source
    assert "pathvqa_eval.py" in source
    assert "--annotation-file" in source
    assert "pathvqa_eval_summary.json" in source
    assert "pathvqa_eval_summary.csv" in source
    assert "pathvqa_{eval_split}_summary.json" in source
    assert "pathvqa_{eval_split}_summary.csv" in source
