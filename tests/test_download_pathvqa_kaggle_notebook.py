import ast
import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "download_pathvqa_kaggle.ipynb"
REQUIRED_TAGS = {
    "config",
    "dependencies",
    "helpers",
    "download",
    "materialize",
    "validation",
    "preview",
    "archive",
}
EXPECTED_OUTPUTS = {
    "all_data.json",
    "train.json",
    "val.json",
    "test.json",
    "answer_list.json",
    "test_val_combined.json",
}


def _load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _cell_source(cell):
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def _tagged_source(notebook, tag):
    matches = [
        _cell_source(cell)
        for cell in notebook["cells"]
        if tag in cell.get("metadata", {}).get("tags", [])
    ]
    assert len(matches) == 1, f"expected one cell tagged {tag!r}, got {len(matches)}"
    return matches[0]


def test_notebook_is_valid_nbformat_and_code_cells_compile():
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(_cell_source(cell), filename=f"cell_{index}")


def test_notebook_has_required_tagged_cells_and_kaggle_defaults():
    notebook = _load_notebook()
    tags = {
        tag
        for cell in notebook["cells"]
        for tag in cell.get("metadata", {}).get("tags", [])
    }
    assert REQUIRED_TAGS <= tags

    config = _tagged_source(notebook, "config")
    assert 'DATASET_ID = "flaviagiammarino/path-vqa"' in config
    assert 'Path("/kaggle/working/pathvqa")' in config
    assert "OVERWRITE_IMAGES = False" in config
    assert "CREATE_ARCHIVE = False" in config


def test_notebook_writes_expected_outputs_and_no_auxiliary_assets():
    notebook = _load_notebook()
    source = "\n".join(_cell_source(cell) for cell in notebook["cells"])
    for filename in EXPECTED_OUTPUTS:
        assert filename in source

    lowered = source.lower()
    assert "gdown" not in lowered
    assert "drive.google.com" not in lowered
    assert "pathvqa_teacher_checkpoint_file_id" not in lowered
    assert "pathvqa_synthetic_data_file_id" not in lowered
    assert "git clone" not in lowered
    assert "github.com/fantastichaha11/seltda" not in lowered


def test_conversion_helpers_build_raw_and_seltda_records():
    notebook = _load_notebook()
    namespace = {}
    exec(compile(_tagged_source(notebook, "helpers"), "helpers", "exec"), namespace)

    assert namespace["infer_answer_type"]("yes") == "yes/no"
    assert namespace["infer_answer_type"]("12") == "number"
    assert namespace["infer_question_type"]("How many cells?", "number") == "how many"

    raw_qa, raw_vqa, train_record = namespace["build_annotation_records"](
        {"question": "Is tissue visible?", "answer": "yes"},
        split_name="train",
        image_stem="train_000000",
        question_id=7,
    )
    assert raw_qa == {
        "image": "train_000000",
        "question": "Is tissue visible?",
        "answer": "yes",
    }
    assert raw_vqa["img_id"] == "train_000000"
    assert raw_vqa["question_id"] == 7
    assert train_record == {
        "image": "train/train_000000.jpg",
        "question": "Is tissue visible?",
        "answer": ["yes"],
        "dataset": "pathvqa",
        "question_id": 7,
    }

    _, _, val_record = namespace["build_annotation_records"](
        {"question": "What is visible?", "answer": "nuclei"},
        split_name="val",
        image_stem="val_000000",
        question_id=8,
    )
    assert val_record["answer"] == "nuclei"
    assert val_record["question_type"] == "what"
    assert val_record["answer_type"] == "other"
