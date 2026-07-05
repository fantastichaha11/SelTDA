# VizWiz and DAQUAR Dataset Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add converter, config, evaluation, and experiment-script support for VizWiz-VQA and DAQUAR using the repo's existing `generic_vqa` training path.

**Architecture:** Keep model/training code unchanged. Add a tiny `dataset_adapters` utility module for answer normalization, JSON writing, answer-list construction, and metric helpers, then build dataset-specific converter/evaluator scripts around it. Use `configs/*.yaml` with `dataset_name: generic_vqa`, plus shell scripts patterned after `examples/run_pathvqa_experiment.sh`.

**Tech Stack:** Python 3, Pydantic-compatible dict JSON records, Hydra/OmegaConf config files, PyTorch training entrypoints already in repo, pytest, Bash.

---

## File Structure

- Create `dataset_adapters/__init__.py`: package marker and exported helper names.
- Create `dataset_adapters/generic_vqa.py`: shared helpers for normalized answers, answer lists, JSON IO, VQA soft accuracy, exact match, and question-prefix buckets.
- Create `convert_vizwiz.py`: converts official VizWiz train/val JSON plus image folders into `train.json`, `val.json`, `answer_list.json`, `vizwiz_val_metadata.json`.
- Create `vizwiz_eval.py`: evaluates `train_vqa.py` results using VQA-style soft accuracy and answerability/type strata.
- Create `convert_daquar.py`: converts DAQUAR JSON or table-style QA files into `train.json`, `val.json`, `answer_list.json`, `daquar_val_metadata.json`.
- Create `daquar_eval.py`: evaluates DAQUAR results with normalized exact match and question-prefix buckets.
- Create `configs/vizwiz.yaml`: generic VQA config for VizWiz.
- Create `configs/daquar.yaml`: generic VQA config for DAQUAR.
- Create `examples/run_vizwiz_experiment.sh`: end-to-end VizWiz script.
- Create `examples/run_daquar_experiment.sh`: end-to-end DAQUAR script.
- Create `tests/test_vizwiz_support.py`: converter/evaluator/config smoke tests for VizWiz.
- Create `tests/test_daquar_support.py`: converter/evaluator/config smoke tests for DAQUAR.

Do not modify `train_vqa.py`, `data/vqa_dataset.py`, `data/__init__.py`, BLIP model files, or existing evaluator behavior.

---

### Task 1: Shared Generic VQA Adapter Helpers

**Files:**
- Create: `dataset_adapters/__init__.py`
- Create: `dataset_adapters/generic_vqa.py`
- Test: `tests/test_vizwiz_support.py`

- [ ] **Step 1: Write failing helper tests**

Create `tests/test_vizwiz_support.py` with these initial tests:

```python
import json
from pathlib import Path

from dataset_adapters.generic_vqa import (
    build_answer_list,
    exact_match_accuracy,
    normalize_answer,
    question_prefix,
    vqa_soft_accuracy,
    write_json,
)


def test_normalize_answer_basic_punctuation_and_case():
    assert normalize_answer("  The Bottle!!! ") == "the bottle"
    assert normalize_answer("NO.") == "no"
    assert normalize_answer("") == ""


def test_build_answer_list_is_deterministic():
    records = [
        {"answer": ["zebra", "apple"]},
        {"answer": ["apple", "bottle"]},
    ]
    assert build_answer_list(records) == ["apple", "bottle", "zebra"]


def test_vqa_soft_accuracy_uses_reference_count_over_three():
    refs = ["cat", "cat", "dog", "cat", "horse"]
    assert vqa_soft_accuracy("cat", refs) == 1.0
    assert vqa_soft_accuracy("dog", refs) == 1 / 3
    assert vqa_soft_accuracy("bird", refs) == 0.0


def test_exact_match_accuracy_normalizes_answers():
    predictions = {
        1: "Bottle!",
        2: "wrong",
    }
    references = {
        1: ["bottle"],
        2: ["chair"],
    }
    assert exact_match_accuracy(predictions, references) == 0.5


def test_question_prefix_buckets_common_forms():
    assert question_prefix("How many cans are there?") == "how many"
    assert question_prefix("Where is the label?") == "where"
    assert question_prefix("Is this readable?") == "is/are"
    assert question_prefix("Name the object") == "other"


def test_write_json_creates_parent_directory(tmp_path):
    out = tmp_path / "nested" / "data.json"
    write_json(out, [{"a": 1}])
    assert json.loads(out.read_text()) == [{"a": 1}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_vizwiz_support.py -q
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'dataset_adapters'`.

- [ ] **Step 3: Add package marker**

Create `dataset_adapters/__init__.py`:

```python
"""Dataset conversion and evaluation helpers for SelTDA experiments."""

from dataset_adapters.generic_vqa import (
    build_answer_list,
    exact_match_accuracy,
    load_json,
    normalize_answer,
    question_prefix,
    vqa_soft_accuracy,
    write_json,
)

__all__ = [
    "build_answer_list",
    "exact_match_accuracy",
    "load_json",
    "normalize_answer",
    "question_prefix",
    "vqa_soft_accuracy",
    "write_json",
]
```

- [ ] **Step 4: Implement shared helper module**

Create `dataset_adapters/generic_vqa.py`:

```python
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence


_PUNCT_RE = re.compile(r"[\t\n\r\f\v]+|[\"'`.,!?;:()\[\]{}]")
_SPACE_RE = re.compile(r"\s+")


def normalize_answer(answer: object) -> str:
    """Normalize short VQA answers without applying dataset-specific stemming."""
    text = "" if answer is None else str(answer)
    text = text.strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _iter_answers(records: Iterable[Mapping]) -> Iterable[str]:
    for record in records:
        answers = record.get("answer", [])
        if isinstance(answers, str):
            answers = [answers]
        for answer in answers:
            normalized = normalize_answer(answer)
            if normalized:
                yield normalized


def build_answer_list(*record_groups: Iterable[Mapping]) -> list[str]:
    answers: set[str] = set()
    for records in record_groups:
        answers.update(_iter_answers(records))
    return sorted(answers)


def vqa_soft_accuracy(prediction: str, references: Sequence[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_references = [normalize_answer(ref) for ref in references]
    matches = sum(ref == normalized_prediction for ref in normalized_references)
    return min(1.0, matches / 3.0)


def exact_match_accuracy(
    predictions: Mapping[int, str],
    references: Mapping[int, Sequence[str]],
) -> float:
    if not references:
        return 0.0
    correct = 0
    for question_id, ref_answers in references.items():
        prediction = normalize_answer(predictions.get(question_id, ""))
        normalized_refs = {normalize_answer(answer) for answer in ref_answers}
        if prediction in normalized_refs:
            correct += 1
    return correct / len(references)


def question_prefix(question: str) -> str:
    q = normalize_answer(question)
    if q.startswith("how many"):
        return "how many"
    if q.startswith("where"):
        return "where"
    if q.startswith("what"):
        return "what"
    if q.startswith("is ") or q.startswith("are "):
        return "is/are"
    return "other"
```

- [ ] **Step 5: Run helper tests**

Run:

```bash
pytest tests/test_vizwiz_support.py -q
```

Expected: PASS for all helper tests.

- [ ] **Step 6: Commit**

```bash
git add dataset_adapters/__init__.py dataset_adapters/generic_vqa.py tests/test_vizwiz_support.py
git commit -m "feat: add generic VQA adapter helpers"
```

---

### Task 2: VizWiz Converter

**Files:**
- Create: `convert_vizwiz.py`
- Modify: `tests/test_vizwiz_support.py`

- [ ] **Step 1: Add failing VizWiz converter test**

Append to `tests/test_vizwiz_support.py`:

```python
from PIL import Image

from convert_vizwiz import convert_vizwiz_dataset


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(255, 255, 255)).save(path)


def test_convert_vizwiz_dataset_outputs_generic_records(tmp_path):
    root = tmp_path / "vizwiz"
    train_img = root / "images" / "train" / "VizWiz_train_00000001.jpg"
    val_img = root / "images" / "val" / "VizWiz_val_00000001.jpg"
    _write_image(train_img)
    _write_image(val_img)

    ann_dir = root / "annotations"
    ann_dir.mkdir(parents=True)
    train_annotations = [
        {
            "image": "VizWiz_train_00000001.jpg",
            "question": "What drink is this?",
            "answerable": 1,
            "answer_type": "other",
            "answers": [
                {"answer": "Soda", "answer_confidence": "yes"},
                {"answer": "soda", "answer_confidence": "yes"},
                {"answer": "can", "answer_confidence": "maybe"},
            ],
        }
    ]
    val_annotations = [
        {
            "image": "VizWiz_val_00000001.jpg",
            "question": "Can this be answered?",
            "answerable": 0,
            "answer_type": "unanswerable",
            "answers": [
                {"answer": "unanswerable", "answer_confidence": "yes"},
                {"answer": "unanswerable", "answer_confidence": "yes"},
                {"answer": "text", "answer_confidence": "no"},
            ],
        }
    ]
    (ann_dir / "train.json").write_text(json.dumps(train_annotations))
    (ann_dir / "val.json").write_text(json.dumps(val_annotations))

    convert_vizwiz_dataset(root, output_root=root)

    train = json.loads((root / "train.json").read_text())
    val = json.loads((root / "val.json").read_text())
    answer_list = json.loads((root / "answer_list.json").read_text())
    metadata = json.loads((root / "vizwiz_val_metadata.json").read_text())

    assert train == [
        {
            "dataset": "vizwiz",
            "image": "train/VizWiz_train_00000001.jpg",
            "question": "What drink is this?",
            "question_id": 0,
            "answer": ["soda", "soda", "can"],
        }
    ]
    assert val[0]["image"] == "val/VizWiz_val_00000001.jpg"
    assert val[0]["answer"] == ["unanswerable", "unanswerable", "text"]
    assert answer_list == ["can", "soda", "text", "unanswerable"]
    assert metadata["1000000"]["answerable"] == 0
    assert metadata["1000000"]["answer_type"] == "unanswerable"
```

- [ ] **Step 2: Run converter test to verify it fails**

Run:

```bash
pytest tests/test_vizwiz_support.py::test_convert_vizwiz_dataset_outputs_generic_records -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'convert_vizwiz'`.

- [ ] **Step 3: Implement VizWiz converter**

Create `convert_vizwiz.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from dataset_adapters.generic_vqa import build_answer_list, load_json, normalize_answer, write_json


VAL_ID_OFFSET = 1_000_000


def _answers(raw: dict[str, Any]) -> list[str]:
    answers = raw.get("answers", [])
    if not isinstance(answers, list):
        raise ValueError(f"VizWiz record answers must be a list: {raw}")
    normalized = []
    for answer in answers:
        if isinstance(answer, dict):
            value = answer.get("answer", "")
        else:
            value = answer
        normalized_answer = normalize_answer(value)
        if normalized_answer:
            normalized.append(normalized_answer)
    if not normalized:
        raise ValueError(f"VizWiz record has no usable answers: {raw}")
    return normalized


def _majority_answer(answers: list[str]) -> str:
    counts: dict[str, int] = {}
    for answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _convert_split(
    raw_records: list[dict[str, Any]],
    split: str,
    image_dir: Path,
    id_offset: int,
    include_unanswerable: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    converted: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}

    if not image_dir.is_dir():
        raise FileNotFoundError(f"VizWiz image directory not found: {image_dir}")

    for idx, raw in enumerate(raw_records):
        image_name = raw.get("image")
        question = raw.get("question")
        if not image_name or not question:
            raise ValueError(f"VizWiz record missing image or question: {raw}")
        answers = _answers(raw)
        if not include_unanswerable and _majority_answer(answers) == "unanswerable":
            continue
        image_path = image_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"VizWiz image missing: {image_path}")
        question_id = id_offset + idx
        converted.append(
            {
                "dataset": "vizwiz",
                "image": f"{split}/{image_name}",
                "question": str(question).strip(),
                "question_id": question_id,
                "answer": answers,
            }
        )
        metadata[str(question_id)] = {
            "answerable": int(raw.get("answerable", 1)),
            "answer_type": str(raw.get("answer_type", "unknown")),
        }

    if not converted:
        raise ValueError(f"VizWiz {split} split is empty after conversion")
    return converted, metadata


def convert_vizwiz_dataset(
    vizwiz_root: str | Path,
    *,
    train_annotations: str | Path | None = None,
    val_annotations: str | Path | None = None,
    train_image_dir: str | Path | None = None,
    val_image_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    include_unanswerable: bool = True,
) -> None:
    root = Path(vizwiz_root)
    output_root = Path(output_root) if output_root is not None else root
    train_annotations = Path(train_annotations) if train_annotations else root / "annotations" / "train.json"
    val_annotations = Path(val_annotations) if val_annotations else root / "annotations" / "val.json"
    train_image_dir = Path(train_image_dir) if train_image_dir else root / "images" / "train"
    val_image_dir = Path(val_image_dir) if val_image_dir else root / "images" / "val"

    train_raw = load_json(train_annotations)
    val_raw = load_json(val_annotations)
    if not isinstance(train_raw, list) or not isinstance(val_raw, list):
        raise ValueError("VizWiz annotations must be JSON lists")

    train_records, _ = _convert_split(
        train_raw,
        "train",
        train_image_dir,
        id_offset=0,
        include_unanswerable=include_unanswerable,
    )
    val_records, val_metadata = _convert_split(
        val_raw,
        "val",
        val_image_dir,
        id_offset=VAL_ID_OFFSET,
        include_unanswerable=include_unanswerable,
    )

    write_json(output_root / "train.json", train_records)
    write_json(output_root / "val.json", val_records)
    write_json(output_root / "answer_list.json", build_answer_list(train_records, val_records))
    write_json(output_root / "vizwiz_val_metadata.json", val_metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VizWiz-VQA to SelTDA generic VQA JSON.")
    parser.add_argument("--vizwiz-root", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, default=None)
    parser.add_argument("--val-annotations", type=Path, default=None)
    parser.add_argument("--train-image-dir", type=Path, default=None)
    parser.add_argument("--val-image-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--exclude-unanswerable",
        action="store_true",
        help="Drop records whose majority answer is unanswerable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_vizwiz_dataset(
        args.vizwiz_root,
        train_annotations=args.train_annotations,
        val_annotations=args.val_annotations,
        train_image_dir=args.train_image_dir,
        val_image_dir=args.val_image_dir,
        output_root=args.output_root,
        include_unanswerable=not args.exclude_unanswerable,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run VizWiz converter tests**

Run:

```bash
pytest tests/test_vizwiz_support.py -q
```

Expected: PASS.

- [ ] **Step 5: Check CLI help**

Run:

```bash
python convert_vizwiz.py --help
```

Expected: prints usage including `--vizwiz-root`.

- [ ] **Step 6: Commit**

```bash
git add convert_vizwiz.py tests/test_vizwiz_support.py
git commit -m "feat: convert VizWiz to generic VQA format"
```

---

### Task 3: VizWiz Evaluator

**Files:**
- Create: `vizwiz_eval.py`
- Modify: `tests/test_vizwiz_support.py`

- [ ] **Step 1: Add failing evaluator test**

Append to `tests/test_vizwiz_support.py`:

```python
from vizwiz_eval import evaluate_vizwiz


def test_vizwiz_eval_soft_accuracy_and_strata(tmp_path):
    annotations = [
        {
            "question_id": 1,
            "question": "What is this?",
            "image": "val/a.jpg",
            "dataset": "vizwiz",
            "answer": ["bottle", "bottle", "bottle", "can"],
        },
        {
            "question_id": 2,
            "question": "Can this be answered?",
            "image": "val/b.jpg",
            "dataset": "vizwiz",
            "answer": ["unanswerable", "unanswerable", "text"],
        },
    ]
    metadata = {
        "1": {"answerable": 1, "answer_type": "other"},
        "2": {"answerable": 0, "answer_type": "unanswerable"},
    }
    results = [
        {"question_id": 1, "answer": "Bottle!"},
        {"question_id": 2, "answer": "unanswerable"},
    ]
    ann_path = tmp_path / "val.json"
    meta_path = tmp_path / "vizwiz_val_metadata.json"
    result_path = tmp_path / "vqa_result.json"
    ann_path.write_text(json.dumps(annotations))
    meta_path.write_text(json.dumps(metadata))
    result_path.write_text(json.dumps(results))

    metrics = evaluate_vizwiz(result_path, ann_path, meta_path)

    assert metrics["overall"] == 1.0
    assert metrics["answerable"] == 1.0
    assert metrics["unanswerable"] == 2 / 3
    assert metrics["by_answer_type"]["other"] == 1.0
    assert metrics["by_answer_type"]["unanswerable"] == 2 / 3
```

- [ ] **Step 2: Run evaluator test to verify it fails**

Run:

```bash
pytest tests/test_vizwiz_support.py::test_vizwiz_eval_soft_accuracy_and_strata -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'vizwiz_eval'`.

- [ ] **Step 3: Implement VizWiz evaluator**

Create `vizwiz_eval.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from dataset_adapters.generic_vqa import load_json, vqa_soft_accuracy, write_json


def _prediction_lookup(results: list[dict[str, Any]]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for row in results:
        if "question_id" not in row or "answer" not in row:
            raise ValueError(f"Result row must contain question_id and answer: {row}")
        lookup[int(row["question_id"])] = str(row["answer"])
    return lookup


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def evaluate_vizwiz(
    result_file: str | Path,
    annotation_file: str | Path,
    metadata_file: str | Path,
) -> dict[str, Any]:
    annotations = load_json(annotation_file)
    results = load_json(result_file)
    metadata = load_json(metadata_file)
    if not isinstance(annotations, list) or not isinstance(results, list):
        raise ValueError("VizWiz annotations and results must be JSON lists")

    predictions = _prediction_lookup(results)
    overall_scores: list[float] = []
    answerable_scores: list[float] = []
    unanswerable_scores: list[float] = []
    by_answer_type: dict[str, list[float]] = {}

    for ann in annotations:
        question_id = int(ann["question_id"])
        if question_id not in predictions:
            raise ValueError(f"Missing prediction for question_id={question_id}")
        meta = metadata.get(str(question_id))
        if meta is None:
            raise ValueError(f"Missing VizWiz metadata for question_id={question_id}")
        score = vqa_soft_accuracy(predictions[question_id], ann.get("answer", []))
        overall_scores.append(score)
        answer_type = str(meta.get("answer_type", "unknown"))
        by_answer_type.setdefault(answer_type, []).append(score)
        if int(meta.get("answerable", 1)) == 1:
            answerable_scores.append(score)
        else:
            unanswerable_scores.append(score)

    return {
        "overall": round(_mean(overall_scores), 4),
        "answerable": round(_mean(answerable_scores), 4),
        "unanswerable": round(_mean(unanswerable_scores), 4),
        "by_answer_type": {
            answer_type: round(_mean(scores), 4)
            for answer_type, scores in sorted(by_answer_type.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VizWiz generic VQA predictions.")
    parser.add_argument("result_file", type=Path)
    parser.add_argument("--annotation-file", type=Path, default=Path("datasets/vizwiz/val.json"))
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=Path("datasets/vizwiz/vizwiz_val_metadata.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_vizwiz(args.result_file, args.annotation_file, args.metadata_file)
    print(metrics)
    write_json(Path(args.result_file).parent / "vizwiz_eval.json", metrics)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run VizWiz tests**

Run:

```bash
pytest tests/test_vizwiz_support.py -q
```

Expected: PASS.

- [ ] **Step 5: Check CLI help**

Run:

```bash
python vizwiz_eval.py --help
```

Expected: prints usage including `--metadata-file`.

- [ ] **Step 6: Commit**

```bash
git add vizwiz_eval.py tests/test_vizwiz_support.py
git commit -m "feat: evaluate VizWiz predictions"
```

---

### Task 4: DAQUAR Converter

**Files:**
- Create: `convert_daquar.py`
- Create: `tests/test_daquar_support.py`

- [ ] **Step 1: Write failing DAQUAR converter tests**

Create `tests/test_daquar_support.py`:

```python
import csv
import json
from pathlib import Path

from PIL import Image

from convert_daquar import convert_daquar_dataset


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(255, 255, 255)).save(path)


def test_convert_daquar_dataset_from_json_records(tmp_path):
    root = tmp_path / "daquar"
    _write_image(root / "images" / "image1.png")
    _write_image(root / "images" / "image2.png")
    train = [{"image": "image1.png", "question": "What is on the table?", "answer": "book"}]
    test = [{"image": "image2.png", "question": "Where is the chair?", "answer": "kitchen"}]
    (root / "train.json").write_text(json.dumps(train))
    (root / "test.json").write_text(json.dumps(test))

    convert_daquar_dataset(root, train_qa=root / "train.json", test_qa=root / "test.json")

    converted_train = json.loads((root / "train.json").read_text())
    converted_val = json.loads((root / "val.json").read_text())
    answer_list = json.loads((root / "answer_list.json").read_text())
    metadata = json.loads((root / "daquar_val_metadata.json").read_text())

    assert converted_train == [
        {
            "dataset": "daquar",
            "image": "image1.png",
            "question": "What is on the table?",
            "question_id": 0,
            "answer": ["book"],
        }
    ]
    assert converted_val[0]["question_id"] == 1000000
    assert converted_val[0]["answer"] == ["kitchen"]
    assert answer_list == ["book", "kitchen"]
    assert metadata["1000000"]["answer"] == "kitchen"


def test_convert_daquar_dataset_from_csv_records(tmp_path):
    root = tmp_path / "daquar"
    _write_image(root / "images" / "img_a.png")
    _write_image(root / "images" / "img_b.png")
    train_csv = root / "train.csv"
    test_csv = root / "test.csv"
    with train_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "question", "answer"])
        writer.writeheader()
        writer.writerow({"image": "img_a.png", "question": "What color is the wall?", "answer": "white"})
    with test_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "question", "answer"])
        writer.writeheader()
        writer.writerow({"image": "img_b.png", "question": "Is there a sofa?", "answer": "yes"})

    convert_daquar_dataset(root, train_qa=train_csv, test_qa=test_csv)

    converted_train = json.loads((root / "train.json").read_text())
    converted_val = json.loads((root / "val.json").read_text())
    assert converted_train[0]["answer"] == ["white"]
    assert converted_val[0]["answer"] == ["yes"]
```

- [ ] **Step 2: Run converter tests to verify they fail**

Run:

```bash
pytest tests/test_daquar_support.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'convert_daquar'`.

- [ ] **Step 3: Implement DAQUAR converter**

Create `convert_daquar.py`:

```python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from dataset_adapters.generic_vqa import build_answer_list, load_json, normalize_answer, write_json


VAL_ID_OFFSET = 1_000_000


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"DAQUAR QA file not found: {path}")
    if path.suffix.lower() == ".json":
        data = load_json(path)
        if not isinstance(data, list):
            raise ValueError(f"DAQUAR JSON must contain a list: {path}")
        return [
            {
                "image": str(row.get("image") or row.get("image_id") or row.get("img") or ""),
                "question": str(row.get("question") or row.get("question_str") or ""),
                "answer": str(row.get("answer") or row.get("answers") or ""),
            }
            for row in data
        ]
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        has_header = "question" in sample.lower() and "answer" in sample.lower()
        if has_header:
            reader = csv.DictReader(f, delimiter=delimiter)
            return [
                {
                    "image": str(row.get("image") or row.get("image_id") or row.get("img") or ""),
                    "question": str(row.get("question") or row.get("question_str") or ""),
                    "answer": str(row.get("answer") or row.get("answers") or ""),
                }
                for row in reader
            ]
        reader = csv.reader(f, delimiter=delimiter)
        rows = list(reader)
        if not rows:
            return []
        if len(rows[0]) < 3:
            raise ValueError(f"DAQUAR table rows need image, question, answer columns: {path}")
        return [{"image": row[0], "question": row[1], "answer": row[2]} for row in rows]


def _resolve_image_name(raw_image: str) -> str:
    image = raw_image.strip()
    if not image:
        raise ValueError("DAQUAR record has an empty image field")
    if Path(image).suffix:
        return image
    return f"{image}.png"


def _convert_rows(
    rows: list[dict[str, str]],
    *,
    image_root: Path,
    id_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    converted: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    if not image_root.is_dir():
        raise FileNotFoundError(f"DAQUAR image directory not found: {image_root}")

    for idx, row in enumerate(rows):
        question = row.get("question", "").strip()
        answer = normalize_answer(row.get("answer", ""))
        image = _resolve_image_name(row.get("image", ""))
        if not question or not answer:
            raise ValueError(f"DAQUAR record missing question or answer: {row}")
        if not (image_root / image).is_file():
            raise FileNotFoundError(f"DAQUAR image missing: {image_root / image}")
        question_id = id_offset + idx
        converted.append(
            {
                "dataset": "daquar",
                "image": image,
                "question": question,
                "question_id": question_id,
                "answer": [answer],
            }
        )
        metadata[str(question_id)] = {"answer": answer, "question": question}

    if not converted:
        raise ValueError("DAQUAR split is empty after conversion")
    return converted, metadata


def convert_daquar_dataset(
    daquar_root: str | Path,
    *,
    train_qa: str | Path | None = None,
    test_qa: str | Path | None = None,
    image_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> None:
    root = Path(daquar_root)
    train_qa = Path(train_qa) if train_qa else root / "qa_train.json"
    test_qa = Path(test_qa) if test_qa else root / "qa_test.json"
    image_root = Path(image_root) if image_root else root / "images"
    output_root = Path(output_root) if output_root else root

    train_records, _ = _convert_rows(_read_rows(train_qa), image_root=image_root, id_offset=0)
    val_records, val_metadata = _convert_rows(_read_rows(test_qa), image_root=image_root, id_offset=VAL_ID_OFFSET)

    write_json(output_root / "train.json", train_records)
    write_json(output_root / "val.json", val_records)
    write_json(output_root / "answer_list.json", build_answer_list(train_records, val_records))
    write_json(output_root / "daquar_val_metadata.json", val_metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DAQUAR to SelTDA generic VQA JSON.")
    parser.add_argument("--daquar-root", type=Path, required=True)
    parser.add_argument("--train-qa", type=Path, default=None)
    parser.add_argument("--test-qa", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_daquar_dataset(
        args.daquar_root,
        train_qa=args.train_qa,
        test_qa=args.test_qa,
        image_root=args.image_root,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run DAQUAR converter tests**

Run:

```bash
pytest tests/test_daquar_support.py -q
```

Expected: PASS.

- [ ] **Step 5: Check CLI help**

Run:

```bash
python convert_daquar.py --help
```

Expected: prints usage including `--daquar-root`.

- [ ] **Step 6: Commit**

```bash
git add convert_daquar.py tests/test_daquar_support.py
git commit -m "feat: convert DAQUAR to generic VQA format"
```

---

### Task 5: DAQUAR Evaluator

**Files:**
- Create: `daquar_eval.py`
- Modify: `tests/test_daquar_support.py`

- [ ] **Step 1: Add failing DAQUAR evaluator test**

Append to `tests/test_daquar_support.py`:

```python
from daquar_eval import evaluate_daquar


def test_daquar_eval_exact_match_and_question_prefix(tmp_path):
    annotations = [
        {
            "question_id": 1,
            "question": "What is on the table?",
            "image": "a.png",
            "dataset": "daquar",
            "answer": ["book"],
        },
        {
            "question_id": 2,
            "question": "Where is the chair?",
            "image": "b.png",
            "dataset": "daquar",
            "answer": ["kitchen"],
        },
    ]
    results = [
        {"question_id": 1, "answer": "Book!"},
        {"question_id": 2, "answer": "bedroom"},
    ]
    ann_path = tmp_path / "val.json"
    result_path = tmp_path / "vqa_result.json"
    ann_path.write_text(json.dumps(annotations))
    result_path.write_text(json.dumps(results))

    metrics = evaluate_daquar(result_path, ann_path)

    assert metrics["overall"] == 0.5
    assert metrics["by_question_prefix"]["what"] == 1.0
    assert metrics["by_question_prefix"]["where"] == 0.0
```

- [ ] **Step 2: Run evaluator test to verify it fails**

Run:

```bash
pytest tests/test_daquar_support.py::test_daquar_eval_exact_match_and_question_prefix -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'daquar_eval'`.

- [ ] **Step 3: Implement DAQUAR evaluator**

Create `daquar_eval.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from dataset_adapters.generic_vqa import load_json, normalize_answer, question_prefix, write_json


def _prediction_lookup(results: list[dict[str, Any]]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for row in results:
        if "question_id" not in row or "answer" not in row:
            raise ValueError(f"Result row must contain question_id and answer: {row}")
        lookup[int(row["question_id"])] = normalize_answer(row["answer"])
    return lookup


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def evaluate_daquar(
    result_file: str | Path,
    annotation_file: str | Path,
) -> dict[str, Any]:
    annotations = load_json(annotation_file)
    results = load_json(result_file)
    if not isinstance(annotations, list) or not isinstance(results, list):
        raise ValueError("DAQUAR annotations and results must be JSON lists")

    predictions = _prediction_lookup(results)
    scores: list[float] = []
    by_prefix: dict[str, list[float]] = {}

    for ann in annotations:
        question_id = int(ann["question_id"])
        if question_id not in predictions:
            raise ValueError(f"Missing prediction for question_id={question_id}")
        refs = {normalize_answer(answer) for answer in ann.get("answer", [])}
        score = 1.0 if predictions[question_id] in refs else 0.0
        scores.append(score)
        prefix = question_prefix(str(ann.get("question", "")))
        by_prefix.setdefault(prefix, []).append(score)

    return {
        "overall": round(_mean(scores), 4),
        "by_question_prefix": {
            prefix: round(_mean(prefix_scores), 4)
            for prefix, prefix_scores in sorted(by_prefix.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DAQUAR generic VQA predictions.")
    parser.add_argument("result_file", type=Path)
    parser.add_argument("--annotation-file", type=Path, default=Path("datasets/daquar/val.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_daquar(args.result_file, args.annotation_file)
    print(metrics)
    write_json(Path(args.result_file).parent / "daquar_eval.json", metrics)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run DAQUAR tests**

Run:

```bash
pytest tests/test_daquar_support.py -q
```

Expected: PASS.

- [ ] **Step 5: Check CLI help**

Run:

```bash
python daquar_eval.py --help
```

Expected: prints usage including `--annotation-file`.

- [ ] **Step 6: Commit**

```bash
git add daquar_eval.py tests/test_daquar_support.py
git commit -m "feat: evaluate DAQUAR predictions"
```

---

### Task 6: Dataset Configs

**Files:**
- Create: `configs/vizwiz.yaml`
- Create: `configs/daquar.yaml`
- Modify: `tests/test_vizwiz_support.py`
- Modify: `tests/test_daquar_support.py`

- [ ] **Step 1: Add failing config smoke tests**

Append to `tests/test_vizwiz_support.py`:

```python
from argparse import Namespace

import cli


def test_vizwiz_config_composes_with_local_overrides(tmp_path):
    args = Namespace(
        config="configs/vizwiz.yaml",
        overrides=[
            f"ann_root='{tmp_path}'",
            f"vqa_root='{tmp_path / 'images'}'",
            "wandb=false",
            "torch_home=null",
        ],
    )
    config = cli.load_config(args)
    assert config.dataset_name == "generic_vqa"
    assert list(config.train_files) == ["train"]
    assert config.val_file == "val"
```

Append to `tests/test_daquar_support.py`:

```python
from argparse import Namespace

import cli


def test_daquar_config_composes_with_local_overrides(tmp_path):
    args = Namespace(
        config="configs/daquar.yaml",
        overrides=[
            f"ann_root='{tmp_path}'",
            f"vqa_root='{tmp_path / 'images'}'",
            "wandb=false",
            "torch_home=null",
        ],
    )
    config = cli.load_config(args)
    assert config.dataset_name == "generic_vqa"
    assert list(config.train_files) == ["train"]
    assert config.val_file == "val"
```

- [ ] **Step 2: Run config tests to verify they fail**

Run:

```bash
pytest tests/test_vizwiz_support.py::test_vizwiz_config_composes_with_local_overrides tests/test_daquar_support.py::test_daquar_config_composes_with_local_overrides -q
```

Expected: FAIL with Hydra missing config errors for `vizwiz` and `daquar`.

- [ ] **Step 3: Add VizWiz config**

Create `configs/vizwiz.yaml`:

```yaml
ann_root: datasets/vizwiz
vqa_root: datasets/vizwiz/images
train_files: ['train']
val_file: val
dataset_name: generic_vqa
answer_list: answer_list
truncate_train_dataset_to: null

pretrained: 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
vit: 'base'
batch_size_train: 16
batch_size_test: 16
vit_grad_ckpt: false
vit_ckpt_layer: 0
init_lr: 2e-5
image_size: 480
k_test: 128
inference: 'rank'

weight_decay: 0.05
min_lr: 0
max_epoch: 10
torch_home: null
wandb: false
save_last_only: false
max_checkpoints: 3
```

- [ ] **Step 4: Add DAQUAR config**

Create `configs/daquar.yaml`:

```yaml
ann_root: datasets/daquar
vqa_root: datasets/daquar/images
train_files: ['train']
val_file: val
dataset_name: generic_vqa
answer_list: answer_list
truncate_train_dataset_to: null

pretrained: 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
vit: 'base'
batch_size_train: 16
batch_size_test: 16
vit_grad_ckpt: false
vit_ckpt_layer: 0
init_lr: 2e-5
image_size: 480
k_test: 128
inference: 'rank'

weight_decay: 0.05
min_lr: 0
max_epoch: 10
torch_home: null
wandb: false
save_last_only: false
max_checkpoints: 3
```

- [ ] **Step 5: Run config tests**

Run:

```bash
pytest tests/test_vizwiz_support.py::test_vizwiz_config_composes_with_local_overrides tests/test_daquar_support.py::test_daquar_config_composes_with_local_overrides -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/vizwiz.yaml configs/daquar.yaml tests/test_vizwiz_support.py tests/test_daquar_support.py
git commit -m "feat: add VizWiz and DAQUAR configs"
```

---

### Task 7: VizWiz Experiment Script

**Files:**
- Create: `examples/run_vizwiz_experiment.sh`

- [ ] **Step 1: Create VizWiz experiment script**

Create `examples/run_vizwiz_experiment.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

bool_value() {
    case "${1:-0}" in
        1|true|TRUE|yes|YES|on|ON) echo true ;;
        *) echo false ;;
    esac
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

checkpoint_name() {
    local epoch_count="$1"
    local idx=$((epoch_count - 1))
    if [ "${idx}" -lt 0 ]; then
        die "epoch count must be >= 1, got ${epoch_count}"
    fi
    printf 'checkpoint_%02d.pth' "${idx}"
}

DATASETS_DIR="${DATASETS_DIR:-${PROJECT_ROOT}/datasets}"
VIZWIZ_DIR="${VIZWIZ_DIR:-${DATASETS_DIR}/vizwiz}"
VIZWIZ_IMAGES="${VIZWIZ_IMAGES:-${VIZWIZ_DIR}/images}"
VIZWIZ_TRAIN_ANNOTATIONS="${VIZWIZ_TRAIN_ANNOTATIONS:-${VIZWIZ_DIR}/annotations/train.json}"
VIZWIZ_VAL_ANNOTATIONS="${VIZWIZ_VAL_ANNOTATIONS:-${VIZWIZ_DIR}/annotations/val.json}"
INCLUDE_UNANSWERABLE="$(bool_value "${INCLUDE_UNANSWERABLE:-1}")"

NUM_GPUS="${NUM_GPUS:-1}"
VQA_EPOCHS="${VQA_EPOCHS:-10}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-16}"
VQA_BATCH_SIZE_TRAIN="${VQA_BATCH_SIZE_TRAIN:-16}"
VQA_BATCH_SIZE_TEST="${VQA_BATCH_SIZE_TEST:-16}"
FILTER_KEEP_TOP="${FILTER_KEEP_TOP:-0.75}"
TRUNCATE_GENERATE="${TRUNCATE_GENERATE:-}"
TORCH_HOME_OVERRIDE="${TORCH_HOME_OVERRIDE:-null}"

BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-${PROJECT_ROOT}/cache/vizwiz_baseline_weights}"
STUDENT_OUTPUT_DIR="${STUDENT_OUTPUT_DIR:-${PROJECT_ROOT}/cache/vizwiz_self_trained_weights}"
GEN_OUTPUT_DIR="${GEN_OUTPUT_DIR:-${PROJECT_ROOT}/cache/vizwiz_generation}"
FILTER_OUTPUT_DIR="${FILTER_OUTPUT_DIR:-${PROJECT_ROOT}/cache/vizwiz_filter}"
BASELINE_CKPT="${BASELINE_OUTPUT_DIR}/$(checkpoint_name "${VQA_EPOCHS}")"

SYNTH_RAW="${SYNTH_RAW:-${VIZWIZ_DIR}/synthetic_data_raw.json}"
SYNTH_FILTERED="${SYNTH_FILTERED:-${VIZWIZ_DIR}/synthetic_data.json}"
FILTER_REPORT="${FILTER_REPORT:-${VIZWIZ_DIR}/filter_report.json}"
SCORE_CACHE_DIR="${SCORE_CACHE_DIR:-${VIZWIZ_DIR}/score_cache}"
UNLABELED_ANNOTATIONS="${UNLABELED_ANNOTATIONS:-${VIZWIZ_DIR}/train.json}"

ENABLE_CONF="$(bool_value "${ENABLE_CONF:-1}")"
ENABLE_ITM="$(bool_value "${ENABLE_ITM:-1}")"
ENABLE_XCONS="$(bool_value "${ENABLE_XCONS:-1}")"
RUN_BASELINE="$(bool_value "${RUN_BASELINE:-1}")"

mkdir -p "${BASELINE_OUTPUT_DIR}" "${STUDENT_OUTPUT_DIR}" "${GEN_OUTPUT_DIR}" "${FILTER_OUTPUT_DIR}"

echo "========== Step 1: Convert VizWiz =========="
if [ "${SKIP_CONVERT:-0}" != "1" ]; then
    [ -f "${VIZWIZ_TRAIN_ANNOTATIONS}" ] || die "Missing ${VIZWIZ_TRAIN_ANNOTATIONS}"
    [ -f "${VIZWIZ_VAL_ANNOTATIONS}" ] || die "Missing ${VIZWIZ_VAL_ANNOTATIONS}"
    [ -d "${VIZWIZ_IMAGES}/train" ] || die "Missing ${VIZWIZ_IMAGES}/train"
    [ -d "${VIZWIZ_IMAGES}/val" ] || die "Missing ${VIZWIZ_IMAGES}/val"
    CONVERT_ARGS=()
    if [ "${INCLUDE_UNANSWERABLE}" != "true" ]; then
        CONVERT_ARGS+=(--exclude-unanswerable)
    fi
    python convert_vizwiz.py \
        --vizwiz-root "${VIZWIZ_DIR}" \
        --train-annotations "${VIZWIZ_TRAIN_ANNOTATIONS}" \
        --val-annotations "${VIZWIZ_VAL_ANNOTATIONS}" \
        --train-image-dir "${VIZWIZ_IMAGES}/train" \
        --val-image-dir "${VIZWIZ_IMAGES}/val" \
        --output-root "${VIZWIZ_DIR}" \
        "${CONVERT_ARGS[@]}"
else
    echo "SKIP_CONVERT=1 - reusing converted VizWiz JSON files"
fi

for file in train.json val.json answer_list.json vizwiz_val_metadata.json; do
    [ -f "${VIZWIZ_DIR}/${file}" ] || die "Missing converted file: ${VIZWIZ_DIR}/${file}"
done

echo "========== Step 2: Train VizWiz real-only baseline =========="
if [ "${RUN_BASELINE}" = "true" ]; then
    if [ ! -f "${BASELINE_CKPT}" ]; then
        python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" train_vqa.py \
            --output_dir="${BASELINE_OUTPUT_DIR}" \
            --config configs/vizwiz.yaml \
            --overrides \
                "ann_root='${VIZWIZ_DIR}'" \
                "vqa_root='${VIZWIZ_IMAGES}'" \
                "train_files=[train]" \
                "batch_size_train=${VQA_BATCH_SIZE_TRAIN}" \
                "batch_size_test=${VQA_BATCH_SIZE_TEST}" \
                "max_epoch=${VQA_EPOCHS}" \
                "torch_home=${TORCH_HOME_OVERRIDE}" \
                "wandb=false"
    else
        echo "Baseline checkpoint exists: ${BASELINE_CKPT}"
    fi
else
    echo "RUN_BASELINE=0 - skipping baseline train"
fi

echo "========== Step 3: Generate VizWiz pseudo-QA =========="
if [ "${SKIP_GENERATE:-0}" != "1" ]; then
    [ -f "${UNLABELED_ANNOTATIONS}" ] || die "Missing UNLABELED_ANNOTATIONS=${UNLABELED_ANNOTATIONS}"
    if [ "${UNLABELED_ANNOTATIONS}" != "${VIZWIZ_DIR}/train.json" ]; then
        echo "Using custom UNLABELED_ANNOTATIONS=${UNLABELED_ANNOTATIONS}"
    fi
    GEN_OVERRIDES=(
        "image_folder='${VIZWIZ_IMAGES}'"
        "output_folder='${VIZWIZ_DIR}'"
        "annotations='${UNLABELED_ANNOTATIONS}'"
        "output_annotations_name=$(basename "${SYNTH_RAW}")"
        "batch_size=${GEN_BATCH_SIZE}"
        "num_workers=4"
        "vqa_dataset_origin=vqa"
        "shuffle=true"
        "torch_home=${TORCH_HOME_OVERRIDE}"
    )
    if [ -n "${TRUNCATE_GENERATE}" ]; then
        GEN_OVERRIDES+=("truncate_to=${TRUNCATE_GENERATE}")
    fi
    python generate_questions.py \
        --output_dir="${GEN_OUTPUT_DIR}" \
        --config configs/generate_questions_pathvqa.yaml \
        --overrides "${GEN_OVERRIDES[@]}"
else
    echo "SKIP_GENERATE=1 - reusing ${SYNTH_RAW}"
fi
[ -f "${SYNTH_RAW}" ] || die "Missing raw synthetic file: ${SYNTH_RAW}"

echo "========== Step 4: Filter VizWiz pseudo-QA =========="
if [ "${SKIP_FILTER:-0}" != "1" ]; then
    if [ "${ENABLE_XCONS}" = "true" ] && [ ! -f "${BASELINE_CKPT}" ]; then
        die "X-consistency enabled but baseline checkpoint missing: ${BASELINE_CKPT}"
    fi
    python filter_pseudo.py \
        --output_dir="${FILTER_OUTPUT_DIR}" \
        --config configs/filter_pseudo.yaml \
        --overrides \
            "input='${SYNTH_RAW}'" \
            "image_root='${VIZWIZ_IMAGES}'" \
            "output='${SYNTH_FILTERED}'" \
            "report='${FILTER_REPORT}'" \
            "gates.conf.enabled=${ENABLE_CONF}" \
            "gates.conf.keep_top=${FILTER_KEEP_TOP}" \
            "gates.itm.enabled=${ENABLE_ITM}" \
            "gates.itm.keep_top=${FILTER_KEEP_TOP}" \
            "gates.xcons.enabled=${ENABLE_XCONS}" \
            "gates.xcons.keep_top=${FILTER_KEEP_TOP}" \
            "gates.xcons.student_ckpt='${BASELINE_CKPT}'" \
            "score_cache.dir='${SCORE_CACHE_DIR}'"
else
    echo "SKIP_FILTER=1 - reusing ${SYNTH_FILTERED}"
fi
[ -f "${SYNTH_FILTERED}" ] || die "Missing filtered synthetic file: ${SYNTH_FILTERED}"

echo "========== Step 5: Train VizWiz augmented student =========="
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" train_vqa.py \
        --output_dir="${STUDENT_OUTPUT_DIR}" \
        --config configs/vizwiz.yaml \
        --overrides \
            "ann_root='${VIZWIZ_DIR}'" \
            "vqa_root='${VIZWIZ_IMAGES}'" \
            "train_files=[train,$(basename "${SYNTH_FILTERED}" .json)]" \
            "batch_size_train=${VQA_BATCH_SIZE_TRAIN}" \
            "batch_size_test=${VQA_BATCH_SIZE_TEST}" \
            "max_epoch=${VQA_EPOCHS}" \
            "torch_home=${TORCH_HOME_OVERRIDE}" \
            "wandb=false"
else
    echo "SKIP_TRAIN=1 - skipping student train"
fi

echo "========== Step 6: Evaluate VizWiz =========="
RESULT_FILE="${STUDENT_OUTPUT_DIR}/result/vqa_result.json"
if [ "${SKIP_EVAL:-0}" != "1" ] && [ -f "${RESULT_FILE}" ]; then
    python vizwiz_eval.py \
        --annotation-file "${VIZWIZ_DIR}/val.json" \
        --metadata-file "${VIZWIZ_DIR}/vizwiz_val_metadata.json" \
        "${RESULT_FILE}"
fi
```

- [ ] **Step 2: Make script executable**

Run:

```bash
chmod +x examples/run_vizwiz_experiment.sh
```

Expected: command exits with status 0.

- [ ] **Step 3: Run shell syntax check**

Run:

```bash
bash -n examples/run_vizwiz_experiment.sh
```

Expected: no output and exit status 0.

- [ ] **Step 4: Commit**

```bash
git add examples/run_vizwiz_experiment.sh
git commit -m "feat: add VizWiz experiment script"
```

---

### Task 8: DAQUAR Experiment Script

**Files:**
- Create: `examples/run_daquar_experiment.sh`

- [ ] **Step 1: Create DAQUAR experiment script**

Create `examples/run_daquar_experiment.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

bool_value() {
    case "${1:-0}" in
        1|true|TRUE|yes|YES|on|ON) echo true ;;
        *) echo false ;;
    esac
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

checkpoint_name() {
    local epoch_count="$1"
    local idx=$((epoch_count - 1))
    if [ "${idx}" -lt 0 ]; then
        die "epoch count must be >= 1, got ${epoch_count}"
    fi
    printf 'checkpoint_%02d.pth' "${idx}"
}

DATASETS_DIR="${DATASETS_DIR:-${PROJECT_ROOT}/datasets}"
DAQUAR_DIR="${DAQUAR_DIR:-${DATASETS_DIR}/daquar}"
DAQUAR_IMAGES="${DAQUAR_IMAGES:-${DAQUAR_DIR}/images}"
DAQUAR_TRAIN_QA="${DAQUAR_TRAIN_QA:-${DAQUAR_DIR}/qa_train.json}"
DAQUAR_TEST_QA="${DAQUAR_TEST_QA:-${DAQUAR_DIR}/qa_test.json}"

NUM_GPUS="${NUM_GPUS:-1}"
VQA_EPOCHS="${VQA_EPOCHS:-10}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-16}"
VQA_BATCH_SIZE_TRAIN="${VQA_BATCH_SIZE_TRAIN:-16}"
VQA_BATCH_SIZE_TEST="${VQA_BATCH_SIZE_TEST:-16}"
FILTER_KEEP_TOP="${FILTER_KEEP_TOP:-0.75}"
TRUNCATE_GENERATE="${TRUNCATE_GENERATE:-}"
TORCH_HOME_OVERRIDE="${TORCH_HOME_OVERRIDE:-null}"

BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-${PROJECT_ROOT}/cache/daquar_baseline_weights}"
STUDENT_OUTPUT_DIR="${STUDENT_OUTPUT_DIR:-${PROJECT_ROOT}/cache/daquar_self_trained_weights}"
GEN_OUTPUT_DIR="${GEN_OUTPUT_DIR:-${PROJECT_ROOT}/cache/daquar_generation}"
FILTER_OUTPUT_DIR="${FILTER_OUTPUT_DIR:-${PROJECT_ROOT}/cache/daquar_filter}"
BASELINE_CKPT="${BASELINE_OUTPUT_DIR}/$(checkpoint_name "${VQA_EPOCHS}")"

SYNTH_RAW="${SYNTH_RAW:-${DAQUAR_DIR}/synthetic_data_raw.json}"
SYNTH_FILTERED="${SYNTH_FILTERED:-${DAQUAR_DIR}/synthetic_data.json}"
FILTER_REPORT="${FILTER_REPORT:-${DAQUAR_DIR}/filter_report.json}"
SCORE_CACHE_DIR="${SCORE_CACHE_DIR:-${DAQUAR_DIR}/score_cache}"
UNLABELED_ANNOTATIONS="${UNLABELED_ANNOTATIONS:-${DAQUAR_DIR}/train.json}"

ENABLE_CONF="$(bool_value "${ENABLE_CONF:-1}")"
ENABLE_ITM="$(bool_value "${ENABLE_ITM:-1}")"
ENABLE_XCONS="$(bool_value "${ENABLE_XCONS:-1}")"
RUN_BASELINE="$(bool_value "${RUN_BASELINE:-1}")"

mkdir -p "${BASELINE_OUTPUT_DIR}" "${STUDENT_OUTPUT_DIR}" "${GEN_OUTPUT_DIR}" "${FILTER_OUTPUT_DIR}"

echo "========== Step 1: Convert DAQUAR =========="
if [ "${SKIP_CONVERT:-0}" != "1" ]; then
    [ -f "${DAQUAR_TRAIN_QA}" ] || die "Missing ${DAQUAR_TRAIN_QA}"
    [ -f "${DAQUAR_TEST_QA}" ] || die "Missing ${DAQUAR_TEST_QA}"
    [ -d "${DAQUAR_IMAGES}" ] || die "Missing ${DAQUAR_IMAGES}"
    python convert_daquar.py \
        --daquar-root "${DAQUAR_DIR}" \
        --train-qa "${DAQUAR_TRAIN_QA}" \
        --test-qa "${DAQUAR_TEST_QA}" \
        --image-root "${DAQUAR_IMAGES}" \
        --output-root "${DAQUAR_DIR}"
else
    echo "SKIP_CONVERT=1 - reusing converted DAQUAR JSON files"
fi

for file in train.json val.json answer_list.json daquar_val_metadata.json; do
    [ -f "${DAQUAR_DIR}/${file}" ] || die "Missing converted file: ${DAQUAR_DIR}/${file}"
done

echo "========== Step 2: Train DAQUAR real-only baseline =========="
if [ "${RUN_BASELINE}" = "true" ]; then
    if [ ! -f "${BASELINE_CKPT}" ]; then
        python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" train_vqa.py \
            --output_dir="${BASELINE_OUTPUT_DIR}" \
            --config configs/daquar.yaml \
            --overrides \
                "ann_root='${DAQUAR_DIR}'" \
                "vqa_root='${DAQUAR_IMAGES}'" \
                "train_files=[train]" \
                "batch_size_train=${VQA_BATCH_SIZE_TRAIN}" \
                "batch_size_test=${VQA_BATCH_SIZE_TEST}" \
                "max_epoch=${VQA_EPOCHS}" \
                "torch_home=${TORCH_HOME_OVERRIDE}" \
                "wandb=false"
    else
        echo "Baseline checkpoint exists: ${BASELINE_CKPT}"
    fi
else
    echo "RUN_BASELINE=0 - skipping baseline train"
fi

echo "========== Step 3: Generate DAQUAR pseudo-QA =========="
if [ "${SKIP_GENERATE:-0}" != "1" ]; then
    [ -f "${UNLABELED_ANNOTATIONS}" ] || die "Missing UNLABELED_ANNOTATIONS=${UNLABELED_ANNOTATIONS}"
    if [ "${UNLABELED_ANNOTATIONS}" != "${DAQUAR_DIR}/train.json" ]; then
        echo "Using custom UNLABELED_ANNOTATIONS=${UNLABELED_ANNOTATIONS}"
    fi
    GEN_OVERRIDES=(
        "image_folder='${DAQUAR_IMAGES}'"
        "output_folder='${DAQUAR_DIR}'"
        "annotations='${UNLABELED_ANNOTATIONS}'"
        "output_annotations_name=$(basename "${SYNTH_RAW}")"
        "batch_size=${GEN_BATCH_SIZE}"
        "num_workers=4"
        "vqa_dataset_origin=vqa"
        "shuffle=true"
        "torch_home=${TORCH_HOME_OVERRIDE}"
    )
    if [ -n "${TRUNCATE_GENERATE}" ]; then
        GEN_OVERRIDES+=("truncate_to=${TRUNCATE_GENERATE}")
    fi
    python generate_questions.py \
        --output_dir="${GEN_OUTPUT_DIR}" \
        --config configs/generate_questions_pathvqa.yaml \
        --overrides "${GEN_OVERRIDES[@]}"
else
    echo "SKIP_GENERATE=1 - reusing ${SYNTH_RAW}"
fi
[ -f "${SYNTH_RAW}" ] || die "Missing raw synthetic file: ${SYNTH_RAW}"

echo "========== Step 4: Filter DAQUAR pseudo-QA =========="
if [ "${SKIP_FILTER:-0}" != "1" ]; then
    if [ "${ENABLE_XCONS}" = "true" ] && [ ! -f "${BASELINE_CKPT}" ]; then
        die "X-consistency enabled but baseline checkpoint missing: ${BASELINE_CKPT}"
    fi
    python filter_pseudo.py \
        --output_dir="${FILTER_OUTPUT_DIR}" \
        --config configs/filter_pseudo.yaml \
        --overrides \
            "input='${SYNTH_RAW}'" \
            "image_root='${DAQUAR_IMAGES}'" \
            "output='${SYNTH_FILTERED}'" \
            "report='${FILTER_REPORT}'" \
            "gates.conf.enabled=${ENABLE_CONF}" \
            "gates.conf.keep_top=${FILTER_KEEP_TOP}" \
            "gates.itm.enabled=${ENABLE_ITM}" \
            "gates.itm.keep_top=${FILTER_KEEP_TOP}" \
            "gates.xcons.enabled=${ENABLE_XCONS}" \
            "gates.xcons.keep_top=${FILTER_KEEP_TOP}" \
            "gates.xcons.student_ckpt='${BASELINE_CKPT}'" \
            "score_cache.dir='${SCORE_CACHE_DIR}'"
else
    echo "SKIP_FILTER=1 - reusing ${SYNTH_FILTERED}"
fi
[ -f "${SYNTH_FILTERED}" ] || die "Missing filtered synthetic file: ${SYNTH_FILTERED}"

echo "========== Step 5: Train DAQUAR augmented student =========="
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" train_vqa.py \
        --output_dir="${STUDENT_OUTPUT_DIR}" \
        --config configs/daquar.yaml \
        --overrides \
            "ann_root='${DAQUAR_DIR}'" \
            "vqa_root='${DAQUAR_IMAGES}'" \
            "train_files=[train,$(basename "${SYNTH_FILTERED}" .json)]" \
            "batch_size_train=${VQA_BATCH_SIZE_TRAIN}" \
            "batch_size_test=${VQA_BATCH_SIZE_TEST}" \
            "max_epoch=${VQA_EPOCHS}" \
            "torch_home=${TORCH_HOME_OVERRIDE}" \
            "wandb=false"
else
    echo "SKIP_TRAIN=1 - skipping student train"
fi

echo "========== Step 6: Evaluate DAQUAR =========="
RESULT_FILE="${STUDENT_OUTPUT_DIR}/result/vqa_result.json"
if [ "${SKIP_EVAL:-0}" != "1" ] && [ -f "${RESULT_FILE}" ]; then
    python daquar_eval.py \
        --annotation-file "${DAQUAR_DIR}/val.json" \
        "${RESULT_FILE}"
fi
```

- [ ] **Step 2: Make script executable**

Run:

```bash
chmod +x examples/run_daquar_experiment.sh
```

Expected: command exits with status 0.

- [ ] **Step 3: Run shell syntax check**

Run:

```bash
bash -n examples/run_daquar_experiment.sh
```

Expected: no output and exit status 0.

- [ ] **Step 4: Commit**

```bash
git add examples/run_daquar_experiment.sh
git commit -m "feat: add DAQUAR experiment script"
```

---

### Task 9: Final Verification

**Files:**
- Verify only.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
pytest tests/test_vizwiz_support.py tests/test_daquar_support.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run CLI help checks**

Run:

```bash
python convert_vizwiz.py --help
python vizwiz_eval.py --help
python convert_daquar.py --help
python daquar_eval.py --help
```

Expected: each command prints usage and exits with status 0.

- [ ] **Step 3: Run config compose checks**

Run:

```bash
python -c "from argparse import Namespace; import cli; c=cli.load_config(Namespace(config='configs/vizwiz.yaml', overrides=['wandb=false','torch_home=null'])); print(c.dataset_name, c.val_file)"
python -c "from argparse import Namespace; import cli; c=cli.load_config(Namespace(config='configs/daquar.yaml', overrides=['wandb=false','torch_home=null'])); print(c.dataset_name, c.val_file)"
```

Expected: both commands print `generic_vqa val`.

- [ ] **Step 4: Run shell syntax checks**

Run:

```bash
bash -n examples/run_vizwiz_experiment.sh
bash -n examples/run_daquar_experiment.sh
```

Expected: no output and exit status 0 for both scripts.

- [ ] **Step 5: Run full regression subset that covers dataset loading**

Run:

```bash
pytest tests/test_datasets.py tests/test_cli.py tests/test_train_vqa_resume.py -q
```

Expected: all selected tests pass. If local fixture data required by old dataset tests is absent, record the missing fixture error and run `pytest tests/test_cli.py tests/test_train_vqa_resume.py tests/test_vizwiz_support.py tests/test_daquar_support.py -q` instead.

- [ ] **Step 6: Inspect git diff for unrelated changes**

Run:

```bash
git status --short
git diff --stat
```

Expected: only files from this plan are changed or newly added. Pre-existing unrelated files such as notebooks, `.codex/`, or checkpoint directories must not be staged.
