# PathVQA Kaggle Download Notebook Design

## Objective

Create a self-contained Kaggle notebook named `download_pathvqa_kaggle.ipynb` that downloads the public PathVQA dataset from Hugging Face, materializes its images, and writes annotations that can be consumed directly by the existing SelTDA PathVQA pipeline.

The notebook must not clone or import the SelTDA repository at runtime. It downloads only the PathVQA dataset; it does not download teacher checkpoints or pre-generated pseudo-QA.

## Target Environment

- Runtime: Kaggle Notebook with Internet enabled.
- GPU: not required.
- Dataset source: `flaviagiammarino/path-vqa` on Hugging Face.
- Default output root: `/kaggle/working/pathvqa`.
- Optional archive: `/kaggle/working/pathvqa.zip`, controlled by `CREATE_ARCHIVE=False`.

## Notebook Structure

### 1. Introduction

A Markdown cell explains:

- the notebook downloads and converts PathVQA;
- Kaggle Internet must be enabled;
- no GPU is required;
- all outputs are written under `/kaggle/working/pathvqa`;
- rerunning the notebook does not overwrite existing images unless explicitly requested.

### 2. Configuration

A code cell exposes the user-facing settings:

```python
DATASET_ID = "flaviagiammarino/path-vqa"
OUTPUT_ROOT = Path("/kaggle/working/pathvqa")
JPEG_QUALITY = 95
OVERWRITE_IMAGES = False
CREATE_ARCHIVE = False
```

### 3. Dependencies

Install only the packages needed by the notebook:

- `datasets`
- `huggingface-hub`
- `Pillow`
- `tqdm`

Use `sys.executable -m pip` so installation targets the active Kaggle kernel.

### 4. Conversion Helpers

Define self-contained helpers for:

- mapping Hugging Face splits `train`, `validation`, and `test` to `train`, `val`, and `test`;
- inferring broad answer types (`yes/no`, `number`, and `other`);
- inferring broad question types from question prefixes;
- converting a PIL image to RGB and saving it as JPEG;
- building raw PathVQA-compatible and SelTDA-compatible annotation records;
- writing UTF-8 JSON files;
- validating image references and annotation counts.

The helpers must depend only on standard Python, Pillow, `datasets`, and `tqdm`.

### 5. Download and Materialization

Load the Hugging Face dataset and process each split sequentially. Each image receives a deterministic filename:

```text
train_000000.jpg
val_000000.jpg
test_000000.jpg
```

Images are saved under the corresponding split directory. If an image already exists and `OVERWRITE_IMAGES` is false, image encoding is skipped. Annotations are rebuilt on every run so a previously interrupted run can recover consistently.

Question IDs are unique and monotonically increasing across all splits.

## Output Contract

The notebook writes:

```text
/kaggle/working/pathvqa/
├── images/
│   ├── train/*.jpg
│   ├── val/*.jpg
│   └── test/*.jpg
├── all_data.json
├── train.json
├── val.json
├── test.json
├── answer_list.json
└── test_val_combined.json
```

### `all_data.json`

Contains the six keys expected by the existing PathVQA converter:

- `train_qa`, `train_vqa`
- `val_qa`, `val_vqa`
- `test_qa`, `test_vqa`

### SelTDA split annotations

- `train.json` contains `image`, `question`, `answer` as a one-element list, `dataset`, and `question_id`.
- `val.json` and `test.json` contain `image`, `question`, scalar `answer`, `dataset`, `question_id`, `question_type`, and `answer_type`.
- Image paths are relative to the `images` directory, for example `train/train_000000.jpg`.
- `answer_list.json` contains the sorted unique test answers.
- `test_val_combined.json` contains one evaluation record per unique image across val and test.

## Validation and User Feedback

After writing the data, the notebook must:

1. compare annotation counts against the Hugging Face split sizes;
2. assert that QA and VQA raw record counts match within each split;
3. assert that every annotation references an existing image;
4. assert that all question IDs are unique;
5. print a compact split summary;
6. display one image with its question and answer.

Validation failures must raise a clear exception rather than silently producing a partial success message.

## Optional Archive

When `CREATE_ARCHIVE=True`, create `pathvqa.zip` after validation succeeds. Archiving is disabled by default to avoid duplicating the dataset in Kaggle storage.

## Verification Strategy

Repository-side verification does not require downloading the full dataset:

- parse the notebook as valid JSON;
- parse every code cell with Python `ast`;
- confirm required Markdown and configuration cells exist;
- confirm no checkpoint or pseudo-QA Google Drive IDs appear;
- confirm expected output filenames and validation assertions appear;
- run small in-memory smoke checks for answer-type and question-type inference where practical.

A real end-to-end download remains a Kaggle manual validation because it requires network access and materializes the full dataset.

## Non-Goals

- Downloading SelTDA teacher checkpoints.
- Downloading existing synthetic or pseudo-QA files.
- Training or evaluating a VQA model.
- Cloning the SelTDA repository inside Kaggle.
- Supporting Colab-specific Google Drive mounting.
