# VizWiz and DAQUAR Dataset Support Design

Date: 2026-07-06

## Goal

Add first-class experiment support for two additional low-resource, domain-specific VQA datasets:

- **VizWiz-VQA**: accessibility domain, real images and spoken questions from people who are blind.
- **DAQUAR**: indoor RGB-D scene domain, small classic VQA benchmark built on NYU Depth v2.

This supports the thesis direction: data augmentation for low-resource domain-specific VQA. PathVQA remains the main medical dataset; VizWiz and DAQUAR provide cross-domain evidence without introducing OCR-specific complexity.

## Scope

In scope:

- Convert VizWiz and DAQUAR into the repo's existing JSON format.
- Add dataset configs using the existing `generic_vqa` reader.
- Add evaluation scripts for each dataset.
- Add end-to-end experiment scripts following the PathVQA script structure: convert, train baseline, generate pseudo-QA, filter, train augmented student, evaluate.
- Add focused tests for converters and evaluators using tiny synthetic fixtures.

Out of scope for this phase:

- TextVQA support. It needs an OCR-aware design and would change the model assumptions.
- DAQUAR WUPS metric. The first implementation uses exact-match accuracy because it is consistent with current PathVQA/RSVQA-style evaluation.
- Changes to `train_vqa.py`, model architecture, loss functions, or BLIP internals.
- Downloading full datasets during tests.

## Existing Repo Fit

The repo already has a generic VQA reader:

- `data.vqa_dataset.GenericVqaDataset`
- `configs/generic.yaml`
- `data.__init__.create_dataset(... dataset_name: generic_vqa ...)`

The new datasets should use this path instead of adding new dataset classes. Converter outputs must match the generic reader's expectations:

```json
{
  "dataset": "vizwiz",
  "image": "train/VizWiz_train_00000000.jpg",
  "question": "What is this?",
  "question_id": 123,
  "answer": ["bottle", "bottle", "water bottle"]
}
```

For validation records, the same fields are retained. Evaluators will use the answer fields from converted validation records or sidecar metadata files.

## Architecture

### Shared Helpers

Create a small shared utility module for conversion/evaluation behavior that is common across VizWiz and DAQUAR:

`dataset_adapters/generic_vqa.py`

Responsibilities:

- Normalize answers with the same conservative text cleanup used by VQA-style evaluation: lowercase, trim, collapse whitespace, strip simple punctuation.
- Build `answer_list.json` from train plus validation answers, with a deterministic sort.
- Write JSON atomically enough for experiment scripts: create parent directory, write UTF-8 JSON.
- Compute exact-match accuracy for single-answer datasets.
- Compute VQA-style soft accuracy from multiple reference answers where available.

This helper must stay lightweight. It should not import model code or dataset-specific download libraries.

### VizWiz Converter

Add `convert_vizwiz.py`.

Inputs:

- `--vizwiz-root`: root directory containing downloaded VizWiz images and annotations.
- `--train-annotations`: optional path, default `<root>/annotations/train.json`.
- `--val-annotations`: optional path, default `<root>/annotations/val.json`.
- `--train-image-dir`: optional path, default `<root>/images/train`.
- `--val-image-dir`: optional path, default `<root>/images/val`.
- `--output-root`: optional path, default `<root>`.
- `--include-unanswerable`: default true.

Expected raw VizWiz annotation fields:

- `image`
- `question`
- `answers`
- `answer_type`
- `answerable`

Outputs:

- `<output-root>/train.json`
- `<output-root>/val.json`
- `<output-root>/answer_list.json`
- `<output-root>/vizwiz_val_metadata.json`

Conversion rules:

- Each record gets a stable integer `question_id`. If the raw annotation has no ID, use split-local index with non-overlapping offsets.
- `image` is written as `train/<filename>` or `val/<filename>` so `vqa_root=<root>/images` works.
- `answer` is a list of normalized answer strings from the raw 10 crowd answers.
- Metadata stores `answer_type` and `answerable` keyed by `question_id`.
- If `--include-unanswerable=false`, drop records whose majority answer is `unanswerable`.

### DAQUAR Converter

Add `convert_daquar.py`.

Inputs:

- `--daquar-root`: root directory containing DAQUAR QA files and images.
- `--train-qa`: optional path to full DAQUAR train QA file.
- `--test-qa`: optional path to full DAQUAR test QA file.
- `--image-root`: optional path, default `<root>/images`.
- `--output-root`: optional path, default `<root>`.

Outputs:

- `<output-root>/train.json`
- `<output-root>/val.json`
- `<output-root>/answer_list.json`
- `<output-root>/daquar_val_metadata.json`

Conversion rules:

- Use DAQUAR's official train split as `train.json`.
- Use DAQUAR's official test split as `val.json`, because `generic_vqa` uses `val_file` for evaluation.
- Preserve original image filenames when available. If the raw QA file only provides image IDs, map them through the official image list.
- Store each DAQUAR answer as a one-item list for training and as reference metadata for exact-match evaluation.

DAQUAR raw files have appeared in more than one public packaging style. The converter should support two concrete forms:

- JSON list records with `image`, `question`, and `answer`.
- Tab/CSV-like rows with image ID, question, answer columns.

If neither form is detected, fail with a clear message describing the expected columns.

### Configs

Add:

- `configs/vizwiz.yaml`
- `configs/daquar.yaml`

Both configs use:

```yaml
dataset_name: generic_vqa
train_files: ['train']
val_file: val
answer_list: answer_list
```

Committed config paths should use relative workspace defaults:

- VizWiz: `ann_root: datasets/vizwiz`, `vqa_root: datasets/vizwiz/images`
- DAQUAR: `ann_root: datasets/daquar`, `vqa_root: datasets/daquar/images`

Experiment scripts may still override these paths with absolute paths when users set custom dataset locations.

### Evaluation

Add `vizwiz_eval.py`.

Metrics:

- `overall`: VQA-style soft accuracy using up to 10 reference answers.
- `answerable`: accuracy on records where metadata `answerable == 1`.
- `unanswerable`: accuracy on records where metadata `answerable == 0`.
- `by_answer_type`: per raw `answer_type` when present.

Add `daquar_eval.py`.

Metrics:

- `overall`: exact-match accuracy after answer normalization.
- `by_question_prefix`: coarse buckets based on normalized question prefix: `what`, `where`, `how many`, `is/are`, `other`.

Both scripts accept:

- positional `result_file`, matching `train_vqa.py` output `result/vqa_result.json`
- `--annotation-file`, defaulting to converted `val.json`
- optional dataset metadata path where relevant

Each script writes a JSON metrics file beside the result file:

- `vizwiz_eval.json`
- `daquar_eval.json`

### Experiment Scripts

Add:

- `examples/run_vizwiz_experiment.sh`
- `examples/run_daquar_experiment.sh`

Both scripts follow the PathVQA script shape:

1. Validate raw dataset files exist.
2. Convert annotations unless `SKIP_CONVERT=1`.
3. Train a real-only baseline unless `RUN_BASELINE=0`.
4. Generate pseudo-QA from training images by default, unless `UNLABELED_ANNOTATIONS` points to a separate unlabeled pool.
5. Filter pseudo-QA with configurable gates.
6. Train student on `train + synthetic_data`.
7. Evaluate final student.

Common environment variables:

- `DATASETS_DIR`
- `NUM_GPUS`
- `VQA_EPOCHS`
- `FILTER_KEEP_TOP`
- `TRUNCATE_GENERATE`
- `ENABLE_CONF`
- `ENABLE_ITM`
- `ENABLE_XCONS`
- `RUN_BASELINE`
- `SKIP_CONVERT`
- `SKIP_GENERATE`
- `SKIP_FILTER`
- `SKIP_TRAIN`
- `SKIP_EVAL`
- `UNLABELED_ANNOTATIONS`

VizWiz-specific variables:

- `VIZWIZ_DIR`
- `VIZWIZ_IMAGES`
- `VIZWIZ_TRAIN_ANNOTATIONS`
- `VIZWIZ_VAL_ANNOTATIONS`
- `INCLUDE_UNANSWERABLE`

DAQUAR-specific variables:

- `DAQUAR_DIR`
- `DAQUAR_IMAGES`
- `DAQUAR_TRAIN_QA`
- `DAQUAR_TEST_QA`

The scripts should not automatically download full datasets. VizWiz and DAQUAR official downloads may require manual setup or unstable legacy links. Scripts should print clear missing-file messages and expected directory layouts.

## Data Flow

### VizWiz

```text
official train/val JSON + images
  -> convert_vizwiz.py
  -> datasets/vizwiz/{train,val,answer_list,vizwiz_val_metadata}.json
  -> train_vqa.py real-only baseline
  -> generate_questions.py on train images by default
  -> filter_pseudo.py
  -> train_vqa.py train + filtered synthetic
  -> vizwiz_eval.py
```

### DAQUAR

```text
official train/test QA + NYU image files
  -> convert_daquar.py
  -> datasets/daquar/{train,val,answer_list,daquar_val_metadata}.json
  -> train_vqa.py real-only baseline
  -> generate_questions.py on train images by default
  -> filter_pseudo.py
  -> train_vqa.py train + filtered synthetic
  -> daquar_eval.py
```

Evaluation split images must not be used for pseudo-QA generation by default. A user can opt into transductive experiments by explicitly setting `UNLABELED_ANNOTATIONS` to an evaluation split, but scripts should label that choice in their logs.

## Error Handling

Converters must fail early with explicit messages for:

- Missing annotation files.
- Missing image directories.
- Empty train or validation split after conversion.
- Records with missing `image`, `question`, or `answer`.

Experiment scripts must fail before training if:

- Converted `train.json`, `val.json`, or `answer_list.json` is missing.
- X-consistency is enabled but no baseline checkpoint exists.
- Synthetic raw or filtered data is requested but absent after the relevant step.

Evaluators must fail with a clear message if:

- A prediction has an unknown `question_id`.
- Required metadata is missing for a metric.
- Result JSON is not a list of `{question_id, answer}` records.

## Testing

Add unit tests with tiny local fixtures:

- `tests/test_convert_vizwiz.py`
- `tests/test_convert_daquar.py`
- `tests/test_vizwiz_eval.py`
- `tests/test_daquar_eval.py`

Test cases:

- VizWiz conversion preserves 10 answers, answerability metadata, image relative paths, and answer list.
- VizWiz evaluator returns expected soft accuracy for exact match, partial agreement, and unanswerable records.
- DAQUAR conversion handles one JSON-style fixture and one table-style fixture.
- DAQUAR evaluator returns expected exact-match accuracy after normalization.
- Config smoke tests compose `configs/vizwiz.yaml` and `configs/daquar.yaml` with local overrides.

No test should download datasets or instantiate BLIP.

## Success Criteria

Implementation is complete when:

- `python convert_vizwiz.py --help` and `python convert_daquar.py --help` work.
- `python vizwiz_eval.py --help` and `python daquar_eval.py --help` work.
- Tiny converter/evaluator tests pass.
- Hydra can compose both new configs with local path overrides.
- `bash -n examples/run_vizwiz_experiment.sh` and `bash -n examples/run_daquar_experiment.sh` pass.
- Running scripts with `SKIP_GENERATE=1 SKIP_FILTER=1 SKIP_TRAIN=1` validates converted data and exits cleanly when expected files are present.

## Rationale

VizWiz and DAQUAR are a better first expansion than TextVQA for this thesis phase:

- They are domain-specific but remain compatible with the current open-ended VQA training path.
- Their sizes are practical for low-resource augmentation experiments.
- They do not require adding OCR features, copy mechanisms, or OCR-token metrics before the augmentation story is validated.

TextVQA should be handled in a later spec focused on OCR-aware augmentation.

## References

- VizWiz-VQA: https://vizwiz.org/tasks-and-datasets/vqa/
- DAQUAR / Visual Turing Challenge: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/visual-turing-challenge
