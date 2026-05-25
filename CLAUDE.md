# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SelTDA is a self-training framework for data-scarce VQA (Visual Question Answering) tasks, introduced in CVPR 2023. It is built on top of [salesforce/BLIP](https://github.com/salesforce/BLIP). The pipeline has four stages:

1. **Train teacher** – a VQG (Visual Question Generation) model on labeled data
2. **Generate synthetic data** – the teacher generates (question, answer) pairs from unlabeled images
3. **Filter pseudo-labels** – three quality gates (confidence, image-text matching, cross-consistency) discard noisy samples
4. **Self-train student** – a VQA model fine-tuned on real + filtered synthetic data

## Environment Setup

```bash
conda env create -f environment.yaml   # see setup.md for a minimal pip-only alternative
conda activate blip
export PYTHONNOUSERSITE=1
```

## Data & Checkpoint Setup

```bash
bash dataset.sh
```

This downloads COCO 2017 images (train/val/test/unlabeled), A-OKVQA annotations, the published teacher checkpoint (`cache/teacher_weights/checkpoint_04.pth`), student checkpoint (`cache/student_weights/checkpoint_09.pth`), and the published pre-generated synthetic JSONs. All downloads are idempotent (skipped if already present).

## Common Commands

All entry-point scripts share the CLI pattern from `cli.py`:
```
python <script>.py --config <config.yaml> [--overrides key=value ...]
```
Config values can be overridden on the command line via `--overrides`. Training scripts additionally accept `--output_dir`, `--evaluate`, `--resume`, and distributed training flags.

### 1. Train Teacher (VQG)
```bash
python -m torch.distributed.run --master_port=37770 --nproc_per_node=1 train_vqg.py \
    --config=configs/aokvqg.yaml \
    --output_dir=cache/teacher_weights \
    --overrides batch_size=16
```

### 2. Generate Synthetic Data
```bash
python generate_questions.py \
    --config configs/generate_questions_aokvqa.yaml \
    --overrides \
        output_folder=datasets/aokvqa \
        pretrained=cache/teacher_weights/checkpoint_04.pth \
        questions_per_image=2 \
        output_annotations_name=synthetic_data_raw.json
```
Generation is resumable: if `synthetic_data_raw.json` already exists, images already present in it are skipped.

### 3. Filter Synthetic Data
```bash
python filter_pseudo.py \
    --config configs/filter_pseudo.yaml \
    --overrides \
        input=datasets/aokvqa/synthetic_data_raw.json \
        output=datasets/aokvqa/synthetic_data.json \
        report=datasets/aokvqa/filter_report.json
```

### 4. Train Student (VQA)
```bash
python -m torch.distributed.run --nproc_per_node=1 train_vqa.py \
    --output_dir=cache/self_trained_weights \
    --config configs/aokvqa.yaml \
    --overrides \
        "train_files=[train,synthetic_data]" \
        truncate_train_dataset_to=34000
```

Resume an interrupted run (pass a path or `auto` to pick the latest checkpoint in `output_dir`):
```bash
python -m torch.distributed.run --nproc_per_node=1 train_vqa.py \
    --output_dir=cache/self_trained_weights \
    --resume auto \
    --config configs/aokvqa.yaml \
    --overrides "train_files=[train,synthetic_data]" truncate_train_dataset_to=34000
```

### 5. Evaluate
```bash
python -m torch.distributed.run --nproc_per_node=1 train_vqa.py \
    --output_dir=cache/evals \
    --evaluate \
    --config configs/artvqa.yaml \
    --overrides pretrained=cache/self_trained_weights/checkpoint_09.pth
python artvqa_eval.py cache/evals/result/vqa_result.json
```
Dataset-specific eval scripts: `vqav2_eval.py`, `aokvqa_mc_eval.ipynb`, `okvqa_eval.py`, `artvqa_eval.py`, `advqa_eval.py`, `pathvqa_eval.py`, `rsvqa_lr_eval.py`, `vqa_ce_eval.py`, `vqa_rephrasings_eval.py`.

### Tests
```bash
pytest                            # all tests
pytest -m "not slow"             # skip slow tests
pytest tests/test_gates.py       # single file
pytest tests/test_scorers_clip.py -k "test_score_confidence"  # single test
```

## Architecture

### Config System
All configuration uses [Hydra/OmegaConf](https://hydra.cc). Each script has a default config path (e.g. `configs/vqa.yaml`) and loads it via `cli.parse_args()`. Override any field with `--overrides key=value`. Config is serialized to `output_dir/config.yaml` on each run.

### Models (`models/`)
All models are BLIP variants:
- `blip_vqa.py` — VQA model used for training the student and evaluation
- `blip.py` (`blip_decoder`) — captioning/generation model used as the teacher in `generate_questions.py`
- `blip_itm.py` — image-text matching
- `blip_retrieval.py`, `blip_nlvr.py`, `blip_pretrain.py` — other BLIP tasks
- `med.py` — multimodal encoder-decoder (the BERT-based text model)
- `vit.py` — Vision Transformer backbone

### Data (`data/`)
- `vqa_dataset.py` — `GenericVqaDataset` reads JSON annotation files from `ann_root`. `train_files` config key is a list of base names (without `.json`); files are concatenated at load time. The `image` field in each record is a `parent/filename` relative path resolved against `vqa_root`.
- `vqg_dataset.py` — dataset for the teacher VQG training
- `data/__init__.py` — `create_dataset` factory dispatches on `config.dataset_name`

### Data Format
Every VQA JSON file is a list of dictionaries. Key fields (see `schemas.py` for Pydantic models):
- `image` — `"folder/file.jpg"` path relative to the image root
- `question`, `question_id`, `answer` (list of strings), `dataset`
- Synthetic records also carry `gen_logprob` and `scores` (populated by `filter_pseudo.py`)

Use `schemas.TrainingRecord` / `schemas.TestingRecord` to validate new datasets.

### Filtering (`filtering/`)
Three independent quality gates applied in cascade order:
1. **conf** (`scorers.score_confidence`) — teacher log-probability; keeps top-k% by quantile
2. **itm** (`scorers.score_clip_itm`) — CLIP cosine similarity between image and "Q? A." text
3. **xcons** (`scorers.score_xcons`) — cross-consistency: student zero-shot answer vs. pseudo-answer, scored with sentence-BERT (`filtering/matchers.py`)

`filtering/gates.py` contains `apply_gates` and quantile threshold helpers. `filtering/adapters.py` wraps BLIP and OpenCLIP behind a common interface. `filtering/report.py` builds a JSON diagnostic report.

### Checkpoints
Saved as `checkpoint_XX.pth` dicts containing `model`, `optimizer`, `config`, `epoch`. `checkpoint_utils.py` handles `--resume auto` (picks latest by epoch number) and explicit path resume. Only `train_vqa.py` currently supports resume; `train_vqg.py` does not.

## Gotchas

- **A-OKVQA convert step**: after `bash dataset.sh`, run `python convert_aokvqa.py --config configs/aokvqa.yaml` before any A-OKVQA training/eval.
- **`wandb: true` default**: `configs/aokvqa.yaml` (and others) have W&B enabled by default. Local runs will crash unless you add `--overrides wandb=false`.
- **Dead `torch_home`**: configs point `torch_home` to a network path that won't exist locally. Override with `torch_home=$(pwd)/cache/torch_home` or set `torch_home=null`.
- **A-OKVQA eval set**: the public test set requires server-side scoring. For local evaluation, pass `--overrides use_validation_set_as_test_set=true`.
- **`train_vqg.py` does not support `--resume`**; only `train_vqa.py` does.
- **Filter debug mode**: add `--overrides scoring_only=true` to `filter_pseudo.py` to compute and attach scores to all records without discarding any.
