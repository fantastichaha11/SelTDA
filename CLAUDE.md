# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo layout

- Git root and all commands: `cd SelTDA` (parent `thesis/` is not the repo).

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
pytest tests/test_gate_registry.py tests/test_strata.py tests/test_coreset.py  # C1
pytest tests/test_grounding_cascade.py tests/test_scorers_language_prior.py   # C3
pytest tests/test_iterative_smoke.py tests/test_skill_gap.py                    # C2
pytest tests/test_vqascore_reward.py tests/test_reward.py                     # C-RL / C-RT+
bash scripts/check_invariants.sh   # must pass before PR (vs origin/master)
```

### 6. RL Teacher — C-RL / C-RT+ (`train_vqg_rl.py`)

Online GRPO on the VQG teacher with reward **P2** (ITM + grounding + learnability − KL − repetition). **C-RT+** adds **VQAScore** (frozen [CLIP-FlanT5-xl](https://huggingface.co/zhiqiulin/clip-flant5-xl) via `t2v_metrics`, not BLIP-VQA). Spec: `docs/superpowers/specs/2026-06-04-reasoning-teacher-rl-cot-design.md`, baseline C-RL: `docs/superpowers/specs/2026-06-01-rl-teacher-grpo-design.md`.

**Optional dependency** (separate from main BLIP conda env):

```bash
pip install -r requirements-vqascore.txt   # t2v-metrics; first run downloads HF weights
```

**Run GRPO** (default config on `feat/iterative-seltda`):

```bash
python train_vqg_rl.py --config configs/rl_teacher_aokvqa.yaml \
  --overrides wandb=false torch_home=$(pwd)/cache/torch_home
```

**Ablations / overrides:**

```bash
# C-RL without VQAScore
python train_vqg_rl.py --config configs/rl_teacher_aokvqa.yaml \
  --overrides reward.w_vqa=0 wandb=false

# OOM: move VQAScore off GPU
python train_vqg_rl.py --config configs/rl_teacher_aokvqa.yaml \
  --overrides reward.vqascore_device=cpu wandb=false
```

Config highlights (`configs/rl_teacher_aokvqa.yaml`): `reward.w_vqa`, `reward.vqascore_model` (`clip-flant5-xl`), `reward.vqascore_device` (`cuda`), `reward.w_type: 0` (weak-type off for now). Epoch audit uses **BLIP-ITM-large + BLIP-VQA published** (`filtering/audit.py`) — independent of the VQAScore reward term.

Code: `filtering/reward.py` (`compose_reward`, `w_vqa`), `filtering/vqascore_adapter.py`, `train_vqg_rl.py` (`build_reward_fn`).
```

## Thesis feature branches (C1 / C2 / C3)

| Branch | Contribution | Main deliverables | Spec / plan |
|--------|--------------|-------------------|-------------|
| `feat/pseudo-label-filter` | **C1** Structured curation | Gate registry (REG-1), TS-1 (`filtering/strata.py`), CS-X (`filtering/coreset.py`), `configs/filter_pseudo_{stratified,coreset}.yaml`, `scripts/random_subsample_control.py`, `scripts/sweep_synth_ratio.sh` | `docs/superpowers/specs/2026-05-27-c1-*.md`, `plans/2026-05-27-c1-*.md` |
| `feat/grounding-gates` | **C3** Grounding gates | LP-1 (`filtering/scorers_language_prior.py`), KC-1 + Wikipedia (`filtering/retrieval/`, `scorers_knowledge.py`), `configs/filter_pseudo_grounding.yaml`, `GATE_ORDER` += `lp`, `kcons` | `docs/superpowers/specs/2026-05-27-c3-*.md`, `plans/2026-05-27-c3-*.md` |
| `feat/iterative-seltda` | **C2** Closed-loop + **C-RL / C-RT+** | `orchestration/`, `train_vqg_rl.py`, `filtering/reward.py`, `filtering/vqascore_adapter.py`, `configs/rl_teacher_aokvqa.yaml`, TC-1, `scripts/run_iterative_round.sh` | `docs/superpowers/specs/2026-05-27-c2-*.md`, `2026-06-01-rl-teacher-grpo-design.md`, `2026-06-04-reasoning-teacher-rl-cot-design.md` |

**Fork:** C2 and C3 branch from C1 after gate registry. **Recommended merge order:** `feat/pseudo-label-filter` → `feat/grounding-gates` → `feat/iterative-seltda`.

### C1 — `feat/pseudo-label-filter`

1. `git checkout feat/pseudo-label-filter`
2. Filter with type-stratified thresholds (TS-1): `python filter_pseudo.py --config configs/filter_pseudo_stratified.yaml`
3. Filter TS-1 + coreset (CS-X): `python filter_pseudo.py --config configs/filter_pseudo_coreset.yaml`
4. Random subsample control (IDEA-04): `python scripts/random_subsample_control.py --reference datasets/aokvqa/synthetic_data.json --pool datasets/aokvqa/synthetic_data_raw.json --output datasets/aokvqa/synthetic_random.json --seed 42`
5. SAT-1 sweep manifest: `bash scripts/sweep_synth_ratio.sh` → `research/experiments/saturation/accuracy_vs_ratio.csv` (train/eval via `examples/run_experiment.sh`)
6. Tests: `pytest -m "not slow" tests/test_gate_registry.py tests/test_strata.py tests/test_coreset.py tests/test_filter_pseudo_smoke.py -q`
7. `bash scripts/check_invariants.sh`

### C3 — `feat/grounding-gates`

1. `git checkout feat/grounding-gates`
2. Filter C+I+X + LP-1 + KC-1: `python filter_pseudo.py --config configs/filter_pseudo_grounding.yaml` (KC mainly on `external_knowledge` stratum; needs student, Wikipedia API, NLI model, cache `cache/retrieval/`)
3. Score-only debug: `--overrides scoring_only=true`
4. Tests: `pytest -m "not slow" tests/test_scorers_language_prior.py tests/test_wikipedia_retrieval.py tests/test_scorers_knowledge.py tests/test_grounding_cascade.py -q`
5. First push/pull: `git push -u origin feat/grounding-gates` then `git pull origin feat/grounding-gates`

### C2 — `feat/iterative-seltda`

1. `git checkout feat/iterative-seltda`
2. One IT-1 round (generate → filter → train → eval → skill gap): configure and run `orchestration/iterative_seltda.py` with `orchestration/iterative_aokvqa.yaml`
3. TC-1: set `question_type_schedule` and `weak_types_file` in `configs/generate_questions_aokvqa.yaml`
4. J-1 staged curriculum: `bash scripts/run_iterative_round.sh` → `synthetic_easy.json`, `synthetic_hard.json`, `synthetic_staged.json` (merge 1:3 easy:hard)
5. Tests: `pytest -m "not slow" tests/test_skill_gap.py tests/test_merge_pools.py tests/test_type_schedule.py tests/test_iterative_smoke.py -q`
6. `train_vqg_config` in YAML is optional — orchestrator skips if missing
7. **C-RT+ GRPO teacher**: `pip install -r requirements-vqascore.txt` then `python train_vqg_rl.py --config configs/rl_teacher_aokvqa.yaml --overrides wandb=false` (see §6 above)
8. Tests: `pytest -m "not slow" tests/test_vqascore_reward.py tests/test_reward.py -q`

### Verify on any feature branch

```bash
cd SelTDA
pip install omegaconf hydra-core -q
pytest -m "not slow" -q
bash scripts/check_invariants.sh
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
Cascade via `filtering/gate_registry.py` (`GATE_ORDER`, `enabled_gate_names`, `apply_cascade`). Base scorers in `GATE_SCORERS`; **lp** and **kcons** are scored in `filter_pseudo.py` (need student / NLI / Wikipedia).

1. **conf** — teacher log-probability (quantile; TS-1 per stratum in `strata.py`)
2. **itm** — CLIP image–text match
3. **xcons** — student vs pseudo-answer (`matchers.py`; `sbert_model=None` skips SBERT)
4. **lp** (C3) — answer stability under image corruption
5. **kcons** (C3) — NLI over Wikipedia passages (`filtering/retrieval/`)

`filtering/gates.py`: `apply_gates` delegates to `apply_cascade` (keeps `GateThresholds` API). CS-X: `coreset.py`. `adapters.py` wraps BLIP/OpenCLIP; `report.py` builds filter diagnostics.

**C-RL reward** (`filtering/reward.py`, `train_vqg_rl.py`): P2 terms + optional **VQAScore** (`filtering/vqascore_adapter.py` → `t2v_metrics`, CLIP-FlanT5). **Audit** (`filtering/audit.py`): held-out ITM-large + BLIP-VQA match for hacking detection (not the VQAScore reward model).

### Checkpoints
Saved as `checkpoint_XX.pth` dicts containing `model`, `optimizer`, `config`, `epoch`. `checkpoint_utils.py` handles `--resume auto` (picks latest by epoch number) and explicit path resume. Only `train_vqa.py` currently supports resume; `train_vqg.py` does not.

## Gotchas

- **filter_pseudo ordering**: populate `r["scores"]` before thresholds/cascade; skip `apply_cascade` if no gates enabled; compute stratify `global_fallback` only after `conf` scores exist.
- **Gate registry**: do not replace `apply_gates` with a re-export of `apply_cascade` — breaks `GateThresholds` callers/tests.
- **Test deps**: `filter_pseudo` / `generate_questions` tests import Hydra/OmegaConf — `pip install omegaconf hydra-core` in minimal envs.
- **Feature branch git**: new branches lack upstream — `git pull origin <branch>` or `git push -u origin <branch>` before bare `git pull` works.
- **A-OKVQA convert step**: after `bash dataset.sh`, run `python convert_aokvqa.py --config configs/aokvqa.yaml` before any A-OKVQA training/eval.
- **`wandb: true` default**: `configs/aokvqa.yaml` (and others) have W&B enabled by default. Local runs will crash unless you add `--overrides wandb=false`.
- **Dead `torch_home`**: configs point `torch_home` to a network path that won't exist locally. Override with `torch_home=$(pwd)/cache/torch_home` or set `torch_home=null`.
- **A-OKVQA eval set**: the public test set requires server-side scoring. For local evaluation, pass `--overrides use_validation_set_as_test_set=true`.
- **`train_vqg.py` does not support `--resume`**; only `train_vqa.py` does.
- **Filter debug mode**: add `--overrides scoring_only=true` to `filter_pseudo.py` to compute and attach scores to all records without discarding any.
- **C-RT+ VQAScore**: needs `pip install -r requirements-vqascore.txt`. Default `reward.vqascore_device=cuda` (~12–16 GB extra with `clip-flant5-xl`); use `reward.vqascore_device=cpu` or `reward.w_vqa=0` if OOM. Do not use BLIP student checkpoint as VQAScore — wrong training objective vs Lin et al. VQAScore.
- **`train_vqg_rl.py`**: GRPO teacher only; round orchestration may call `scripts/run_iterative_round.sh` / `orchestration/`. Weak-type reward (`w_type`) currently off in `rl_teacher_aokvqa.yaml`.
