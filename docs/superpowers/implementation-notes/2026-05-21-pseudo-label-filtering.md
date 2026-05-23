# Implementation Notes — Pseudo-label Filtering

> Running log of decisions, deviations, and tradeoffs made during implementation.  
> Spec: `docs/superpowers/specs/2026-05-21-pseudo-label-filtering-design.md`  
> Plan: `docs/superpowers/plans/2026-05-21-pseudo-label-filtering.md`

---

## Session start

- **Date**: 2026-05-21
- **Mode**: Subagent-driven development
- **Branch**: `feat/pseudo-label-filter` (checkout trong repo hiện tại, không worktree)
- **Repo root**: `/home/phongcoder/Workspace/thesis/SelTDA`

---

## Decisions & deviations

### 2026-05-21 — Bỏ git worktree, dùng branch checkout

- **Plan gốc (Task 0)**: `git worktree add ../SelTDA-filter -b feat/pseudo-label-filter`
- **Quyết định**: User yêu cầu checkout branch trong repo hiện tại thay vì tạo folder sibling.
- **Plan đã sửa**: Task 0 + header **Workspace** trong plan.
- **Cleanup**: Xóa worktree cũ `SelTDA-filter/` (`git worktree remove --force`) vì branch bị lock ở worktree đó.

### 2026-05-21 — Task 0: branch + setup.sh

- **Branch**: `feat/pseudo-label-filter` tại `/home/phongcoder/Workspace/thesis/SelTDA`
- **Worktree cũ**: đã `git worktree remove SelTDA-filter --force`
- **Deps** (`requirements.txt`): `open_clip_torch==2.20.0`, `sentence-transformers==2.2.2`, `huggingface_hub==0.13.4` (pin vì ST 2.2.2 cần `cached_download`), `pytest`
- **`setup.sh` sửa**: thêm `export PYTHONNOUSERSITE=1` trước mọi `pip install` — tránh leak `~/.local` vào conda env
- **Torch**: cài vào env `vqa` (2.12.0+cu126) sau khi phát hiện lần chạy `setup.sh` đầu không cài torch vào env (chỉ thấy ở user site-packages)

### 2026-05-21 — Tasks 1–15 implement (Phase 1 code)

**Files mới:**
- `filtering/` — `matchers.py`, `scorers.py`, `gates.py`, `io.py`, `report.py`, `adapters.py`
- `filter_pseudo.py`, `configs/filter_pseudo.yaml`
- `examples/filter_synthetic.sh`, `scripts/check_invariants.sh`
- `tests/test_*.py` + `tests/fixtures/synthetic_data_raw_tiny.json`

**Files sửa (cho phép spec §1.4):**
- `generate_questions.py` — thêm `gen_logprob`, `scores`, `_compute_mean_logprob()`
- `examples/generate_synthetic_data.sh` — output đổi thành `synthetic_data_raw.json`
- `requirements.txt`, `setup.sh`

**Không sửa:** `train_vqa.py`, `data/`, `models/`, eval scripts — `check_invariants.sh` pass.

**Test fixes so với plan (deviation nhỏ):**
- Dummy SBERT stub: dùng vector đối nhau `[1,0]` vs `[-1,0]` cho dog/cat → cosine rescaled = 0.0 (plan ghi `[0,1]` → ra 0.5)
- `thresholds_from_quantile(keep_top=0.75)` trên 10 phần tử: ngưỡng thực tế ≈ 0.325, không 0.25 (plan tính sai quantile)

**Tradeoff Gate 1 trong filter:** `score_confidence` đọc `gen_logprob` thô, rồi `normalize_min_max` trên toàn pool trước khi quantile — không dùng raw logprob trực tiếp làm threshold. Khớp spec §3.1.

### 2026-05-21 — Experiment scripts

- **`dataset.sh`**: thêm download `unlabeled2017.zip` (http://images.cocodataset.org/zips/unlabeled2017.zip) → `datasets/coco2017/unlabeled2017/`; thêm download teacher `checkpoint_04.pth` qua `gdown` (Drive ID `19Y9oQNlYBTkoT4sYuUQWrEV9iUatPkdI`).
- **`examples/run_experiment.sh`**: pipeline end-to-end (dataset → convert → generate → filter → train). Gate 3 dùng BLIP pretrained URL (không file local) vì `cache/blip_pretrained.pth` chưa có sẵn trên máy mới.

*(Entries appended as work proceeds.)*

### 2026-05-23 — Cache image_embeds trong BLIP_Decoder.generate (speedup Gate 1)

- **Vấn đề**: `_compute_mean_logprob` trong `generate_questions.py` gọi `visual_encoder` lần 2 mỗi batch → generate chậm ~30–50%.
- **Quyết định**: Sửa `models/blip.py` — thêm `return_logprob=False` vào `BLIP_Decoder.generate()`. Khi `True`, cache `image_embeds` từ lần encode duy nhất và teacher-force inline trong cùng call.
- **Lý do chọn Hướng B (cache + teacher-force) thay vì `output_scores`**: Báo cáo định hướng yêu cầu raw "log-likelihood của decoder" (`log p_θ(T|I)`). `output_scores` với `top_p=0.9` trả logits sau LogitsProcessor → bias renormalization trên nucleus.
- **Spec cập nhật**: §1.4 cho phép sửa `models/blip.py` (scope hẹp); §3.1 viết lại "Cách lấy"; §4.2, §7.1 cập nhật tương ứng.
- **Files sửa**: `models/blip.py`, `generate_questions.py` (xóa `_compute_mean_logprob`), `scripts/check_invariants.sh` (whitelist `models/blip.py`), `tests/test_blip_generate_logprob.py` (mới).
- **Backward-compat**: `return_logprob=False` default → `train_vqg.py` không đổi behavior.
