# Thiết kế C-RL — Reward-Internalized Self-Training (Online-GRPO Teacher)

- **Ngày**: 2026-06-01
- **Contribution**: C-RL — Online-RL teacher với filter nội hoá vào reward (thesis Ch. 4 / RL extension của C2)
- **Branch**: `feat/iterative-seltda` (cùng nhánh C2; fork từ `feat/pseudo-label-filter`)
- **Trạng thái**: Drafted — chờ user review
- **Dependency**: C1 scorers (`filtering/adapters.py`, `matchers.py`, `strata.py`); C3 `filtering/scorers_language_prior.py` (LP-1); C2 restart policy & skill-gap khái niệm

---

## 0. Tóm tắt

SelTDA gốc: teacher cố định sinh pseudo-QA → **filter post-hoc** loại ~30% rác → train student (một vòng). C-RL **nội hoá filter vào reward** và biến teacher thành **policy GRPO online**: teacher học sinh QA (a) đúng weak question type, (b) grounded & đúng, (c) khó với student — **không cần filter pass sau sinh**. Đa vòng, teacher tích luỹ RL, student restart mỗi vòng.

**Janusian lõi**: pseudo-QA giá trị phải đồng thời **ĐÚNG** (model độc lập trả lời được, grounded) **× KHÓ** (student gốc còn bất định). Chỉ đúng → quá dễ, vô ích; chỉ khó → thưởng cả rác. Tích hai vế là sweet spot và **khó hack**.

**§1.4**: chỉ thêm script mới (`train_vqg_rl.py`, `orchestration/`, `filtering/reward.py`, `filtering/audit.py`). **Không** sửa `train_vqa.py` / `train_vqg.py` / loss / eval core.

---

## 1. Yêu cầu & ràng buộc

### 1.1 In scope

| ID | Thành phần | Mô tả |
|----|------------|--------|
| RL-1 | Online-GRPO teacher | Teacher (BLIP decoder) update mỗi step bằng GRPO; tích luỹ qua round |
| RW-P2 | Reward Grounded-Learnability | `type + itm + grounding + learnability − KL − repetition` |
| WT-1 | Weak-type freezing | Eval student_base 1 lần → cố định `WEAK_TYPES` |
| AUD-1 | Held-out hacking audit | BLIP-ITM-Large + BLIP-VQA độc lập; chỉ đo, không lọc |
| HRP-1 | Hacking response protocol | Tự động L1–L3; L4–L5 dừng cho người |

### 1.2 Out of scope

- Filter post-hoc trên pool sinh ra (đã nội hoá vào reward) — trừ khi HRP-1 L4 kích hoạt
- Học weak-type động mỗi round (type **cố định** từ student_base)
- Δaccuracy-as-reward (P3) — không chọn; reward không train student mỗi step
- Sửa loss/eval student; PathVQA / multi-dataset
- DPO preference-pair (thay bằng GRPO online)

### 1.3 Success criteria

| Metric | Target | Falsifier |
|--------|--------|-----------|
| A-OKVQA val overall (vs vanilla SelTDA 1-round) | **≥ +1.0** tuyệt đối | Round cuối < +0.5 → báo saturation |
| Mỗi weak-type accuracy | **≥ +3** vs vanilla | Regression type khác > 0.5 overall |
| (Phụ) Hacking curve | reward vs judge tách rời được đo & xử lý | Judge sụp không phát hiện được |
| (Phụ) Saturation | đường cong acc vs round | — |

### 1.4 Ràng buộc đầu tư

- Dataset: **A-OKVQA** (51k unlabeled COCO), **R = 2–3** round.
- Không API ngoài (tái lập được); mọi judge là model local.

---

## 2. Kiến trúc

### 2.1 Vòng lặp tổng thể

```
Bước 0 (1 lần):
  eval student_base (train trainset-only) trên val
  → WEAK_TYPES = top-k question type theo error rate (k=2–3)

Mỗi round r = 1..R:
  (1) Teacher-GRPO (RL-1): nhiều EPOCH, mỗi epoch gồm nhiều step:
        mỗi step:
          - với mỗi ảnh I trong batch: sample G completions (Q,A) từ teacher_r
          - tính reward P2 cho từng completion
          - advantage = (r − mean_G) / (std_G + eps)
          - update teacher_r theo GRPO + KL(teacher_r ‖ teacher_ref)
        SAU MỖI EPOCH (nhịp RẺ — AUD-1 epoch-level):
          - cheap audit pool mẫu bằng ITM-Large + VQA judge → judge_score
          - HRP-1 L1–L3: nếu hacking → adaptive KL / down-weight / early-stop GRPO
            (early-stop = dừng vòng GRPO của round này, rollback teacher về epoch judge-cao nhất)
        ← teacher TÍCH LUỸ qua round (không restart)
  (2) Generate pool_r từ teacher_r  (KHÔNG filter post-hoc)
  (3) Train student_r MỚI từ cache/student_weights pretrained BLIP (restart)
        - train_files = [train, pool_r]  (trainset GỐC ∪ data teacher-RL sinh; subprocess train_vqa.py, không sửa)
  (4) Eval student_r trên val (nhịp ĐẮT — round-level)
        → acc_r là TÍN HIỆU CONTINUE/STOP CHÍNH
        → ghi orchestration/state/round_{r}.json
  (5) Continue/stop: nếu acc_r − acc_{r-1} ≥ 0.5 → round tiếp; else STOP (saturation)
      (acc_0 = vanilla SelTDA 1-round baseline)
```

**Hai nhịp audit (two-cadence)**:

| Nhịp | Khi nào | Đo gì | Dùng để |
|------|---------|-------|---------|
| **Epoch-level (rẻ)** | sau mỗi epoch GRPO | ITM-Large + VQA judge trên mẫu pool | Phát hiện hacking + HRP L1–L3 early-stop GRPO |
| **Round-level (đắt)** | sau train student_r | student_r val accuracy | Tín hiệu continue/stop chính của vòng round |

**Init policy (bất đối xứng — quan trọng):**

| | Khởi tạo round r | Lý do |
|--|------------------|-------|
| **Teacher** | **Tiếp tục từ teacher_{r-1}** (round 1 init từ pretrained VQG_IC) | Tích luỹ kỹ năng sinh qua round → compound gain |
| **Student** | **Restart từ pretrained BLIP** mỗi round (KHÔNG dùng student_{r-1}) | Tránh confirmation bias (Cascante-Bonilla 2001.06001) |

- **teacher_ref**: teacher gốc (frozen pretrained VQG_IC) — **neo KL cố định về gốc** suốt mọi round, KHÔNG re-anchor về teacher_{r-1}.
- **KL decay qua round**: `kl_beta_r = kl_beta_0 · gamma^(r-1)` (gamma < 1) → ràng buộc **lỏng dần** khi teacher chín, để KL-neo-gốc không kìm việc tích luỹ ở round sau. (Lưu ý: trong-round, HRP-L1 vẫn có thể tăng beta tạm thời nếu phát hiện hacking.)
- Resume mid-experiment: `round_{r}.json` lưu đường dẫn `teacher_{r}` checkpoint để khôi phục trọng số teacher tích luỹ.
- **Vì sao tách nhịp**: train student là đắt (giờ/round) → không thể chạy mỗi epoch; nhưng phát hiện hacking cần nhanh → dùng judge rẻ mỗi epoch. Student val accuracy (không hack được) là trọng tài cuối cùng ở mức round.

### 2.2 Reward P2 (`filtering/reward.py`)

Cho mỗi completion `(I, Q, A)`:

```
R(I,Q,A) =  w_type · TypeMatch
          + w_itm  · ITM
          + w_g    · Grounding
          + w_l    · Learnability
          − w_kl   · KL_term
          − w_d    · Repetition
```

| Term | Định nghĩa | Tái dùng |
|------|------------|----------|
| `TypeMatch` | `1` nếu `classify_question_type(Q,A) ∈ WEAK_TYPES`, else `0` (hoặc soft) | `filtering/strata.py` |
| `ITM` | CLIP image–text match score (term **độc lập, tách riêng**) | `filtering/adapters.py` |
| `Grounding` | `XCONS_frozen(I,Q,A) × LP_flip(I,Q,A)` | `matchers.py` + `scorers_language_prior.py` |
| `Learnability` | `1 − max_prob(student_base(I,Q))` (bất định student gốc) | BLIP-VQA forward |
| `KL_term` | `KL(teacher_r ‖ teacher_ref)` | GRPO trainer |
| `Repetition` | phạt trùng theo embedding/n-gram trong batch | SBERT/n-gram |

- **XCONS_frozen**: student frozen độc lập trả lời `Q`, so khớp `A` (SBERT/EM). **Độc lập với teacher** → diệt circular reward (không dùng `gen_logprob` của teacher).
- **LP_flip**: đáp án **đổi khi che ảnh** → grounded thật (LP-1, C3).
- **ITM tách riêng** để theo dõi per-term acquisition (ITM là term dễ game nhất → cảnh báo hacking sớm).
- Trọng số `w_*` khởi tạo cố định (config); log đóng góp từng term mỗi step.

### 2.3 GRPO trainer (`train_vqg_rl.py`)

- Group size `G` (vd 8) completions/ảnh; advantage chuẩn hoá trong nhóm; **không value model**.
- KL coefficient `β` tới `teacher_ref`; **adaptive** (xem HRP-1 L1).
- Sampling top-p (vd 0.92), temperature config.
- Log/step: mean reward, từng term, KL, entropy, per-term acquisition rate.
- Checkpoint teacher theo judge-score (cho rollback L3).

---

## 3. Weak-type freezing (`orchestration/weak_types.py`)

```python
def compute_weak_types(
    eval_results_path: Path,
    type_classifier: Callable[[str, str], str],   # strata.classify_question_type
    k: int = 3,
) -> list[str]:
    """Parse eval của student_base → error rate per type → top-k."""
```

Chạy **một lần** trước round 1. Kết quả ghi `orchestration/state/weak_types.json`, cố định cho toàn experiment.

---

## 4. Generate & train (không sửa core)

- Generate: tái dùng `generate_questions.py` với teacher_r checkpoint; **bỏ** bước `filter_pseudo.py`.
- Train student: subprocess `train_vqa.py` với `train_files=[train, pool_r]`, `--resume` không dùng (restart từ pretrained), `wandb=false`, `torch_home` override theo gotchas.

---

## 5. Audit (two-cadence) (`filtering/audit.py`, AUD-1)

### 5.1 Epoch-level (rẻ) — phát hiện hacking
Sau **mỗi epoch GRPO**, chấm mẫu ngẫu nhiên `N_audit` (vd 200) completions hiện hành bằng **hai trọng tài độc lập với reward**:

1. **BLIP-ITM-Large** — khớp ảnh–(Q,A); khác CLIP dùng trong term `ITM`.
2. **BLIP-VQA (published, frozen)** — trả lời pseudo-Q rồi so khớp `A`; khác student_base dùng trong reward.

`judge_score = agg(itm_large, vqa_match)`. Deliverable: đường cong **mean_reward vs judge_score** qua epoch.

**Định nghĩa hacking**: `Δmean_reward > 0` nhưng `Δjudge_score < −ε` (ε config), HOẶC per-term acquisition lệch hẳn về một term rẻ (vd ITM > τ·tổng gain). → kích hoạt HRP-1 (§6).

### 5.2 Round-level (đắt) — tín hiệu continue/stop
Sau khi train `student_r`, **eval val accuracy** = trọng tài cuối cùng. `acc_r` quyết định continue/stop (§2.1 bước 5). Đây là tín hiệu **không hack được** bởi teacher (teacher không thể làm student giỏi thật lên bằng cách game proxy). Judge epoch-level chỉ chống hacking *trong* round; student-eval xác nhận giá trị *thật* cuối round.

---

## 6. Hacking response protocol (HRP-1)

Tự động **L1–L3**; **L4–L5** dừng, ghi `state/hacking_alert_r.json` cho người quyết định.

| Mức | Tín hiệu | Hành động (tự động trừ khi ghi chú) |
|-----|----------|--------------------------------------|
| L0 | reward↑, judge↑ | Tiếp tục |
| L1 | phân kỳ nhẹ | **Adaptive KL**: `β ← β·c` (c>1); giảm LR |
| L2 | một term áp đảo gain (vd ITM) | **Down-weight** term đó (`w_itm ← w_itm·d`, d<1); log |
| L3 | judge giảm rõ trong round | **Early-stop GRPO + rollback** teacher về checkpoint judge-cao nhất |
| L4 | hacking dai dẳng ≥2 round | **[MANUAL]** bật light post-hoc filter cho pool_r **HOẶC** promote judge→reward + xoay auditor mới |
| L5 | không cứu được | **[MANUAL]** dừng & báo cáo (negative result đã đặc tả — vẫn là contribution) |

Ba cơ chế nền: early-stop-theo-judge (L3, gần như luôn bật), adaptive-KL (L1), per-term down-weight (L2, tận dụng ITM tách riêng).

---

## 7. Baselines

| Baseline | Mục đích | Bắt buộc? |
|----------|----------|-----------|
| **Vanilla SelTDA 1-round** (teacher gốc, có filter post-hoc) | Baseline chính | ✅ |
| **BoN-only** (teacher gốc sample N, giữ best theo P2, không update) | Test GRPO có thừa không (Simplicity Test) | Recommended-optional |

---

## 8. Cấu trúc thư mục

```
train_vqg_rl.py                       # NEW — GRPO trainer cho BLIP teacher
orchestration/
  rl_teacher_loop.py                  # NEW — vòng round chính
  weak_types.py                       # NEW — Bước 0
  state/
    weak_types.json                   # NEW
    round_{r}.json                    # NEW — resume (acc_r = stop signal)
    round_{r}_epochs.jsonl            # NEW — log epoch-level: reward, từng term, judge_score, KL
    hacking_alert_{r}.json            # NEW — L4/L5 cho người
filtering/
  reward.py                           # NEW — P2 reward
  audit.py                            # NEW — BLIP-ITM-Large + BLIP-VQA judges
configs/
  rl_teacher_aokvqa.yaml              # NEW
tests/
  test_reward.py                      # NEW — từng term + tổng hợp
  test_weak_types.py                  # NEW
  test_audit.py                       # NEW — judge độc lập, hacking detection
  test_rl_smoke.py                    # NEW — mock GRPO 1 step + 1 round mock subprocess
```

---

## 9. Cấu hình (phác thảo `configs/rl_teacher_aokvqa.yaml`)

```yaml
rounds: 3
early_stop_delta: 0.5

weak_types:
  k: 3
  student_base_eval: cache/evals/student_base/vqa_result.json

grpo:
  group_size: 8
  top_p: 0.92
  temperature: 1.0
  kl_beta: 0.1                 # = kl_beta_0 (round 1)
  kl_beta_decay_gamma: 0.7     # kl_beta_r = kl_beta_0 * gamma^(r-1)
  lr: 1.0e-6
  epochs_per_round: 3
  steps_per_epoch: 200

reward:
  w_type: 1.0
  w_itm: 0.5
  w_grounding: 1.0
  w_learnability: 0.5
  w_kl: 0.1
  w_repetition: 0.3

audit:
  every_epoch: true            # nhip re: ITM-Large + VQA judge sau moi epoch GRPO
  n_audit: 200
  itm_large_ckpt: <path>
  vqa_judge_ckpt: <path>
  hacking_eps: 0.02
  itm_dominance_tau: 0.6
  rollback_to_best_judge: true # L3: rollback teacher ve epoch judge-cao nhat

hrp:
  adaptive_kl_c: 1.5
  downweight_d: 0.5
  auto_levels: [1, 2, 3]      # L4-L5 manual
```

---

## 10. Rủi ro

| Rủi ro | Mitigation |
|--------|------------|
| Reward hacking (no filter safety net) | AUD-1 + HRP-1; ITM tách riêng để khoanh vùng; KL + early-stop theo judge |
| Mode/entropy collapse (online policy-gradient) | KL tới teacher_ref + Repetition term + entropy log |
| Circular reward (conf của teacher) | **Loại bỏ** conf-own-logprob; chỉ dùng scorer độc lập |
| GRPO bất ổn / variance | Group-relative advantage; lr thấp; smoke test trước scale |
| Learnability thưởng rác (rác cũng bất định) | **Nhân** với Grounding (phải đúng & grounded mới tính) |
| Compute (3 round × GRPO + train student) | A-OKVQA only; steps_per_round giới hạn; resume `round_{r}.json` |
| §1.4 vi phạm | Mọi thứ ở script mới; subprocess `train_vqa.py`/`generate_questions.py` không sửa |

---

## 11. References

- DataEnvGym (Khan 2024, arXiv 2410.06215) — iterative student-feedback foundation
- Rafailov 2024 (arXiv 2406.02900) — reward hacking trong DPO/RL alignment
- Cascante-Bonilla 2020 (arXiv 2001.06001) — restart/curriculum self-training
- GRPO (DeepSeekMath, arXiv 2402.03300) — group-relative policy optimization
- LP-1 grounding gate: `docs/superpowers/specs/2026-05-27-c3-grounding-gates-design.md`
- C2 closed-loop: `docs/superpowers/specs/2026-05-27-c2-closed-loop-design.md`
- RW-1 deep-dive: `research/ideation/2026-05-26-RW1-design-deep-dive.md`
