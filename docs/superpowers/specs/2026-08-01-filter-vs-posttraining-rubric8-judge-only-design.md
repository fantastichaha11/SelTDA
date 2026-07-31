# Filter vs. Post-training, Rubric-8, Judge-only, and 60K Data-Scaling Experiment Design

**Date:** 2026-08-01

**Status:** Approved conversational design; pending written-spec review

**Datasets:** PathVQA and VizWiz

**Compute window:** 2--3 days

**Fixed training seed:** 42

## 1. Objective

This experiment package tests four distinct claims without rerunning conditions that already exist:

1. Post-training the synthetic-data teacher is more effective than globally filtering a larger candidate pool when both methods use the same original four-criterion judge rubric.
2. Expanding the judge rubric from four to eight criteria improves post-training when all external reward penalties remain fixed.
3. The Prometheus judge score alone provides a useful post-training signal, independently of the four hand-designed reward penalties.
4. Increasing the Full student training set from 40,000 to 60,000 examples improves downstream performance under the same teacher, rubric, penalties, and student recipe.

The experiment does not claim robustness across training seeds. All new conditions use seed 42 to match the existing checkpoints. Paired uncertainty over test examples will be reported separately and will not be presented as seed-level uncertainty.

## 2. Experimental Questions and Comparisons

| Question | Primary comparison | Controlled variables |
|---|---|---|
| Does post-training outperform filtering? | `PT-4-J` vs. `Filter-4` | Rubric-4 judge score only, original teacher initialization, final 40,000 student examples, student recipe, seed |
| Do the external penalties help? | `PT-4-JP` vs. `PT-4-J` | Rubric-4, post-training recipe, final data budget, student recipe, seed |
| Does the expanded rubric help? | `PT-8-JP` vs. `PT-4-JP` | All four penalty weights, post-training recipe, final data budget, student recipe, seed |
| Does judge-only post-training help over no judge? | `PT-4-J` vs. `SelTDA` | Original teacher family, final data budget, student recipe, seed |
| Does more Full training data help? | `PT-4-JP-60K` vs. `PT-4-JP` | Same post-trained teacher, Rubric-4, penalties, real subset, student initialization, epochs, seed; only the total example count changes |
| Which complete condition performs best? | All conditions | Same downstream validation/test protocol |

`PT-4-J` is the only new post-training condition that removes the external penalties. `PT-4-JP`, `PT-4-JP-60K`, and `PT-8-JP` retain the existing penalty weights.

## 3. Conditions

Run the same matrix on PathVQA and VizWiz.

| ID | Teacher/data method | Rubric | External penalties | Status |
|---|---|---|---|---|
| `SelTDA` | Original SelTDA generation | None | None | Reuse existing result |
| `Filter-4` | Generate a 2x global pool from the original teacher, judge, rank globally, retain the required quota | Rubric-4 | Not applied | New |
| `PT-4-J` | Post-train the original teacher | Rubric-4 | All set to zero | New |
| `PT-4-JP` | Existing Full condition | Rubric-4 | Existing four penalties | Reuse existing result |
| `PT-4-JP-60K` | Reuse the `PT-4-JP` teacher and expand its student data from 40K to 60K | Rubric-4 | Existing four penalties | New |
| `PT-8-JP` | Post-train the original teacher | Rubric-8 | Existing four penalties | New |

There is no `Filter-8` condition and no `PT-8-J` condition in this design.

Across both datasets, the design therefore requires eight new downstream conditions: `Filter-4`, `PT-4-J`, `PT-8-JP`, and `PT-4-JP-60K` for PathVQA, plus the same four conditions for VizWiz. Existing `SelTDA` and 40K `PT-4-JP` results are reused. `PT-4-JP-60K` does not retrain the teacher; it only extends synthetic generation and trains a new student.

## 4. Shared Data Budget

For each dataset, define the 40K and 60K synthetic quotas:

\[
N_{\mathrm{syn},40} = 40{,}000 - N_{\mathrm{real}},
\qquad
N_{\mathrm{syn},60} = 60{,}000 - N_{\mathrm{real}},
\]

where `N_real` is the number of real examples actually admitted by the downstream student loader. Every condition except `PT-4-JP-60K` must satisfy:

\[
N_{\mathrm{real}} + N_{\mathrm{syn},40} = 40{,}000.
\]

The `PT-4-JP-60K` condition must satisfy:

\[
N_{\mathrm{real}} + N_{\mathrm{syn},60} = 60{,}000.
\]

The run manifest must record `N_real`, the applicable synthetic quota, and the final loader-observed total. A condition is invalid if it merely sets a truncation maximum but does not demonstrate that the loader admitted exactly 40,000 or 60,000 examples as specified.

All methods use the same real examples. The 60K dataset must be nested: it contains every real and synthetic record used by the 40K `PT-4-JP` condition, plus exactly 20,000 additional valid synthetic records from the same teacher checkpoint. This prevents data-composition changes from being mistaken for a data-volume effect.

## 5. Global Filter Protocol

`Filter-4` implements the user's global-ranking definition, not best-of-K selection and not a per-image quota.

1. Start from the original, non-post-trained teacher checkpoint used by SelTDA.
2. Generate until the eligible raw pool contains at least `2 * N_syn_40` pseudo-QA examples. If the final generation batch overshoots, retain the first `2 * N_syn_40` eligible examples by stable generation index so the scored pool is exactly 2x.
3. An eligible example must parse successfully, resolve to an existing image, contain a non-empty question and answer, and not duplicate the tuple `(canonical image identifier, normalized question, normalized answer)`. The same question or answer may appear for different images. Invalid or duplicate generations do not count toward the 2x target and must be replenished.
4. Score every eligible example once with the dataset's Rubric-4 Prometheus judge at temperature 0.
5. Sort the entire dataset-level pool by descending judge score. Do not impose a per-image quota.
6. Resolve equal integer scores using a deterministic seed-42 tie key stored in the output manifest.
7. Retain the first `N_syn_40` examples and combine them with the fixed real subset.

The filtered output must report:

- raw generation attempts;
- eligible pool size, which must equal `2 * N_syn_40`;
- retained synthetic count, which must equal `N_syn_40`;
- invalid and exact-duplicate rejection counts;
- unique-image coverage;
- duplicate-question and duplicate-QA rates after selection;
- question-prefix and answer-type distributions;
- judge calls and elapsed time.

Global ranking is intentionally allowed to concentrate on a subset of images. Coverage is measured and reported rather than corrected with a hidden per-image constraint.

## 6. Post-training Protocol

All new post-training conditions use the existing Full recipe unless this document explicitly changes a reward field:

- original dataset-specific teacher initialization;
- 8 candidates per image;
- 200 update steps per epoch for 3 epochs, totaling 600 updates;
- batch size 2;
- learning rate `1e-6`;
- generation length 5--40 tokens;
- top-p `0.9`;
- seed 42;
- the same image-pool construction and checkpoint schedule.

After post-training, each 40K condition generates exactly `N_syn_40` valid synthetic examples using the same downstream generation settings. Each resulting student uses the same initialization, real subset, optimizer, batch size, epoch count, data order seed, and evaluation procedure.

### 6.1 Existing penalty condition (`JP`)

`PT-4-JP`, `PT-4-JP-60K`, and `PT-8-JP` use the existing penalty weights:

| Penalty | Weight |
|---|---:|
| Duplicate question | 0.35 |
| Yes/no answer | 0.35 |
| Length | 0.10 |
| Generic answer | 0.15 |

The maximum question and answer lengths remain 30 and 12 words for penalty calculation.

### 6.2 Judge-only condition (`J`)

`PT-4-J` uses the old Rubric-4 judge and sets all four external penalty weights to `0.0`. If the integer judge score is `s` in `[1, 5]`, its only scalar reward is:

\[
r_{\mathrm{judge}} = \frac{s-1}{4}.
\]

Group-normalized advantage computation over the eight candidates remains part of optimization. It transforms the judge rewards but introduces no additional reward signal. No duplicate, yes/no, length, or generic-answer adjustment is permitted in this condition.

### 6.3 60K data-scaling condition

`PT-4-JP-60K` reuses the exact teacher checkpoint from the existing 40K `PT-4-JP` condition. It does not perform additional teacher updates and does not change the rubric or penalty weights.

Its student dataset is constructed as follows:

1. Preserve the complete real subset and `N_syn_40` synthetic subset used by the 40K `PT-4-JP` student.
2. Resume after the final accepted generation index stored by the 40K manifest, using the same teacher, generation settings, and seed-42 deterministic ordering, until 20,000 additional valid records are available. Do not restart generation and replace the existing prefix.
3. Reject only invalid records and duplicates of `(canonical image identifier, normalized question, normalized answer)` already present in the nested dataset.
4. Append the 20,000 records without re-ranking or filtering, yielding exactly `N_syn_60` synthetic records and 60,000 total records.
5. Train a new student from the same initialization for the same 10 epochs, with the same optimizer, batch size, data-order seed, and evaluation protocol as the 40K student.

Keeping the epoch count fixed means the 60K student performs more optimizer steps and consumes more compute. The comparison measures the practical benefit of adding data under the standard training recipe; it is not an equal-step or equal-compute data-efficiency experiment. The report must include the optimizer-step and GPU-hour increase.

## 7. Rubrics

Rubric-4 is the exact text used by the existing Full run. For PathVQA, this means the `DEFAULT_RUBRIC` text currently loaded because the PathVQA YAML names a rubric but does not embed one. The implementation must snapshot the effective rubric text in every run directory; `rubric_name` alone is not sufficient provenance.

Rubric-8 retains the four original concerns and adds four explicit criteria. It produces one holistic integer score from 1 to 5, preserving the reward scale used by Rubric-4.

### 7.1 PathVQA Rubric-8

The eight criteria are:

1. visual grounding;
2. medical correctness;
3. answer specificity;
4. practical usefulness for the question;
5. question answerability from visible image evidence;
6. discriminative pathological or morphological content;
7. evidence-calibrated diagnostic scope;
8. usefulness as a non-trivial synthetic training example.

Use the following rubric text:

> You are evaluating a synthetic question-answer pair intended for training a visual question answering model on pathology images. Use only the visible image, the synthetic question, and the synthetic answer. There is no reference answer.
>
> Evaluate the pair on eight criteria: (1) the answer is grounded in visible image evidence; (2) the answer is medically correct; (3) the answer is appropriately specific; (4) the answer is practically useful for the question; (5) the question itself is answerable from the visible image; (6) the pair targets a discriminative pathological or morphological feature when such a feature is visible; (7) the diagnostic scope and medical terminology do not exceed what the image supports; and (8) the pair provides a meaningful, non-trivial training signal rather than a vague or generic association.
>
> A yes/no answer is acceptable when the question is meaningful and the image provides sufficient evidence. Do not reward specialized medical terminology merely because it sounds precise. Judge semantic quality rather than mechanically preferring a fixed answer length.
>
> Score 1: The question is invalid or not answerable from the image, or the answer is incorrect, hallucinated, unrelated, or medically misleading.
>
> Score 2: The pair is weakly related to the image, but its answer is unreliable, ambiguous, overly generic, or supported by insufficient visual evidence.
>
> Score 3: The question is probably answerable and the answer is plausible, but the pair is imprecise, trivial, incomplete, diagnostically over-broad, or only moderately useful for training.
>
> Score 4: The question is clearly answerable from the image and the answer is visually grounded, medically sound, appropriately specific, concise, and useful, with at most minor omissions.
>
> Score 5: The pair satisfies all requirements of Score 4 and targets a clear discriminative pathological or morphological feature, providing an unambiguous, evidence-calibrated, and high-value training example.

### 7.2 VizWiz Rubric-8

The eight criteria are:

1. visual grounding;
2. direct relevance to the question;
3. appropriate specificity and usefulness;
4. avoidance of hallucinated, unsafe, or misleading claims;
5. compatibility between the question and the image content;
6. awareness of visibility limits caused by blur, occlusion, framing, or lighting;
7. correct answerable-versus-unanswerable calibration;
8. usefulness as a non-trivial synthetic training example.

Use the following rubric text:

> You are evaluating a synthetic question-answer pair intended for training a visual question answering system for blind or low-vision users. Use only the visible image, the synthetic question, and the synthetic answer. There is no reference answer.
>
> Evaluate the pair on eight criteria: (1) every asserted object, attribute, text, count, or spatial relation is visually grounded; (2) the answer directly addresses the question; (3) the answer is appropriately specific and useful; (4) the answer avoids hallucinated, unsafe, or misleading claims; (5) the question is compatible with content that is present or reasonably requested from the image; (6) the pair recognizes visibility limits caused by blur, occlusion, framing, lighting, or missing evidence; (7) it chooses the answerable or unanswerable mode correctly; and (8) it provides a meaningful, non-trivial training signal.
>
> An unanswerable response is correct only when the requested information genuinely cannot be recovered from the image. Do not reward an unanswerable response when the information is clearly visible. Yes/no answers are acceptable when visually grounded. Judge semantic quality rather than mechanically preferring a fixed answer length.
>
> Score 1: The pair is invalid, unrelated, contradicted by the image, confidently hallucinates information, selects a clearly wrong answerability mode, or could seriously mislead the user.
>
> Score 2: The pair has slight relevance but is weakly grounded, vague, insufficiently supported, or poorly calibrated to image visibility.
>
> Score 3: The answer is plausible and mostly relevant, but the pair is incomplete, imprecise, generic, incorrectly confident, or only moderately useful for training.
>
> Score 4: The answer is clearly grounded, direct, appropriately specific, safe, and useful, or it correctly identifies that the requested information is unanswerable, with at most minor omissions.
>
> Score 5: The pair satisfies all requirements of Score 4 and provides a highly clear, unambiguous, visibility-calibrated, and informative training example without unsupported details.

## 8. Fairness Controls

The following fields must be identical across conditions unless the experimental definition above requires a difference:

- seed 42;
- original teacher checkpoint for filtering and post-training initialization;
- teacher architecture;
- K=8 for post-training;
- teacher update count, optimizer, learning rate, and sampling parameters;
- final real and synthetic counts, except for the pre-registered 40K-to-60K scaling difference;
- student initialization and architecture;
- student optimizer, batch size, epochs, and data-order seed;
- validation and test splits;
- evaluation code and metric version.

The filter receives a 2x candidate pool by design. This is not an equal-generation-compute comparison. The core 40K conditions have an equal final student data budget, while generation cost, judge calls, GPU-hours, and discarded candidates are reported as efficiency outcomes. `PT-4-JP-60K` is excluded from equal-data comparisons and is interpreted only against the nested 40K `PT-4-JP` condition.

## 9. Evaluation

### 9.1 Primary downstream results

Report validation and test accuracy for every condition. Report the overall score and available strata:

- PathVQA: yes/no versus open-ended and question/answer type when metadata permits;
- VizWiz: yes/no, number, other, and unanswerable.

For each test question, retain the per-example score. Use 10,000 paired bootstrap resamples to report a 95% confidence interval for these pre-registered differences:

- `PT-4-J - Filter-4`;
- `PT-4-JP - PT-4-J`;
- `PT-8-JP - PT-4-JP`;
- `PT-4-J - SelTDA`;
- `PT-4-JP-60K - PT-4-JP`.

The report must include the absolute score change, not only relative percentage improvement. Bootstrap intervals characterize test-example uncertainty for fixed checkpoints; they must not be described as training-seed confidence intervals.

### 9.2 Data-quality diagnostics

For the final synthetic datasets from `Filter-4`, `PT-4-J`, `PT-4-JP`, `PT-4-JP-60K`, and `PT-8-JP`, report:

- unique-image coverage;
- exact question and exact QA duplicate rates;
- question-prefix distribution;
- answer-type distribution;
- yes/no, generic-answer, and unanswerable rates where applicable;
- mean and percentile question/answer lengths.

These diagnostics explain whether global ranking collapses coverage and whether the penalties or expanded rubric change the generated-data distribution.

### 9.3 Rubric signal diagnostic

Build a fixed probe set of 200 generated QA pairs per dataset before starting the full Rubric-8 run. Use 25 fixed validation-split images, the original teacher, seed 42, and eight candidates per image, yielding 25 natural K=8 groups. These probe outputs are analysis-only and never enter teacher updates, filtering, or student training. Score every probe pair with Rubric-4 and Rubric-8 using the same judge checkpoint and temperature 0. Report:

- mean score under each rubric;
- mean paired score difference;
- Spearman rank correlation;
- score distribution by question prefix and answer type;
- within-group winner-change rate when probe examples are organized into K=8 groups.

This diagnostic demonstrates that the eight-criterion rubric changes the reward signal. It is not a `Filter-8` training condition and its scored outputs must not be used to select the downstream training set.

### 9.4 Efficiency accounting

Record for every new condition:

- generated candidates and retained candidates;
- discarded candidates;
- judge calls;
- teacher post-training time;
- synthetic generation time;
- student training time;
- total GPU-hours;
- GPU model and peak allocated memory when available.

Wall-clock and GPU-hour comparisons are valid only for conditions run on the same hardware and software stack. Candidate counts and judge calls remain reportable when hardware differs.

## 10. Interpretation Rules

The thesis may make the following claims only when the associated comparison supports them:

- If `PT-4-J > Filter-4`, post-training outperforms global filtering under the same Rubric-4 judge-only signal and final data budget.
- If `PT-4-JP > PT-4-J`, the external penalties provide additional value beyond the judge score. If the ordering reverses, report that the penalties are unnecessary or harmful under this setup.
- If `PT-8-JP > PT-4-JP`, the expanded dataset-specific rubric improves post-training while penalty weights are held fixed.
- If `PT-4-JP-60K > PT-4-JP`, adding 20,000 synthetic examples improves the practical Full training recipe. This result must not be described as an equal-compute gain because the 60K student performs more optimizer steps.
- `PT-8-JP > Filter-4` may identify the best complete system, but it cannot isolate the effect of post-training because both rubric and reward construction differ.
- A positive result at seed 42 is evidence for the fixed-seed experiment, not proof of robustness across random initializations.

The strongest result requires consistent positive downstream changes on both datasets, positive paired bootstrap intervals, and data-quality diagnostics that do not reveal severe coverage collapse or duplication. Mixed results must be described by dataset rather than averaged into a universal claim.

## 11. Run Order and Compute Protection

Run the new conditions in the following order:

1. Validate counts, effective rubric snapshots, score parsing, deterministic tie-breaking, and output paths with a small non-reportable smoke run.
2. Run `Filter-4` and `PT-4-J` on PathVQA, then train and evaluate their students. This establishes the main filter-versus-post-training comparison early.
3. Run `Filter-4` and `PT-4-J` on VizWiz and evaluate their students.
4. Run `PT-8-JP` on PathVQA and VizWiz, then train and evaluate their students.
5. Extend the existing `PT-4-JP` synthetic datasets and train the 60K students for PathVQA and VizWiz.
6. Run diagnostics, paired bootstrap, and efficiency aggregation.

Every expensive stage must support resume without overwriting a completed checkpoint or dataset. Each condition writes to a distinct output directory. A run stops and is invalidated if it produces non-finite losses, fails to reach the specified update count, cannot prove its required 40,000- or 60,000-example loader count, or silently falls back to a different rubric or teacher checkpoint.

If the 2--3 day window becomes binding, preserve the order above. The `PT-4-J` versus `Filter-4` comparison is the primary claim; Rubric-8 is the next priority. Do not shorten only one side of a controlled comparison and then report it as equivalent.

## 12. Artifacts and Reproducibility

Each condition must store:

- fully resolved configuration;
- git commit and dirty-worktree indicator;
- seed and hardware information;
- original and effective rubric text;
- teacher initialization and final checkpoint paths;
- candidate/reward logs;
- final synthetic JSON and content hash;
- final student data manifest with real/synthetic counts;
- student checkpoint and predictions;
- validation/test metrics and per-example scores;
- elapsed-time and judge-call summary.

Existing `SelTDA` and `PT-4-JP` results may be reused only if their checkpoint, data manifest, seed, and evaluation artifacts can be identified. Missing provenance must be disclosed rather than reconstructed by assumption.

## 13. Implementation Scope After Spec Approval

The later implementation plan will cover only work required by this design:

- dataset-specific Rubric-8 configuration;
- Rubric-4 judge-only post-training configurations;
- global 2x pool ranking and exact-count manifests;
- nested 40K-to-60K Full data construction and student scripts;
- isolated PathVQA and VizWiz run scripts;
- data-quality, bootstrap, rubric-diagnostic, and efficiency reports;
- focused tests for reward isolation, deterministic ranking, exact counts, and resolved-rubric provenance.

Unrelated notebook changes, existing user checkpoints, untracked research papers, and other dirty-worktree files are outside scope and must remain untouched.

## 14. Acceptance Criteria

The experiment package is complete when:

1. all eight new downstream conditions finish or a failure is explicitly documented;
2. every core downstream loader admits exactly 40,000 examples, and both `PT-4-JP-60K` loaders admit exactly 60,000;
3. `Filter-4` ranks an eligible global pool of exactly `2 * N_syn_40` without a per-image quota;
4. `PT-4-J` has zero external penalties and uses only the Rubric-4 score as reward;
5. `PT-4-JP`, `PT-4-JP-60K`, and `PT-8-JP` use identical nonzero penalty settings;
6. each 60K dataset contains the complete corresponding 40K `PT-4-JP` dataset plus exactly 20,000 new synthetic records from the same teacher;
7. the effective four- and eight-criterion rubric texts are archived;
8. validation/test metrics, paired bootstrap intervals, data diagnostics, and compute accounting are produced for both datasets;
9. claims in the thesis follow the interpretation rules in Section 10.
