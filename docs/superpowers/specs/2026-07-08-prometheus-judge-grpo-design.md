# Prometheus-Vision No-Reference Judge and GRPO Teacher Design

## Objective

Build a no-reference VLM reward judge for PathVQA using Prometheus-Vision, compare the pretrained judge against a PathVQA-adapted judge, then use the better frozen judge as the reward source for GRPO teacher training.

The first experiment is intentionally controlled:

- Judge training data comes only from PathVQA train annotations.
- Judge evaluation uses PathVQA val annotations.
- The judge prompt does not include the ground-truth or reference answer.
- The judge is not trained on pseudo-QA.
- GRPO teacher training uses PathVQA train images only.
- The image pool is config-driven so external unlabeled pathology images can be swapped in later.

## Motivation

The recent VLM-as-a-judge literature supports using VLM judges for multimodal evaluation, but it also shows that judge scores are noisy and task-dependent. This design treats the judge as a reward signal to validate, not as ground truth.

Prometheus-Vision is a good starting point because it is an open VLM evaluator trained to follow user-defined rubrics and emit feedback plus a 1-5 score. The no-reference setup is closer to the eventual GRPO setting than a reference-answer scorer: the teacher will generate new QA pairs, and there will not be a ground-truth answer available for those candidates.

## Non-Goals

- Do not train the judge on teacher-generated pseudo-QA in this phase.
- Do not use PathVQA val or test to update the judge or teacher.
- Do not require external unlabeled pathology data for the first experiment.
- Do not modify the student training code path as part of this design.
- Do not claim the judge score is an absolute correctness probability.

## Data Contract

### PathVQA Splits

PathVQA converted files already follow the project format:

- `datasets/pathvqa/train.json`
- `datasets/pathvqa/val.json`
- `datasets/pathvqa/test.json`
- `datasets/pathvqa/images/`

The current converter writes records with:

- `image`
- `question`
- `answer`
- `dataset`
- `question_id`
- val/test also include `question_type` and `answer_type`

### Judge Train Data

For each PathVQA train record:

```text
positive = (image, question, ground_truth_answer)
negative = (image, question, corrupted_answer)
```

Negatives are generated only from PathVQA train information:

- random answer from train answer pool
- answer sampled from the same question prefix, such as `what`, `where`, `how many`, `is/are`
- answer sampled from the same inferred answer type, such as yes/no, count, short phrase
- yes/no flip for yes/no questions
- generic medical answers such as `tissue`, `cells`, `abnormality`, `lesion` when they differ from the gold answer
- same-image answer swap when an image has multiple train QA records

The train negative sampler must never read val or test answers.

### Judge Val Data

Val evaluation uses PathVQA val positives and corrupted val negatives. The negative sampling vocabulary must come from train-derived answer pools wherever a pool is needed. This lets val measure generalization without leaking val answers into the judge training process.

The val ground-truth answer is used only to define labels for evaluation metrics and construct the positive candidate. It is not included in the judge prompt.

## Judge Prompt

Prometheus-Vision expects image, instruction, response to evaluate, rubric, and usually a reference answer. In this design, the reference-answer field is set to a neutral no-reference value such as `No reference answer is provided. Judge only from the image, question, and candidate answer.`

The instruction:

```text
Answer the visual question about this pathology image:
{question}
```

The response to evaluate:

```text
{candidate_answer}
```

The rubric:

```text
Evaluate whether the candidate answer is correct for the pathology image and question.
Consider visual grounding, medical correctness, specificity, and usefulness for VQA teacher training.

Score 1: The answer is wrong, hallucinated, unrelated to the image/question, or medically misleading.
Score 2: The answer is mostly wrong but contains a minor relevant visual or medical term.
Score 3: The answer is partially correct, too generic, or missing important specificity.
Score 4: The answer is mostly correct and grounded but incomplete or slightly imprecise.
Score 5: The answer is correct, visually grounded, medically specific, and useful for training.
```

The parser extracts an integer score in `[1, 5]`. The scalar reward is:

```text
reward = (score - 1) / 4
```

## Model Strategy

### Stage A: Pretrained Prometheus-Vision Baseline

Run the pretrained Prometheus-Vision model in no-reference mode on PathVQA val judge examples.

Metrics:

- pairwise accuracy: `score(positive) > score(negative)`
- tie rate: `score(positive) == score(negative)`
- AUROC for positive-vs-negative examples
- mean margin: `score(positive) - score(negative)`
- score distribution by answer type and question prefix

This is the zero-shot baseline.

### Stage B: PathVQA-Adapted Judge

Fine-tune Prometheus-Vision on PathVQA train-derived pairwise examples.

Training objective:

```text
loss = -log sigmoid(score_positive - score_negative)
```

The implementation can realize this in one of two ways:

- direct scalar reward head if the model wrapper supports it cleanly
- token-logprob scoring over rating tokens `1` through `5`, using expected score or chosen score logits

The initial implementation should prefer the least invasive path that preserves compatibility with Hugging Face loading and LoRA/PEFT.

After fine-tuning, run the same val evaluation as Stage A. Use the fine-tuned judge for GRPO only if it improves val pairwise accuracy or AUROC without producing pathological score collapse.

### Stage C: Frozen Judge Reward for GRPO Teacher

Freeze the selected judge. The teacher generates `K` QA candidates for each image in the configured image pool.

For each candidate:

```text
candidate_i = (question_i, answer_i)
base_reward_i = judge.score(image, question_i, answer_i)
```

Apply lightweight penalties:

- duplicate question penalty within the same image group
- too-long question or answer penalty
- generic answer penalty
- yes/no overproduction penalty across a batch or epoch

Then group-normalize rewards for GRPO:

```text
reward_i = base_reward_i - penalties_i
advantage_i = (reward_i - mean(group_rewards)) / (std(group_rewards) + eps)
```

## Image Pool Abstraction

The first GRPO run uses PathVQA train images only. The code should still support a config-driven image pool.

Initial config shape:

```yaml
image_pool:
  name: pathvqa_train
  annotations: datasets/pathvqa/train.json
  image_root: datasets/pathvqa/images
  use_ground_truth_qa: false
```

Future external unlabeled image pool:

```yaml
image_pool:
  name: external_unlabeled_pathology
  annotations: datasets/external_pathology/images.json
  image_root: datasets/external_pathology/images
  use_ground_truth_qa: false
```

The GRPO teacher must not consume ground-truth train answers as targets during reward optimization. Train annotations are used only to enumerate train images in the first controlled run.

## Reward Hacking Monitoring

Log these during GRPO:

- mean judge reward
- reward standard deviation within candidate groups
- KL or token-level divergence from the base teacher policy when available
- question length
- answer length
- duplicate question rate
- yes/no question ratio
- generic answer rate
- held-out PathVQA val judge score, computed without teacher updates on val
- final downstream student accuracy on the selected evaluation split

Stop or flag a GRPO run if reward increases while diagnostics degrade, especially if duplicate rate, generic answer rate, or yes/no ratio rises sharply.

## Proposed Files

New package:

```text
judge/
  __init__.py
  data.py
  prometheus.py
  reward.py
```

New scripts:

```text
scripts/eval_prometheus_judge.py
scripts/train_prometheus_judge.py
scripts/train_teacher_grpo.py
```

New configs:

```text
configs/prometheus_judge_pathvqa.yaml
configs/grpo_teacher_pathvqa_prometheus.yaml
```

The existing `filtering/adapters.py` VLM backend can be reused where useful, but this judge module should remain separate from the pseudo-label filter gate to avoid mixing experiment concerns.

## Testing Plan

Unit tests:

- PathVQA train judge dataset builder emits positive and negative examples.
- Train negative sampler does not read val/test files.
- No-reference Prometheus prompt does not include ground-truth answer content or any non-neutral reference-answer content.
- Score parser extracts integer scores from valid Prometheus-style outputs.
- Score parser handles malformed output using a documented fallback.
- Reward conversion maps scores 1-5 to 0.0-1.0.
- Pairwise metrics handle ties explicitly.
- Image pool config can switch from PathVQA train to an external unlabeled pool.

Smoke tests:

- Build a tiny judge dataset from fixtures.
- Evaluate pretrained judge path with a mocked Prometheus scorer.
- Fine-tune script dry-run with mocked model outputs or a tiny local fake model.
- GRPO reward batch computes group-normalized advantages for `K` candidates with a mocked judge.

Manual validation:

- Inspect 20 positive/negative train pairs to confirm negatives are not trivial or mislabeled.
- Inspect 20 val scoring examples for Prometheus parse quality.
- After the first real judge eval, check score histograms for collapse to a single rating.

## Success Criteria

Pretrained judge baseline is useful if it beats random preference:

```text
pairwise_accuracy > 0.60
```

Fine-tuned judge is selected for GRPO if it improves over pretrained on PathVQA val by a meaningful margin:

```text
pairwise_accuracy improves by at least 0.03
or AUROC improves by at least 0.03
```

GRPO teacher training is considered promising only if higher judge reward is accompanied by stable diagnostics and improved downstream student validation or test accuracy.

## References

- Prometheus-Vision: Vision-Language Model as a Judge for Fine-Grained Evaluation, ACL Findings 2024.
- MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark, ICML 2024.
- Self-Improving VLM Judges Without Human Annotations, arXiv 2512.05145.
- VLM Judges Can Rank but Cannot Score: Task-Dependent Uncertainty in Multimodal Evaluation, arXiv 2604.25235.
