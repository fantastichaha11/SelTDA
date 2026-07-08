import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm

from judge.data import build_answer_pools, build_judge_pairs, infer_answer_type, load_records, question_prefix
from judge.prometheus import PrometheusVisionScorer, StaticPrometheusScorer
from judge.reward import compute_pairwise_metrics, score_distribution


def build_scorer(config):
    mock_score = OmegaConf.select(config, "eval.mock_score", default=None)
    if mock_score is not None:
        return StaticPrometheusScorer(score=int(mock_score), feedback="mock")

    model_cfg = config.model
    return PrometheusVisionScorer(
        model_path=str(model_cfg.model_path),
        model_base=None
        if str(model_cfg.model_base).lower() in {"none", "null"}
        else str(model_cfg.model_base),
        conv_mode=str(model_cfg.conv_mode),
        device=str(model_cfg.device),
        temperature=float(model_cfg.temperature),
        max_new_tokens=int(model_cfg.max_new_tokens),
    )


def run_eval(config, scorer=None):
    train_records = load_records(config.data.train_annotations)
    val_records = load_records(config.data.val_annotations)
    max_examples = OmegaConf.select(config, "eval.max_examples", default=None)
    if max_examples is not None:
        val_records = val_records[: int(max_examples)]

    pools = build_answer_pools(train_records)
    pairs = build_judge_pairs(
        val_records,
        pools=pools,
        seed=int(OmegaConf.select(config, "train.seed", default=42)),
        negatives_per_positive=int(config.data.negatives_per_positive),
    )

    scorer = scorer or build_scorer(config)
    image_root = Path(str(config.data.image_root))
    positive_scores = []
    negative_scores = []
    examples = []

    for pair in tqdm(pairs, desc="Prometheus judge eval"):
        positive_result = scorer.score(
            str(image_root / pair.positive.image),
            pair.positive.question,
            pair.positive.answer,
        )
        negative_result = scorer.score(
            str(image_root / pair.negative.image),
            pair.negative.question,
            pair.negative.answer,
        )
        positive_scores.append(float(positive_result.score))
        negative_scores.append(float(negative_result.score))
        examples.append(
            {
                "image": pair.positive.image,
                "question": pair.positive.question,
                "positive_answer": pair.positive.answer,
                "negative_answer": pair.negative.answer,
                "positive_score": positive_result.score,
                "negative_score": negative_result.score,
                "question_prefix": question_prefix(pair.positive.question),
                "positive_answer_type": infer_answer_type(pair.positive.answer),
                "negative_answer_type": infer_answer_type(pair.negative.answer),
                "positive_feedback": positive_result.feedback,
                "negative_feedback": negative_result.feedback,
            }
        )

    metrics = compute_pairwise_metrics(positive_scores, negative_scores)
    metrics["num_pairs"] = len(pairs)

    distribution_rows = []
    for example in examples:
        distribution_rows.append(
            {
                "question_prefix": example["question_prefix"],
                "answer_type": example["positive_answer_type"],
                "score": example["positive_score"],
            }
        )
        distribution_rows.append(
            {
                "question_prefix": example["question_prefix"],
                "answer_type": example["negative_answer_type"],
                "score": example["negative_score"],
            }
        )
    metrics["score_distribution"] = score_distribution(distribution_rows)

    output_path = Path(str(config.eval.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"metrics": metrics, "examples": examples}, indent=2),
        encoding="utf-8",
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prometheus_judge_pathvqa.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--mock-score", type=int, default=None)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if args.output is not None:
        config.eval.output = args.output
    if args.max_examples is not None:
        config.eval.max_examples = args.max_examples
    if args.mock_score is not None:
        config.eval.mock_score = args.mock_score

    metrics = run_eval(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
