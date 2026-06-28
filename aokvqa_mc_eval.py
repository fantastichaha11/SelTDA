#!/usr/bin/env python3
"""A-OKVQA multiple-choice accuracy (same logic as aokvqa_mc_eval.ipynb)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

import cli
from data import create_dataset, create_loader
from models.blip_vqa import blip_vqa

logger = logging.getLogger(__name__)

AOKVQA_VAL_ANN = "aokvqa_v1p0_val.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A-OKVQA multiple-choice accuracy on the official val split."
    )
    parser.add_argument("--config", default="configs/aokvqa.yaml")
    parser.add_argument("--checkpoint", required=True, help="Student checkpoint .pth")
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSON metrics here (default: print only)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--med-config", default="configs/med_config.json")
    parser.add_argument("--overrides", nargs="+", default=[])
    return parser.parse_args()


def prep_answer_candidates(model, answer_list: list[str], device: str):
    answer_candidates = model.tokenizer(
        answer_list, padding="longest", return_tensors="pt"
    ).to(device)
    answer_candidates.input_ids[:, 0] = model.tokenizer.bos_token_id
    return answer_candidates


@torch.no_grad()
def evaluate_mc(
    model,
    val_loader,
    val_annotations: list[dict],
    device: str,
) -> dict:
    if len(val_annotations) != len(val_loader.dataset):
        raise ValueError(
            f"Val annotation count ({len(val_annotations)}) != "
            f"dataset length ({len(val_loader.dataset)})"
        )

    model_choices: list[str] = []
    correct_choices: list[str] = []

    sample_idx = 0
    progress = tqdm(total=len(val_loader.dataset), desc="A-OKVQA MC eval")
    for image, question, _question_id in val_loader:
        batch_size = int(image.shape[0])
        image = image.to(device, non_blocking=True)

        for b in range(batch_size):
            ann = val_annotations[sample_idx]
            answer_list = ann["choices"]
            answer_candidates = prep_answer_candidates(model, answer_list, device=device)
            correct_answer = answer_list[ann["correct_choice_idx"]]

            if isinstance(question, (list, tuple)):
                sample_question = [question[b]]
            else:
                sample_question = [question]

            answer_ids = model(
                image[b : b + 1],
                sample_question,
                answer_candidates,
                train=False,
                inference="rank",
                k_test=len(answer_list),
            )
            answer_id = int(answer_ids.detach().cpu().reshape(-1)[0].item())
            model_choices.append(answer_list[answer_id])
            correct_choices.append(correct_answer)
            sample_idx += 1
            progress.update(1)
    progress.close()

    correct = sum(m == c for m, c in zip(model_choices, correct_choices))
    total = len(model_choices)
    accuracy = correct / total if total else 0.0

    return {
        "metric": "aokvqa_mc_accuracy",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "model_choices": model_choices,
        "correct_choices": correct_choices,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    config = cli.load_config(
        argparse.Namespace(config=args.config, overrides=args.overrides)
    )
    config.use_validation_set_as_test_set = True

    ann_root = Path(config.ann_root)
    val_ann_path = ann_root / AOKVQA_VAL_ANN
    if not val_ann_path.is_file():
        raise FileNotFoundError(
            f"Missing {val_ann_path} (download A-OKVQA via dataset.sh)"
        )

    with val_ann_path.open(encoding="utf-8") as f:
        val_annotations = json.load(f)

    _train, val = create_dataset(config.dataset_name, config)
    val_loader, = create_loader(
        datasets=(val,),
        samplers=(None,),
        batch_size=(args.batch_size,),
        num_workers=(args.num_workers,),
        is_trains=(False,),
        collate_fns=(None,),
    )

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    logger.info("Loading checkpoint %s", checkpoint)
    model = blip_vqa(
        pretrained=str(checkpoint),
        image_size=config.image_size,
        vit=config.vit,
        vit_grad_ckpt=config.vit_grad_ckpt,
        vit_ckpt_layer=config.vit_ckpt_layer,
        med_config=args.med_config,
    )
    model.eval()
    model.to(device)

    results = evaluate_mc(model, val_loader, val_annotations, device=device)
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "val_annotations": str(val_ann_path.resolve()),
        "metric": results["metric"],
        "accuracy": results["accuracy"],
        "correct": results["correct"],
        "total": results["total"],
    }

    print(
        f"A-OKVQA MC accuracy: {summary['accuracy']:.4%} "
        f"({summary['correct']}/{summary['total']})"
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
