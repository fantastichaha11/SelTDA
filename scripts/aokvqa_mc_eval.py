#!/usr/bin/env python3
"""A-OKVQA multiple-choice accuracy on the val split (ported from aokvqa_mc_eval.ipynb)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import create_dataset, create_loader
from models.blip_vqa import blip_vqa


def prep_answer_candidates(model, answer_list, device):
    answer_candidates = model.tokenizer(
        answer_list, padding="longest", return_tensors="pt"
    ).to(device)
    answer_candidates.input_ids[:, 0] = model.tokenizer.bos_token_id
    return answer_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Student checkpoint .pth")
    parser.add_argument("--config", default="configs/aokvqa.yaml")
    parser.add_argument("--val-json", default="datasets/aokvqa/aokvqa_v1p0_val.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = OmegaConf.load(ROOT / args.config)
    config.use_validation_set_as_test_set = True
    config.pretrained = args.checkpoint
    config.batch_size_test = int(config.get("batch_size_test", 16))

    device = args.device if torch.cuda.is_available() else "cpu"
    _, val_ds = create_dataset("aokvqa", config)
    val_loader = create_loader([val_ds], [None], [config.batch_size_test], [4], [False])[1]

    model = blip_vqa(
        pretrained=args.checkpoint,
        image_size=config.image_size,
        vit=config.vit,
        vit_grad_ckpt=config.vit_grad_ckpt,
        vit_ckpt_layer=config.vit_ckpt_layer,
    )
    model.eval().to(device)

    val_path = ROOT / args.val_json
    with val_path.open() as f:
        val_annotations = json.load(f)

    model_choices = []
    correct_choices = []
    for n, (image, question, _qid) in enumerate(
        tqdm(val_loader, total=len(val_loader), desc="A-OKVQA MC val")
    ):
        ann = val_annotations[n]
        answer_list = ann["choices"]
        answer_candidates = prep_answer_candidates(model, answer_list, device)
        image = image.to(device, non_blocking=True)
        correct = answer_list[ann["correct_choice_idx"]]
        answer_ids = model(
            image,
            question,
            answer_candidates,
            train=False,
            inference="rank",
            k_test=len(answer_list),
        )
        idx = int(answer_ids[0]) if torch.is_tensor(answer_ids) else int(answer_ids)
        model_choices.append(answer_list[idx])
        correct_choices.append(correct)

    acc = sum(m == c for m, c in zip(model_choices, correct_choices)) / len(correct_choices)
    print(f"A-OKVQA val MC accuracy: {acc * 100:.2f}% ({len(correct_choices)} questions)")
    return acc


if __name__ == "__main__":
    main()
