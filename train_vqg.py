import argparse
import os
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
import hydra
from omegaconf import OmegaConf
# import wandb
import data

from models.blip import decoder_from_config
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval
import cli
import wandb_utils


class Trainer:
    def __init__(
        self, model, data_loader, optimizer, device, wandb_logger=None, print_freq=50
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.device = device
        # self.wandb_logger = wandb_logger
        self.print_freq = print_freq

    def train_one_epoch(self, epoch):
        self.model.train()

        self.metric_logger = utils.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter(
            "lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
        self.metric_logger.add_meter(
            "loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
        )
        header = "Train Caption Epoch: [{}]".format(epoch)
        print_freq = 50

        for i, (image, caption, _) in enumerate(
            self.metric_logger.log_every(self.data_loader, print_freq, header)
        ):
            self.train_step(image, caption, i)

        # gather the stats from all processes
        self.metric_logger.synchronize_between_processes()
        print("Averaged stats:", self.metric_logger.global_avg())
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in self.metric_logger.meters.items()
        }

    def train_step(self, image, caption, batch_idx):
        image = image.to(self.device)

        loss = self.model(image, caption)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.metric_logger.update(loss=loss.item())
        self.metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        # if batch_idx % self.print_freq == 0:
        #     if utils.is_main_process() and self.wandb_logger:
        #         self.wandb_logger.log(
        #             data={
        #                 "loss": loss.item(),
        #                 "lr": self.optimizer.param_groups[0]["lr"],
        #             }
        #         )


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Caption generation:"
    print_freq = 10

    result = []
    for image, image_id in metric_logger.log_every(data_loader, print_freq, header):

        image = image.to(device)

        captions = model.generate(
            image,
            sample=False,
            num_beams=config["num_beams"],
            max_length=config["max_length"],
            min_length=config["min_length"],
        )

        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})

    return result


def main(args, config):
    utils.init_distributed_mode(args)
    wandb_logger = wandb_utils.init_wandb(args, config, job_type="vqg-train")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating captioning dataset")
    train_dataset, test_dataset = create_dataset("vqg", config)
    if utils.is_main_process():
        wandb_logger.update_summary(
            {
                "train_dataset_size": len(train_dataset),
                "eval_dataset_size": len(test_dataset),
                "world_size": utils.get_world_size(),
                "seed": seed,
                "evaluate_only": args.evaluate,
            },
            prefix="run",
        )

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(
            [train_dataset, test_dataset],
            [True, False],
            num_tasks,
            global_rank,
        )
    else:
        samplers = [None, None]

    train_loader, test_loader = create_loader(
        [train_dataset, test_dataset],
        samplers,
        batch_size=[config["batch_size"]] * 3,
        num_workers=[4, 4, 4],
        is_trains=[True, False],
        collate_fns=[None, None],
    )

    #### Model ####
    print("Creating model")
    model = decoder_from_config(config)

    model = model.to(device)
    if utils.is_main_process() and bool(OmegaConf.select(config, "wandb_watch_model", default=True)):
        wandb_logger.watch_model(
            model,
            log=str(OmegaConf.select(config, "wandb_watch_log", default="all")),
            log_freq=int(OmegaConf.select(config, "wandb_watch_log_freq", default=100)),
        )

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config["init_lr"],
        weight_decay=config["weight_decay"],
    )

    best = 0
    best_epoch = 0

    trainer = Trainer(
        data_loader=train_loader,
        optimizer=optimizer,
        device=device,
        # wandb_logger=wandb_logger,
        model=model,
    )

    print("Start training")
    start_time = time.time()
    epochs = list(range(config.max_epoch))
    for epoch in epochs:
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(
                optimizer,
                epoch,
                config["max_epoch"],
                config["init_lr"],
                config["min_lr"],
            )

            # train_stats = train(model, train_loader, optimizer, epoch, device, wandb_logger=wandb_logger)
            train_stats = trainer.train_one_epoch(epoch=epoch)
            if utils.is_main_process():
                wandb_logger.log_metrics(train_stats, prefix="train", step=epoch)
                wandb_logger.log_metrics({"epoch": epoch}, prefix="run", step=epoch)

        if utils.is_main_process():

            if args.evaluate:
                pass
            else:

                if config.save_last_only:
                    should_save = epoch == epochs[-1]
                else:
                    should_save = True

                if should_save:
                    save_obj = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": config,
                        "epoch": epoch,
                    }
                    torch.save(
                        save_obj,
                        os.path.join(args.output_dir, "checkpoint_%02d.pth" % epoch),
                    )
                    wandb_logger.log_artifact(
                        os.path.join(args.output_dir, "checkpoint_%02d.pth" % epoch),
                        name=f"{Path(args.output_dir).name}-checkpoint-{epoch:02d}",
                        artifact_type="model",
                        metadata={"epoch": epoch, "dataset_name": str(config.dataset_name)},
                    )

                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                }

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if args.evaluate:
            break
        # dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    if utils.is_main_process():
        wandb_logger.update_summary({"total_seconds": total_time}, prefix="run")
        wandb_logger.log_artifact(
            os.path.join(args.output_dir, "config.yaml"),
            name=f"{Path(args.output_dir).name}-config",
            artifact_type="config",
        )
        wandb_logger.finish()


if __name__ == "__main__":
    args, config = cli.parse_args(default_config_path="configs/vqg.yaml")
    cli.setup(args, config)
    main(args, config)
