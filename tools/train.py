# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from ymir_exc.util import get_bool, get_merged_config

from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
from nanodet.util import NanoDetLightningLogger, cfg, convert_old_model, load_config, load_model_weight, mkdir
from ymir.utils import YmirMonitorCallback, modify_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--local_rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()
    return args


def main(args):
    load_config(cfg, args.config)
    ymir_cfg = get_merged_config()
    if get_bool(ymir_cfg, 'ddp_debug', False):
        os.environ.setdefault('TORCH_DISTRIBUTED_DETAIL', 'DEBUG')

    if get_bool(ymir_cfg, 'profiler', False):
        os.makedirs(ymir_cfg.ymir.output.models_dir, exist_ok=True)
        profiler = pl.profilers.SimpleProfiler(dirpath=ymir_cfg.ymir.output.models_dir, filename='profiler.txt')
    else:
        profiler = None

    modify_config(cfg, ymir_cfg)
    if cfg.model.arch.head.num_classes != len(cfg.class_names):
        raise ValueError("cfg.model.arch.head.num_classes must equal len(cfg.class_names), "
                         "but got {} and {}".format(cfg.model.arch.head.num_classes, len(cfg.class_names)))
    local_rank = int(args.local_rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    mkdir(local_rank, cfg.save_dir)

    mkdir(local_rank, ymir_cfg.ymir.output.tensorboard_dir)
    logger = NanoDetLightningLogger(ymir_cfg.ymir.output.tensorboard_dir)
    logger.dump_cfg(cfg)

    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        pl.seed_everything(args.seed)

    logger.info("Setting up data...")
    train_dataset = build_dataset(cfg.data.train, "train")
    val_dataset = build_dataset(cfg.data.val, "test")

    evaluator = build_evaluator(cfg.evaluator, val_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=True,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False,
    )

    logger.info("Creating model...")
    task = TrainingTask(cfg, evaluator)

    if "load_model" in cfg.schedule and os.path.exists(cfg.schedule.load_model):
        ckpt = torch.load(cfg.schedule.load_model)
        if "pytorch-lightning_version" not in ckpt:
            warnings.warn("Warning! Old .pth checkpoint is deprecated. "
                          "Convert the checkpoint with tools/convert_old_checkpoint.py ")
            ckpt = convert_old_model(ckpt)
        load_model_weight(task.model, ckpt, logger)
        logger.info("Loaded model weight from {}".format(cfg.schedule.load_model))

    model_resume_path = (os.path.join(ymir_cfg.ymir.input.models_dir, "model_last.ckpt")
                         if "resume" in cfg.schedule else None)

    if os.path.exists(model_resume_path):
        logger.info(f"resume from {model_resume_path}")
    else:
        warnings.warn(f'resume weight file {model_resume_path} not exist!')
        model_resume_path = None

    if cfg.device.gpu_ids == -1:
        logger.info("Using CPU training")
        accelerator, devices = "cpu", None
    else:
        accelerator, devices = "gpu", cfg.device.gpu_ids

    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        max_epochs=cfg.schedule.total_epochs,
        check_val_every_n_epoch=cfg.schedule.val_intervals,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=cfg.log.interval,
        num_sanity_val_steps=0,
        profiler=profiler,
        resume_from_checkpoint=model_resume_path,
        callbacks=[TQDMProgressBar(refresh_rate=0), YmirMonitorCallback()],  # disable tqdm bar
        logger=logger,
        benchmark=cfg.get("cudnn_benchmark", True),
        gradient_clip_val=cfg.get("grad_clip", 0.0),
    )

    trainer.fit(task, train_dataloader, val_dataloader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
