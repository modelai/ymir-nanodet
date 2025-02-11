import glob
import logging
import os.path as osp
from typing import List

import cv2
import pytorch_lightning as pl
import torch
import torch.utils.data as td
from easydict import EasyDict as edict
from nanodet.data.transform import Pipeline
from nanodet.util.yacs import CfgNode
from pytorch_lightning.callbacks import Callback
from ymir_exc import monitor
from ymir_exc.util import get_bool, get_weight_files


# TODO save and load config file for ymir
def get_config_file(ymir_cfg: edict) -> str:
    root_dir = ymir_cfg.ymir.input.models_dir
    config_files = glob.glob(osp.join(root_dir, '**', 'train_cfg.yml'), recursive=True)
    if config_files:
        return config_files[0]
    else:
        return ""


def get_best_weight_file(ymir_cfg: edict) -> str:
    """
    if ymir offer pretrained weights file, use it.
    else find suitable coco pretrained weight file
    """
    # get ymir pretrained weights
    weight_files = get_weight_files(ymir_cfg, suffix=('.pth', '.ckpt'))

    if len(weight_files) == 0:
        assert ymir_cfg.ymir.run_training, 'only training mode can load pre-train weights'
        # find suitable coco pretrained weight
        coco_pretrained_files = [f for f in glob.glob('/weights/**/*', recursive=True) if f.endswith(('.pth', '.ckpt'))]
        model_name = osp.splitext(osp.basename(ymir_cfg.param.config_file))[0]

        suitable_weight_files = [f for f in coco_pretrained_files if osp.basename(f).startswith(model_name)]
        if len(suitable_weight_files) > 0:
            logging.info(f'use coco pretrained weight file {suitable_weight_files[0]}')
            return suitable_weight_files[0]
        else:
            return ""
    else:
        # choose weight file by priority, best > newest > others
        best_weight_files = [f for f in weight_files if osp.basename(f).find('best') > -1]
        if best_weight_files:
            return max(best_weight_files, key=osp.getctime)

        return max(weight_files, key=osp.getctime)


def get_converted_dataset_info(cfg: edict):
    """
    avoid DDP write
    """
    info = dict()
    for split in ['train', 'val']:
        split_json_file = osp.join(cfg.ymir.output.root_dir, 'ymir_dataset', f'ymir_{split}.json')
        info[split] = dict(img_dir=cfg.ymir.input.assets_dir, ann_file=split_json_file)
    return info


def modify_config(cfg: CfgNode, ymir_cfg: edict):
    """
    modify nanodet yacs cfgnode
    change save_dir, class_names, class_number and dataset directory
    """
    cfg.defrost()
    cfg.save_dir = ymir_cfg.ymir.output.models_dir
    cfg.model.arch.head.num_classes = len(ymir_cfg.param.class_names)
    cfg.model.arch.aux_head.num_classes = len(ymir_cfg.param.class_names)
    gpu_id: str = ymir_cfg.param.get('gpu_id', '0')
    gpu_count: int = ymir_cfg.param.get('gpu_count', 1)
    if gpu_count > 0:
        cfg.device.gpu_ids = [int(x) for x in gpu_id.split(',')]
    else:
        cfg.device.gpu_ids = -1  # use cpu to train

    cfg.class_names = ymir_cfg.param.class_names

    if ymir_cfg.ymir.run_training:
        ymir_dataset_info = get_converted_dataset_info(ymir_cfg)

        cfg.data.train.name = 'CocoDataset'
        cfg.data.train.img_path = ymir_dataset_info['train']['img_dir']
        cfg.data.train.ann_path = ymir_dataset_info['train']['ann_file']
        cfg.data.val.name = 'CocoDataset'
        cfg.data.val.img_path = ymir_dataset_info['val']['img_dir']
        cfg.data.val.ann_path = ymir_dataset_info['val']['ann_file']

        input_size: int = int(ymir_cfg.param.get('input_size', -1))
        if input_size > 0:
            cfg.data.train.input_size = [input_size, input_size]
            cfg.data.val.input_size = [input_size, input_size]

        resume: bool = get_bool(ymir_cfg, 'resume', False)
        cfg.schedule.resume = resume

        load_from: str = ymir_cfg.param.get('load_from', '')
        if load_from:
            cfg.schedule.load_from = load_from

        # auto load pretrained weight if not set by user
        if not resume and not load_from:
            best_weight_file = get_best_weight_file(ymir_cfg)
            cfg.schedule.load_from = best_weight_file

        # TODO if user want workers_per_gpu = -1?
        workers_per_gpu = int(ymir_cfg.param.workers_per_gpu)
        if workers_per_gpu > 0:
            cfg.device.workers_per_gpu = workers_per_gpu

        batch_size_per_gpu = int(ymir_cfg.param.batch_size_per_gpu)

        # change batch size and gpu_ids for small dataset
        with open(ymir_cfg.ymir.input.training_index_file, 'r') as fp:
            lines = fp.readlines()
        train_dataset_size = len(lines)
        if train_dataset_size < batch_size_per_gpu:
            batch_size_per_gpu = max(2, train_dataset_size)
            cfg.device.gpu_ids = [0]
        elif train_dataset_size < batch_size_per_gpu * gpu_count:
            cfg.device.gpu_ids = [0]

        if batch_size_per_gpu > 0:
            cfg.device.batchsize_per_gpu = batch_size_per_gpu

        learning_rate = ymir_cfg.param.learning_rate
        if learning_rate > 0:
            cfg.schedule.optimizer.lr = learning_rate

        epochs = ymir_cfg.param.epochs
        if epochs > 0:
            cfg.schedule.total_epochs = epochs
            cfg.schedule.val_intervals = max(1, epochs // 10)

    cfg.freeze()


class YmirMonitorCallback(Callback):
    """
    write training process for ymir monitor
    """

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch ends.
        To access all batch outputs at the end of the epoch, either:
        1. Implement `training_epoch_end` in the `LightningModule` and access outputs via the module OR
        2. Cache data across train batch hooks inside the callback implementation to post-process in this hook.
        """
        max_epochs = trainer.max_epochs
        current_epoch = trainer.current_epoch

        monitor_gap = max(1, max_epochs // 100)
        if current_epoch % monitor_gap == 0 and trainer.global_rank in [0, -1]:
            monitor.write_monitor_logger(percent=current_epoch / max_epochs)

    # TODO batch-level monitor logger
    # def on_train_batch_start(self,
    #                          trainer: "pl.Trainer",
    #                          pl_module: "pl.LightningModule",
    #                          batch: Any,
    #                          batch_idx: int) -> None:
    #     """Called when the train batch begins."""
    #     batch_per_epoch = trainer.num_training_batches
    #     if trainer.num_training_batches == float("inf"):
    #         batch_per_epoch = 100

    #     if trainer.current_epoch == 0 and trainer.global_rank in [0, -1] and batch_idx < 10:
    #         monitor.write_monitor_logger(percent=batch_idx / batch_per_epoch / trainer.max_epochs)


class NanodetYmirDataset(td.Dataset):

    def __init__(self, images: List[str], cfg: CfgNode):
        super().__init__()
        self.images = images
        self.input_size = cfg.data.val.input_size
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def __getitem__(self, index):
        img_info = {"id": index}
        image_path = self.images[index]
        img_info["file_name"] = image_path
        img = cv2.imread(image_path)
        if img is None:
            print("image {} read failed.".format(image_path))
            raise FileNotFoundError("Cant load image! Please check image path!")

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, img=img)
        meta = self.pipeline(None, meta, self.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))
        return meta

    def __len__(self):
        return len(self.images)
