import glob
import os.path as osp

import pytorch_lightning as pl
from easydict import EasyDict as edict
from pytorch_lightning.callbacks import Callback
from ymir_exc import monitor
from ymir_exc.util import convert_ymir_to_coco, get_weight_files

from nanodet.util.yacs import CfgNode


# TODO save and load config file for ymir
def get_config_file(ymir_cfg: edict) -> str:
    root_dir = ymir_cfg.ymir.input.models_dir
    config_files = glob.glob(osp.join(root_dir, '**', 'train_cfg.yml'), recursive=True)
    if config_files:
        return config_files[0]
    else:
        return ""


def get_best_weight_file(ymir_cfg: edict) -> str:
    weight_files = get_weight_files(ymir_cfg, suffix=('.pth', '.ckpt'))

    if len(weight_files) == 0:
        return ""
    else:
        # choose weight file by priority, best > newest > others
        best_weight_files = [f for f in weight_files if osp.basename(f).find('best') > -1]
        if best_weight_files:
            return max(best_weight_files, key=osp.getctime)

        return max(weight_files, key=osp.getctime)


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
    gpu_ids = [int(x) for x in gpu_id.split(',')]
    cfg.device.gpu_ids = gpu_ids
    cfg.class_names = ymir_cfg.param.class_names

    if ymir_cfg.ymir.run_training:
        cfg.data.train.name = 'CocoDataset'
        ymir_dataset_info = convert_ymir_to_coco(cat_id_from_zero=False)
        cfg.data.train.img_path = ymir_dataset_info['train']['img_dir']
        cfg.data.train.ann_path = ymir_dataset_info['train']['ann_file']
        cfg.data.val.name = 'CocoDataset'
        cfg.data.val.img_path = ymir_dataset_info['val']['img_dir']
        cfg.data.val.ann_path = ymir_dataset_info['val']['ann_file']

        # TODO if user want workers_per_gpu = -1?
        workers_per_gpu = int(ymir_cfg.param.workers_per_gpu)
        if workers_per_gpu > 0:
            cfg.device.workers_per_gpu = workers_per_gpu

        batch_size_per_gpu = int(ymir_cfg.param.batch_size_per_gpu)
        if batch_size_per_gpu > 0:
            cfg.device.batchsize_per_gpu = batch_size_per_gpu

        learning_rate = ymir_cfg.param.learning_rate
        if learning_rate > 0:
            cfg.schedule.optimizer.lr = learning_rate

        epochs = ymir_cfg.param.epochs
        if epochs > 0:
            cfg.schedule.total_epochs = epochs

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
