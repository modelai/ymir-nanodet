import os
import sys

import cv2
import torch
import torch.distributed as dist
import torch.utils.data as td
from easydict import EasyDict as edict
from tqdm import tqdm
from ymir_exc import monitor
from ymir_exc import result_writer as rw
from ymir_exc.util import YmirStage, get_bool, get_merged_config, get_ymir_process

from demo.demo import Predictor
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.util import Logger, cfg, load_config
from ymir.mining_base import ALDDMining, binary_classification_entropy, multiple_classification_entropy
from ymir.utils import NanodetYmirDataset, get_best_weight_file, get_config_file

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


class NanodetALDDMining(ALDDMining):
    """
    Active Learning for Deep Detection Neural Networks (ICCV 2019)
    official code: https://gitlab.com/haghdam/deep_active_learning

    modify for nanodet:
    1. no class weight support
    2. multiple class support
    3. changed hyper-parameter
    """
    def __init__(self, ymir_cfg: edict, task='mining'):
        self.ymir_cfg = ymir_cfg
        if ymir_cfg.ymir.run_mining and ymir_cfg.ymir.run_infer:
            # multiple task, run mining first, infer later
            if task == 'infer':
                self.task_idx = 1
            elif task == 'mining':
                self.task_idx = 0
            else:
                raise Exception(f'unknown task {task}')

            self.task_num = 2
        else:
            self.task_idx = 0
            self.task_num = 1

        gpu_id: str = str(ymir_cfg.param.get('gpu_id', '0'))
        gpu_count: int = len(gpu_id.split(','))

        if gpu_count == 1:
            device = f"cuda:{gpu_id}"
        elif gpu_count > 1:
            gpu_id_rank = gpu_id.split(',')[RANK]
            device = f"cuda:{gpu_id_rank}"
        else:
            device = 'cpu'

        weight_file = get_best_weight_file(ymir_cfg)
        config_file = get_config_file(ymir_cfg)

        if not config_file:
            config_file = ymir_cfg.param.get('config_file', "")
            if not config_file:
                raise Exception('no config file defined or found!')

        load_config(cfg, config_file)

        logger = Logger(LOCAL_RANK, use_tensorboard=False)
        self.predictor = Predictor(cfg, weight_file, logger, device=device)
        self.num_classes = len(ymir_cfg.param.class_names)
        self.resize_shape = [x // 8 for x in cfg.data.val.input_size]
        self.max_pool_size = 13
        self.avg_pool_size = 9
        self.align_corners = False
        self.cfg = cfg
        self.batch_size_per_gpu = int(ymir_cfg.param.get('batch_size_per_gpu', 16))
        self.num_workers_per_gpu = int(ymir_cfg.param.get('num_workers_per_gpu', 4))
        self.pin_memory = get_bool(ymir_cfg, 'pin_memory', False)
        self.device = device
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def extract_feats(self, img):
        """
        single image interface
        return class_scores: B,C,N
            - B: batch size
            - C: class number
            - N: point number
        """
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            # eg: B=2, C=1+4*(7+1)=33, H, W in {52, 26, 13, 7}
            # batch size ,num_classes + 4*(reg_max+1), H, W
            # [[B,C,52,52], [B,C,26,26], [B,C,13,13], [B,C,7,7]]
            outputs = self.predictor.model.extract_feats(meta['img'])

        conf_maps = [x[:, :self.num_classes, :, :] for x in outputs]
        return conf_maps

    def get_entropy(self, feature_map: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 1:
            # binary cross entropy
            return binary_classification_entropy(feature_map)
        else:
            # multi-class cross entropy
            return multiple_classification_entropy(feature_map, activation='sigmoid')

    def write_monitor_logger(self, stage: YmirStage, p: float):
        monitor.write_monitor_logger(
            percent=get_ymir_process(stage=stage, p=p, task_idx=self.task_idx, task_num=self.task_num))


def main() -> int:
    ymir_cfg = get_merged_config()
    miner = NanodetALDDMining(ymir_cfg)

    with open(ymir_cfg.ymir.input.candidate_index_file, 'r') as f:
        images = [line.strip() for line in f.readlines()]

    # origin dataset
    if RANK != -1:
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
        images_rank = images[RANK::WORLD_SIZE]
    else:
        images_rank = images
    origin_dataset = NanodetYmirDataset(images_rank, miner.cfg)
    origin_dataset_loader = td.DataLoader(origin_dataset,
                                          batch_size=miner.batch_size_per_gpu,
                                          shuffle=False,
                                          sampler=None,
                                          num_workers=miner.num_workers_per_gpu,
                                          pin_memory=miner.pin_memory,
                                          drop_last=False)

    mining_results = dict()
    rank_dataset_size = len(images_rank)
    pbar = tqdm(origin_dataset_loader) if RANK in [0, -1] else origin_dataset_loader

    for idx, batch in enumerate(pbar):
        # batch-level sync, avoid 30min time-out error
        if LOCAL_RANK != -1:
            dist.barrier()
            
        with torch.no_grad():
            outputs = miner.predictor.model.extract_feats(batch['img'].float().to(miner.device))
            scores = miner.mining(outputs)

        for each_imgname, each_score in zip(batch['img_info']["file_name"], scores):
            mining_results[each_imgname] = each_score.item()

        if RANK in [-1, 0]:
            miner.write_monitor_logger(stage=YmirStage.TASK, p=idx * miner.batch_size_per_gpu / rank_dataset_size)

    ymir_mining_result = []
    if WORLD_SIZE == 1:
        for img_file, score in mining_results.items():
            ymir_mining_result.append((img_file, score))
    else:
        torch.save(mining_results, f'/out/mining_results_{RANK}.pt')
        dist.barrier()

        results = []
        for rank in range(WORLD_SIZE):
            results.append(torch.load(f'/out/mining_results_{rank}.pt'))

        for result in results:
            for img_file, score in result.items():
                ymir_mining_result.append((img_file, score))

    assert len(ymir_mining_result) == len(
        images), f'gather methods failed, gather {len(ymir_mining_result)}, expected {len(images)}'
    if RANK in [0, -1]:
        rw.write_mining_result(mining_result=ymir_mining_result)

    if LOCAL_RANK != -1:
        print(f'rank: {RANK}, start destroy process group')
        dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    sys.exit(main())
