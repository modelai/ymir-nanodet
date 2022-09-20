import os
import sys
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.utils.data as td
from tqdm import tqdm
from ymir_exc import monitor
from ymir_exc import result_writer as rw
from ymir_exc.util import get_bool, get_merged_config

from demo.demo import Predictor
from nanodet.util import Logger, cfg, load_config
from ymir.utils import NanodetYmirDataset, get_best_weight_file, get_config_file, modify_config

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def main() -> int:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    ymir_cfg = get_merged_config()
    gpu_id: str = str(ymir_cfg.param.get('gpu_id', '0'))
    gpu_count: int = len(gpu_id.split(','))
    conf_threshold: float = float(ymir_cfg.param.get('conf_thres', '0.2'))
    class_names: List[str] = ymir_cfg.param.get('class_names')
    batch_size_per_gpu: int = int(ymir_cfg.param.get('batch_size_per_gpu', 16))
    num_workers_per_gpu: int = int(ymir_cfg.param.get('num_workers_per_gpu', 4))
    pin_memory = get_bool(ymir_cfg, 'pin_memory', False)
    weight_file = get_best_weight_file(ymir_cfg)

    # find config file from ymir model first, then use user defined config file
    config_file = get_config_file(ymir_cfg)
    if not config_file:
        config_file = ymir_cfg.param.get('config_file', "")
        if not config_file:
            raise Exception('no config file defined or found!')

    load_config(cfg, config_file)
    modify_config(cfg, ymir_cfg)

    logger = Logger(LOCAL_RANK, use_tensorboard=False)
    if gpu_count == 1:
        device = f"cuda:{gpu_id}"
    elif gpu_count > 1:
        gpu_id_rank = gpu_id.split(',')[RANK]
        device = f"cuda:{gpu_id_rank}"
    else:
        device = 'cpu'

    predictor = Predictor(cfg, weight_file, logger, device=device)

    with open(ymir_cfg.ymir.input.candidate_index_file, 'r') as f:
        images = [line.strip() for line in f.readlines()]

    if RANK != -1:
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
        images_rank = images[RANK::WORLD_SIZE]
    else:
        images_rank = images

    origin_dataset = NanodetYmirDataset(images_rank, cfg)
    origin_dataset_loader = td.DataLoader(origin_dataset,
                                          batch_size=batch_size_per_gpu,
                                          shuffle=False,
                                          sampler=None,
                                          num_workers=num_workers_per_gpu,
                                          pin_memory=pin_memory,
                                          drop_last=False)

    rank_dataset_size = len(images_rank)
    monitor_gap = max(1, rank_dataset_size // 100)
    # tbar = tqdm(images_rank) if RANK in [0, -1] else images_rank
    tbar = tqdm(origin_dataset_loader) if RANK in [0, -1] else origin_dataset_loader

    rank_infer_result = dict()
    for idx, batch in enumerate(tbar):
        if RANK in [-1, 0] and idx % monitor_gap == 0:
            monitor.write_monitor_logger(percent=idx * batch_size_per_gpu / rank_dataset_size)

        with torch.no_grad():
            batch['img'] = batch['img'].float().to(device)
            results = predictor.model.inference(batch)

        anns = []

        id_name_map = dict()
        for img_id, file_name in zip(batch['img_info']['id'], batch['img_info']['file_name']):
            id_name_map[int(img_id)] = file_name

        for img_id, det_result in results.items():
            for class_id in det_result.keys():
                for bbox in det_result[class_id]:
                    xmin, ymin, xmax, ymax, conf = bbox
                    if conf < conf_threshold:
                        continue

                    ann = rw.Annotation(class_name=class_names[class_id],
                                        score=conf,
                                        box=rw.Box(x=int(xmin), y=int(ymin), w=int(xmax - xmin), h=int(ymax - ymin)))
                    anns.append(ann)

            # assume img_id is index
            image_file = id_name_map[int(img_id)]
            rank_infer_result[image_file] = anns

    all_infer_result: Dict[str, list] = {}
    if WORLD_SIZE == 1:
        all_infer_result = rank_infer_result
    else:
        torch.save(rank_infer_result, f'/out/infer_results_{RANK}.pt')
        dist.barrier()

        for rank in range(WORLD_SIZE):
            all_infer_result.update(torch.load(f'/out/infer_results_{rank}.pt'))

    if RANK in [0, -1]:
        rw.write_infer_result(infer_result=all_infer_result)

    if LOCAL_RANK != -1:
        print(f'rank: {RANK}, start destroy process group')
        dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    sys.exit(main())
