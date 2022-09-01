import os
import sys
from typing import List

import torch
from demo.demo import Predictor
from tqdm import tqdm
from ymir.utils import get_best_weight_file, get_config_file, modify_config
from ymir_exc import monitor
from ymir_exc import result_writer as rw
from ymir_exc.util import get_merged_config

from nanodet.util import Logger, cfg, load_config

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def main() -> int:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    ymir_cfg = get_merged_config()
    gpu_id: str = ymir_cfg.param.get('gpu_id', '0')
    gpu_count: int = len(gpu_id.split(','))
    conf_threshold: float = float(ymir_cfg.param.get('conf_thres', '0.2'))
    class_names: List[str] = ymir_cfg.param.get('class_names')
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
        image_files = [line.strip() for line in f.readlines()]

    if RANK != -1:
        image_files = image_files[RANK::WORLD_SIZE]

    total_size = len(image_files)
    monitor_gap = max(1, total_size // 100)
    tbar = tqdm(image_files) if RANK in [0, -1] else image_files

    ymir_infer_result = dict()
    for idx, image_file in enumerate(tbar):
        if RANK in [-1, 0] and idx % monitor_gap == 0:
            monitor.write_monitor_logger(percent=idx / total_size)

        meta, res = predictor.inference(image_file)
        anns = []
        for class_id in res[0].keys():
            for bbox in res[0][class_id]:
                xmin, ymin, xmax, ymax, conf = bbox
                if conf < conf_threshold:
                    continue

                ann = rw.Annotation(class_name=class_names[class_id],
                                    score=conf,
                                    box=rw.Box(x=int(xmin), y=int(ymin), w=int(xmax - xmin), h=int(ymax - ymin)))
                anns.append(ann)

        ymir_infer_result[image_file] = anns

    if RANK in [0, -1]:
        rw.write_infer_result(infer_result=ymir_infer_result)
    return 0


if __name__ == '__main__':
    sys.exit(main())
