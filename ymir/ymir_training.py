import glob
import logging
import os.path as osp
import shutil
import subprocess
import sys

from ymir_exc.util import find_free_port, get_merged_config, write_ymir_training_result


def main() -> int:
    ymir_cfg = get_merged_config()
    config_file = ymir_cfg.param.config_file
    # gpu_id = ymir_cfg.param.get('gpu_id')
    gpu_count = int(ymir_cfg.param.get('gpu_count'))
    # export_format = ymir_cfg.param.get('export_format', 'ark:raw')
    seed = int(ymir_cfg.param.get('seed', 2345))

    commands = ['python3']
    if gpu_count > 1:
        port = find_free_port()
        commands.extend(f'-m torch.distributed.launch --nproc_per_node {gpu_count} --master_port {port}'.split())

    commands.extend(f'tools/train.py {config_file} --seed {seed}'.split())

    logging.info(f'start training: {commands}')
    subprocess.run(commands, check=True)

    logging.info('training finished')

    eval_txt_file = osp.join(ymir_cfg.ymir.output.models_dir, 'model_best', 'eval_results.txt')
    with open(eval_txt_file, 'r') as fp:
        lines = fp.readlines()

    best_map50: float = 0
    # epoch_info = dict()
    for line in lines:
        key, value = line.strip().split(':')
        # if key == 'Epoch':
        #     epoch_info = dict(epoch=int(value))
        # else:
        #     epoch_info[key] = float(value)
        if key == 'AP_50':
            map50 = float(value)
            if map50 > best_map50:
                best_map50 = map50

    pth_files = glob.glob(osp.join(ymir_cfg.ymir.output.models_dir, '**', '*.pth'), recursive=True)
    ckpt_files = glob.glob(osp.join(ymir_cfg.ymir.output.models_dir, '**', '*.ckpt'), recursive=True)
    cfg_files = glob.glob(osp.join(ymir_cfg.ymir.output.tensorboard_dir, '**', 'train_cfg.yml'), recursive=True)

    src_files = pth_files + ckpt_files + cfg_files
    dst_files = [osp.join(ymir_cfg.ymir.output.models_dir, osp.basename(f)) for f in src_files]

    for src_file, dst_file in zip(src_files, dst_files):
        if osp.exists(dst_file):
            continue
        shutil.copy(src_file, dst_file)

    write_ymir_training_result(ymir_cfg, best_map50, dst_files, id='nanodet')
    return 0


if __name__ == '__main__':
    sys.exit(main())
