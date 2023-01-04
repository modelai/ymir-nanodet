import sys

from ymir_exc.executor import Executor
from ymir_exc.util import find_free_port, get_merged_config


def main():
    ymir_cfg = get_merged_config()
    gpu_id: str = ymir_cfg.param.get('gpu_id', '0')
    gpu_count: int = len(gpu_id.split(','))
    port: int = find_free_port()

    torchrun_cmd = f'torchrun --standalone --nnodes 1 --nproc_per_node {gpu_count} --master_port {port}'
    apps = dict(training='python3 ymir/ymir_training.py',
                mining=f'{torchrun_cmd} ymir/ymir_mining.py',
                infer=f'{torchrun_cmd} ymir/ymir_infer.py')
    executor = Executor(apps=apps)
    executor.start()
    return 0


if __name__ == '__main__':
    sys.exit(main())
