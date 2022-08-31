import sys

from ymir_exc.executor import Executor


def main():
    apps = dict(training='python3 ymir/ymir_training.py',
                mining='python3 ymir/ymir_mining.py',
                infer='python3 ymir/ymir_infer.py')
    executor = Executor(apps=apps)
    executor.start()
    return 0


if __name__ == '__main__':
    sys.exit(main())
