# ymir-nanodet

## entrypoint
- `ymir/start.py`

## training
- `ymir/ymir_training.py`
- [x] hyper-parameter custom
- [x] multi-gpu training
- [x] tensorboard
- [x] map and weight file save
- [x] monitor process
- [x] finetune/resume/transfer learning
- [x] local-pretrained weight
- [ ] real-time map and weight file save

### auto load_from/resume
```
resume: bool = get_bool(ymir_cfg, 'resume', False)
cfg.schedule.resume = resume

load_from: str = ymir_cfg.param.get('load_from', '')
if load_from:
    cfg.schedule.load_from = load_from

# auto load pretrained weight if not set by user
if not resume and not load_from:
    best_weight_file = get_best_weight_file(ymir_cfg)
    cfg.schedule.load_from = best_weight_file
```

## infer
- `ymir/ymir_infer.py`
- [x] monitor process
- [x] nms
- [x] batch infer
- [x] multi-gpu infer

## mining
- [x] monitor process
- [x] batch mining
- [x] multi-gpu mining

## bug
- [x] when the `epochs` is small and dataset is small.
```
FileNotFoundError: [Errno 2] No such file or directory: '/out/models/model_best/eval_results.txt'
```

## changelog

- 2022/09/09: support `nanodet-plus` coco pretrained weights
- 2022/09/20: support `aldd mining`
