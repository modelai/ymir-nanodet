from ymir_exc.util import get_merged_config
from ymir_exc.util import convert_ymir_to_coco
from nanodet.util.yacs import CfgNode


def modify_config(cfg: CfgNode):
    """
    modify nanodet yacs cfgnode
    change save_dir, class_names, class_number and dataset directory
    """
    ymir_cfg = get_merged_config()
    ymir_dataset_info = convert_ymir_to_coco(cat_id_from_zero=False)
    
    cfg.defrost()
    cfg.save_dir = ymir_cfg.ymir.output.models_dir
    cfg.model.arch.head.num_classes = len(ymir_cfg.param.class_names)
    cfg.model.arch.aux_head.num_classes = len(ymir_cfg.param.class_names)

    cfg.data.train.img_path = ymir_cfg.ymir.input.assets_dir
    cfg.data.train.ann_path =
    cfg.freeze()
