from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device


import argparse
import os

## argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='nas_base',help='set the name')
args = parser.parse_args()

name = args.name
print(f'The name is {name}')

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
# cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
cfg = Config.fromfile('/opt/ml/baseline/mmdetection/configs/_trash_/nas.py')

# root='../../dataset/' 
root='../../dataset/'

# dataset config 수정
cfg.data.train.classes = classes
cfg.data.train.img_prefix = root
cfg.data.train.ann_file = root + 'stratified_v2/train_fold_1.json' # train json 정보
cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize

cfg.data.val.ann_file = root + 'stratified_v2/val_fold_1.json' # valid json 정보
cfg.data.val.img_prefix = root
cfg.data.val.classes = classes

cfg.data.test.classes = classes
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = root + 'test.json' # test json 정보
cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

cfg.data.samples_per_gpu = 16

cfg.seed = 2022
cfg.gpu_ids = [0]

if not os.path.exists(f'./work_dirs/{name}'):
    os.mkdir(f'./work_dirs/{name}')
cfg.work_dir = f'./work_dirs/{name}'

# cfg.model.roi_head.bbox_head.num_classes = 10


cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
cfg.device = get_device()

datasets = [build_dataset(cfg.data.train)]

print(datasets[0])

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(project='level2-objectdetection', name=f'{name}', entity='hi-ai')
            )])
# for wandb
cfg.log_config = log_config

model = build_detector(cfg.model)
model.init_weights()

# 모델 학습
train_detector(model, datasets[0], cfg, distributed=False, validate=True)

# def train_detector(model,
#                    dataset,
#                    cfg,
#                    distributed=False,
#                    validate=False,
#                    timestamp=None,
#                    meta=None):