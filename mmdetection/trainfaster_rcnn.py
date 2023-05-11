from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
# cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
# cfg = Config.fromfile('/opt/ml/baseline/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
cfg = Config.fromfile('/opt/ml/baseline/mmdetection/configs/_trash_/faster.py')
"""
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
"""
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
cfg.work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash'

cfg.model.roi_head.bbox_head.num_classes = 10

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
cfg.device = get_device()

datasets = [build_dataset(cfg.data.train)]

print(datasets[0])
# TODO: wandb
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(project='level2-objectdetection', name='kfold5_faster_rcnn_r50_fpn_1x_coco', entity='hi-ai')
            )])
# for wandb
cfg.log_config = log_config

model = build_detector(cfg.model)
model.init_weights()

# 모델 학습
train_detector(model, datasets[0], cfg, distributed=False, validate=True)

# ## make confusion matrix

# cfg.model.train_cfg = None

# dataset = build_dataset(cfg.data.test)
# data_loader = build_dataloader(
#         dataset,
#         samples_per_gpu=1,
#         workers_per_gpu=cfg.data.workers_per_gpu,
#         dist=False,
#         shuffle=False)

# ##################################################################
# # confusion matrix################################################
# ##################################################################

# from mmdet.apis import inference_detector, show_result
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import json
# import cv2
# import torch

# # validation set으로 confusion matrix 만들기
# cfg.data.val.ann_file = root + 'stratified_v2/val_fold_1.json' # valid json 정보
# cfg.data.val.img_prefix = root
# cfg.data.val.classes = classes

# model.eval()
# outputs = []
# gt = []
# for i, data in enumerate(data_loader):
#     with torch.no_grad():
#         result = model(return_loss=False, rescale=True, **data)
#     outputs.extend(result)
#     gt.extend(data['img_metas'][0].data[0]['gt_labels'].cpu().numpy())

# # confusion matrix


# confusion_matrix = np.zeros((10,10))
# for i in range(len(outputs)):
#     for j in range(len(outputs[i])):
#         confusion_matrix[gt[i]][outputs[i][j]] += 1

# # confusion matrix 시각화
# plt.figure(figsize=(10,10))
# sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
# plt.xlabel('Predicted Class')
# plt.ylabel('True Class')
# plt.show()

