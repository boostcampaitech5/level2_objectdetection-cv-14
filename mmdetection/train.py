# 모듈 import
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
#cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
cfg=Config.fromfile('./configs/yolof/yolof_r50_c5_8x8_1x_coco.py')
root='../../dataset/'

# dataset config 수정
cfg.data.train.classes = classes
cfg.data.train.img_prefix = root
#stratified kfold
cfg.data.train.ann_file=root+'stratified_v2/train_fold_1.json'
#cfg.data.train.ann_file = root + 'train.json' # train json 정보
cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize



cfg.data.test.classes = classes
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = root + 'test.json' # test json 정보
cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
#stratified k fold
cfg.data.val.classes = classes
cfg.data.val.img_prefix=root
cfg.data.val.ann_file=root+'stratified_v2/val_fold_1.json'
print(cfg.data.val.pipeline)
cfg.data.val.pipeline[1]['img_scale'] = (512,512) # Resize

cfg.data.samples_per_gpu = 4

cfg.seed = 2022
cfg.gpu_ids = [0]
#cfg.work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash'
cfg.work_dir='./work_dirs/retinanet_effb3_fpn_crop896_8x4_1x_coco'

#cfg.model.roi_heads.bbox_head.num_classes = 10
cfg.model.bbox_head.num_classes = 10
cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
cfg.device = get_device()

# build_dataset
datasets = [build_dataset(cfg.data.train)]
#valid까지 확인
#datasets=[build_dataset(cfg.data.train),build_dataset(cfg.data.val)]
# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)
#model.init_weights()
# #wandb테스트 여기서 이렇게 코드를 쳐서 실행하나 ./configs/_base_/default_runtime.py에서 수정하나 똑같은듯?
# import wandb
# wandb.init(project="level2-objectdetection", entity="hi-ai")
# cfg.log_config.hooks=[
#     dict(type='TextLoggerHook'),
#     dict(type='MMDetWandbHook',
#          init_kwargs={'project':'level2-objectdetection'},
#          interval=10,
#          log_checkpoint=True,
#          log_checkpoint_metadata=True,
#          num_eval_images=100)]

import wandb
wandb.init(project="level2-objectdetection", entity="hi-ai")

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(project='level2-objectdetection', name='yolof', entity='hi-ai')
            )])
# for wandb
cfg.log_config = log_config

# import wandb
# wandb.init(project="level2-objectdetection", entity="hi-ai")


# 모델 학습

train_detector(model, datasets, cfg, distributed=False, validate=True)

preds = []
gts = []

# 테스트 데이터의 예측 결과를 추출합니다.
for i, data in enumerate(data_loader):
    with torch.no_grad():
        result = single_gpu_test(model, data["img"][0].cuda(), show=False)
    preds.extend(result[0])
    gts.extend(data["gt_labels"][0].tolist())

# confusion matrix와 accuracy를 계산합니다.
cm = eval_map(preds, gts, iou_thr=0.5, print_summary=False, dataset=dataset.CLASSES)
acc = sum([1 for i, j in zip(preds, gts) if i == j]) / len(preds)

# wandb에 confusion matrix와 accuracy를 로깅합니다.
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(preds, gts, class_names=dataset.CLASSES),
           "accuracy": acc})