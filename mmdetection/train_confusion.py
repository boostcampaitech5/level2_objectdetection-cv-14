# 모듈 import

import torch
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device

import argparse
import os


def main():
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    
    config_path = args.config_path

    # config file 들고오기
    cfg = Config.fromfile(config_path)

    # root setting
    root='../../dataset/'

    # dataset config 수정
    
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + f'stratified_v2/train_fold_{args.valid_num}.json' # train json 정보
    # cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = root
    cfg.data.val.ann_file = root + f'stratified_v2/val_fold_{args.valid_num}.json' # valid json 정보

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json' # test json 정보
    # cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
    
    cfg.train_pipeline = cfg.train_pipeline
    cfg.val_pipeline = cfg.test_pipeline
    cfg.test_pipeline = cfg.test_pipeline

    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.val.pipeline = cfg.val_pipeline
    cfg.data.test.pipeline = cfg.test_pipeline

    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = args.batch_size # batch size

    cfg.seed = args.seed
    cfg.gpu_ids = [0]
    work_dir = f'/opt/ml/baseline/mmdetection/work_dirs/{name}'
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    cfg.work_dir = work_dir

    # model num_classes 설정
    # for i in range(args.bbox_head):
    #     cfg.model.roi_head.bbox_head[i].num_classes = 10

    #cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.optimizer_config.grad_clip = None
    cfg.optimizer.lr=args.lr
    #cfg.lr_config.step=[args.lr_step]
    
    print(cfg.optimizer)
    print(cfg.optimizer_config)
    
    cfg.evaluation.classwise = True
    #cfg.evaluation = dict(interval=1, start=1,save_best='auto')
    #cfg.evaluation = dict(interval=1, save_best='bbox_mAP_50', metric='bbox')
    #cfg.evaluation = dict(interval=1, save_best='bbox_mAP_50', metric='mAP')
    # cfg.evaluation['save_best'] = 'bbox_mAP_50' 
    #cfg.evaluation.metric = 'mAP'
    
    cfg.checkpoint_config = dict(max_keep_ckpts=1, interval=1)
    
    log_config = dict(
        _delete=True,
        interval=50,
        hooks=[
            dict(type='TextLoggerHook'),
            dict(
                type='WandbLoggerHook',
                init_kwargs=dict(project='Model', name=f'{name}', entity='hi-ai'),
                interval=50,
                #log_checkpoint=True,
                #log_checkpoint_metadata=True,
                #num_eval_images=100
                )])
    # for wandb
    cfg.log_config = log_config
    
    # cfg.log_config = dict(
    #     _delete=True,
    #     interval=50,
    #     hooks=[
    #         dict(type='TextLoggerHook'),
    #         dict(type='WandbLoggerHook',
    #              init_kwargs=dict(project='Model', name=f'{name}', entity='hi-ai'),
    #              interval=50,
    #              log_checkpoint=True,
    #              log_checkpoint_metadata=True,
    #              num_eval_images=100
    #              )])
    
    
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=args.epoch)
    cfg.device = get_device()


    datasets = [build_dataset(cfg.data.train)]

    print(datasets[0])

    model = build_detector(cfg.model)
    model.init_weights()

    # 모델 학습
    train_detector(model, datasets[0], cfg, distributed=False, validate=True)

## argparse
if __name__ == '__main__':
    from main_file import args
    print(args)
    name = args.name
    print(f'The name is {name}')
    main()


