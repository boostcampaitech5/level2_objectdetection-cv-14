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

    # config_path = '/opt/ml/baseline/mmdetection/configs/_trash_/cascade_base.py'
    config_path = args.config_path

    # config file 들고오기
    cfg = Config.fromfile(config_path)
    # /opt/ml/baseline/mmdetection/configs/nas_fpn/retinanet_r50_fpn_crop640_50e_coco.py

    # root setting
    root='../../dataset/'

    # dataset config 수정
    
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + f'stratified_v3/train_fold_{args.valid_num}.json' # train json 정보
    # cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize

    cfg.data.val.img_prefix = root
    cfg.data.val.classes = classes
    cfg.data.val.ann_file = root + f'stratified_v3/val_fold_{args.valid_num}.json' # valid json 정보

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json' # test json 정보
    # cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

    cfg.data.samples_per_gpu = args.batch_size # batch size

    cfg.seed = args.seed
    cfg.gpu_ids = [0]
    work_dir = f'/opt/ml/baseline/mmdetection/work_dirs/{name}'
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    cfg.work_dir = work_dir

    # model num_classes 설정
    for i in range(args.bbox_head):
        cfg.model.roi_head.bbox_head[i].num_classes = 10
        # cfg.model.roi_head.bbox_head[1].num_classes = 10
        # cfg.model.roi_head.bbox_head[2].num_classes = 10
    # roi head가 3개인 이유 : multi-scale featurefusion
    # first roi head : hight level feature  ## more coarse-grained and capture more global context
    # second roi head : middle level feature ## 
    # third roi head : low level feature ## most detailed and precise features.

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=1, interval=1)
    cfg.device = get_device()


    datasets = [build_dataset(cfg.data.train)]

    print(datasets[0])

    # cfg.model_name = cfg.work_dir.split('/')[-2]
    log_config = dict(
        interval=10,
        hooks=[
            dict(type='TextLoggerHook'),
            dict(
                type='WandbLoggerHook',
                init_kwargs=dict(project='Model', name=f'{name}', entity='hi-ai')
                )])
    # for wandb
    cfg.log_config = log_config

    model = build_detector(cfg.model)
    model.init_weights()

    # 모델 학습
    train_detector(model, datasets[0], cfg, distributed=False, validate=True)



"""
python tools/test.py \
/opt/ml/baseline/mmdetection/configs/_trash_/trash.py \
/opt/ml/baseline/mmdetection/work_dirs/swin_base/epoch_14.pth \
--out /opt/ml/baseline/mmdetection/work_dirs/swin_base/test.pkl
"""

## argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument('--config_path', type=str, default='/opt/ml/baseline/mmdetection/configs/_trash_/cascade_base.py', help='path to config file')  
    parser.add_argument('--valid_num', type=int, default=1, help='valid fold number')
    parser.add_argument('--seed', type=int, default=2023, help='random seed (default: 42)')
    # parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train (default: 1)')
    # parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    # parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    # parser.add_argument("--resize", nargs="+", type=list, default=[224, 192], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training (default: 64)')
    # parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    # parser.add_argument('--model', type=str, default='Res50', help='model type (default: BaseModel)')
    # parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    # parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    # parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    # parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    # parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')

    # Container environment
    # parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    # parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--bbox_head', type=str, default=3, help='number of bbox head (default: 3)')
    parser.add_argument('--name', type=str, default='Cascade_RCNN+ResNet50',help='set the name')
    args = parser.parse_args()
    print(args)

    name = args.name
    print(f'The name is {name}')
    
    main()


