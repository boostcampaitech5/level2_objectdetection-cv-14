_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    #'../_base_/datasets/coco_detection.py',
    '../_base_/datasets/coco_confusion_matrix.py',
    '../_base_/schedules/schedule_CosineAnnealing_fast.py', '../_base_/default_runtime.py'
]

# optimizer
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]))
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
#base_batch_size = (8 GPUs) x (2 samples per GPU)
#auto_scale_lr = dict(base_batch_size=16)

# optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW', 
#                  lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
#                  paramwise_cfg={'decay_rate': 0.8,
#                                 'decay_type': 'layer_wise',
#                                 'num_layers': 12})
