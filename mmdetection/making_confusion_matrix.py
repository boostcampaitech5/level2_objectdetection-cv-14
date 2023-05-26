_base_ = [
    '../_base_/models/cascade_mask_rcnn_convnext_fpn.py',
    '../_base_/datasets/coco_confusion_matrix.py',
    '../_base_/schedules/schedule_CosineAnnealing_fast.py', '../_base_/default_runtime.py'
]
