# main_file.py

import argparse

parser = argparse.ArgumentParser()

# Data and model checkpoints directories

parser.add_argument('--config_path', type=str, default='/opt/ml/baseline/mmdetection/configs/_custom_/convnext/cascade_mask_rcnn_convnext_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in22k_cosinerestart_albu.py', help='path to config file')  
#parser.add_argument('--valid_num', type=int, default=1, help='valid fold number')
parser.add_argument('--seed', type=int, default=2022, help='random seed (default: 42)')
parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training (default: 4)')
#parser.add_argument('--lr', type=float, default=0.000005, help='input lr for training (default: 0.0001)')
#parser.add_argument('--lr_step', type=int, default=4, help='input lr for training (default: 4)')
parser.add_argument('--epoch', type=int, default=36, help='input epoch for training (default: 12)')

# Container environment
#parser.add_argument('--bbox_head', type=str, default=1, help='number of bbox head (default: 3)')
parser.add_argument('--name', type=str, default='Faster_RCNN+ResNet101',help='set the name')
args = parser.parse_args()

