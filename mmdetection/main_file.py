# main_file.py

import argparse

parser = argparse.ArgumentParser()

# Data and model checkpoints directories
parser.add_argument('--config_path', type=str, default='/opt/ml/baseline/mmdetection/configs/_trash_/dark.py', help='path to config file')  
# parser.add_argument('--config_path', type=str, default='/opt/ml/baseline/mmdetection/configs/htc/htc_x101_64x4d_fpn_16x1_20e_coco.py', help='path to config file')  
parser.add_argument('--valid_num', type=int, default=1, help='valid fold number')
parser.add_argument('--seed', type=int, default=2023, help='random seed (default: 42)')
parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training (default: 64)')

# Container environment
parser.add_argument('--bbox_head', type=str, default=3, help='number of bbox head (default: 3)')
parser.add_argument('--name', type=str, default='Universenet50_2008s',help='set the name')
args = parser.parse_args()

