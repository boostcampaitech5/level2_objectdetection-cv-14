# dataset settings
dataset_type = 'CocoDataset'
data_root='/opt/ml/dataset/'
json_root = '/opt/ml/dataset/stratified_v2/'

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)



# 추가하고 싶은 목록

# - CLAHE
# - channel dropout, channel shuffle
# - Random shadow
# - solarize
# - superpixel
# (CutOut, MinIoURandomCrop, MixUp, Mosaic,
#  PhotoMetricDistortion, RandomAffine, RandomShift )

augmentation_transforms = [
    # Add your augmentation transforms here
    dict(type='ChannelDropout', p=1),
    dict(type='ChannelShuffle', p=1),
    dict(type='CLAHE', p=1),
    dict(type='RandomShadow', p=1),
    dict(type='RandomBrightnessContrast', brightness_limit=0.1, contrast_limit=0.15, p=1),
    dict(type='HueSaturationValue', hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=1),
    dict(type='GaussNoise', p=1),
    dict(type='Blur', p=1),
    # dict(type='GaussianBlur', p=1),
    # dict(type='MedianBlur', blur_limit=5, p=1),
    # dict(type='MotionBlur', p=1),
]

import albumentations as A
import os
import cv2
import copy
import glob

def generate_augmented_images(folder_path, save_dir, augmentation_transforms):
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    
    for image_path in image_files:
        image = cv2.imread(image_path)
        filename = os.path.basename(image_path)
        filename, ext = os.path.splitext(filename)
        temp_transform=copy.deepcopy(augmentation_transforms)

        for i, transform_dict in enumerate(temp_transform):
            if 'type' not in transform_dict:
                continue  # 'type' 키가 없는 경우 다음 iteration으로 넘어갑니다.
            transform_type = transform_dict.pop('type')
            transform = getattr(A, transform_type)(**transform_dict)
            augmented = transform(image=image)
            transformed_image = augmented['image']
            transformed_filename = f'{filename}_{i}{ext}'
            save_path = os.path.join(save_dir, transformed_filename)
            cv2.imwrite(save_path, transformed_image)

            
            
            
folder_path = '../../dataset/images/train/'
save_dir = '../../dataset/images/aug/train/'


generate_augmented_images(folder_path, save_dir, augmentation_transforms)