import os
import shutil

train_dir = './images/train'
valid_dir = './images/valid'
label_dir = './labels/valid'

# validation 폴더가 없는 경우 생성
if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)

# 레이블 파일 읽기
label_files = os.listdir(label_dir)

# 레이블 파일을 순회하며 이미지 파일을 validation 폴더로 이동
for label_file in label_files:
    with open(os.path.join(label_dir, label_file), 'r') as f:
        image_number = os.path.splitext(label_file)[0][:4]  # 파일명에서 확장자를 제거하고 처음 4자리를 가져옴

    image_file = f'{image_number}.jpg'  # 이미지 파일명 생성

    # 이미지 파일을 train 폴더에서 validation 폴더로 이동
    shutil.move(os.path.join(train_dir, image_file), os.path.join(valid_dir, image_file))