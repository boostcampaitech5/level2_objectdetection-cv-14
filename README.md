## 프로젝트 개요

산업혁명 이후로 대량 생산 시대가 찾아오면서 쓰레기 처리문제, 매립지 문제 등은 끊임없이 인류를 괴롭히는 문제 중 하나가 되었습니다. 최근에는 코로나 이슈로 쓰레기 문제는 점점 더 심각해지고 있는 상황입니다.


이러한 상황 속에서 분리수거의 중요성은 더 커지고 있습니다. 쓰레기 양이 많아지는 만큼 분리수거가 제대로 되지 않는 쓰레기들도 많아지는 것은 당연한 결과입니다.

이를 해결하기 위해 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어보려고 합니다. 모델에 필요한 데이터셋은 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진들이 제공됩니다.

- **Input :** 쓰레기 객체가 담긴 이미지와 bbox 정보(좌표, 카테고리)
- **Output :** 모델은 bbox 좌표, 카테고리, score 값을 리턴

|  이름  | 역할                                                         | github                         |
| :----: | ------------------------------------------------------------ | ------------------------------ |
|김용환| yolox, yolov5x6 실험                                         | https://github.com/ |
|김우진| htc_Swin_L(22K), yolov5l6 + TTA 실험, EDA 수행               | https://github.com/    |
|신건희| Swin_L_Cascade R-CNN 실험, Oversampling 실험                 | https://github.com/   |
|신중현| Swin-T, L 기반 Cascade R-CNN, HTC 실험                       | https://github.com/    |
|이종휘| Centernet2, DyHead, Universenet 실험Augmentation 실험, Ensenble(WBF) 코드 | https://github.com/    |

# Contents

```
baseline
├── mmdetection
│   ├── config
│   └── ...
├── requirements.txt
│
└── yolo
		├── dataset
		│   ├── images
		│   └── labels
		└── cocotrash.yaml
```



# Dataset

- 전체 이미지 개수 : 9754장 (train : 4883 장 , test : 4871 장)
- 10 class : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기 : (1024, 1024)
- COCO Format



## 모델 아키텍쳐



# Augmentation

# Training

# Inference

# Ensemble

# Reference

