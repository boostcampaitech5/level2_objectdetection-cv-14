## 프로젝트 개요

산업혁명 이후로 대량 생산 시대가 찾아오면서 쓰레기 처리문제, 매립지 문제 등은 끊임없이 인류를 괴롭히는 문제 중 하나가 되었습니다. 최근에는 코로나 이슈로 쓰레기 문제는 점점 더 심각해지고 있는 상황입니다.


이러한 상황 속에서 분리수거의 중요성은 더 커지고 있습니다. 쓰레기 양이 많아지는 만큼 분리수거가 제대로 되지 않는 쓰레기들도 많아지는 것은 당연한 결과입니다.

이를 해결하기 위해 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어보려고 합니다. 모델에 필요한 데이터셋은 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진들이 제공됩니다.

- **Input :** 쓰레기 객체가 담긴 이미지와 bbox 정보(좌표, 카테고리)
- **Output :** 모델은 bbox 좌표, 카테고리, score 값을 리턴

|  이름      | 역할                                                         | github                         |
| :-------: | ------------------------------------------------------------ | ------------------------------ |
|김용환       | yolox, yolov5x6 실험                                         | https://github.com/ |
|김우진       | htc_Swin_L(22K), yolov5l6 + TTA 실험, EDA 수행               | https://github.com/    |
|신건희       | Swin_L_Cascade R-CNN 실험, Oversampling 실험                 | https://github.com/   |
|신중현       | Swin-T, L 기반 Cascade R-CNN, HTC 실험                       | https://github.com/    |
|이종휘       | Centernet2, DyHead, Universenet 실험Augmentation 실험, Ensenble(WBF) 코드 | https://github.com/    |

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

# 실험 세팅

피어셰션을 통해 매일 방향성을 정하고, 노션과 wandb를 이용해 실험 내용을 공유하였습니다.
데이터를 stratified k-fold로 분할하여 평가하였으며, 클래스 별 mAP, confusion matrix 등을 평가 방법으로 사용하였습니다.
EDA

# EDA

EDA를 통해 클래스 불균형, 작은 박스 크기로 인한 분포, 어두운 이미지, 사이드에 치우친 이미지, 흔들린 이미지 등의 문제점을 확인하였습니다.
이러한 문제를 데이터 augmentation을 통해 어느 정도 해결할 수 있을 것으로 기대하였습니다.
작은 박스에 대한 실험 결과, 대부분의 모델에서 small box에 대한 mAP는 낮고, large size에 대한 mAP는 높았습니다.
모델과 하이퍼파라미터 튜닝


실험 결과, yolo와 다른 모델들의 앙상블이 가장 높은 mAP를 달성하였습니다.
yolo의 성능은 낮지만, 확실한 박스의 confidence를 높이는 경향이 있어 앙상블 시 성능 향상에 기여한 것으로 분석되었습니다.
앙상블 결과는 mAP가 높지만 현실적으로 활용하기에는 불필요한 박스가 많아 실용적이지 않을 수 있습니다.
추가적인 후처리나 조정을 통해 현실적으로 활용 가능한 결과물로 변환할 수 있을 것으로 판단됩니다.

## 모델 아키텍쳐

Faster R-CNN, Cascade R-CNN, SSD, YOLO 등의 모델 아키텍처를 고려하였습니다.
Cascade R-CNN은 어려운 샘플에 대한 처리에 강점이 있다고 판단하여 사용하였습니다.
Swin-Large, ConvNeXt-XLarge와 같은 pretrained 모델을 백본으로 선택하였으며, 상대적으로 낮은 learning rate를 설정하였습니다.



# Augmentation 

augmentation을 적용하여 전체적인 mAP를 상승시켰습니다.


# Training

# Inference

# Ensemble

앙상블을 위해 다양한 모델들을 사용하였으며, 모델의 다양성이 mAP 향상에 중요한 역할을 한다고 판단하였습니다.
여러 모델을 앙상블하기 위해 Weighted Boxes Fusion(WBF) 방식을 사용하였습니다.

# Reference
---
blog : https://comlini8-8.tistory.com/m/97

약초의 숲 blog : https://herbwood.tistory.com/2

Cascade RCNN : https://arxiv.org/abs/1712.00726

Swin Transformer : https://arxiv.org/abs/2103.14030

ConvNext : https://arxiv.org/abs/2201.03545

WBF paper : https://arxiv.org/abs/1910.13302

pretrained model Github : https://github.com/facebookresearch/ConvNeXt, https://github.com/microsoft/Swin-Transformer 

