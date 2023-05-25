## 프로젝트 개요

![](https://lh3.googleusercontent.com/u/0/docs/ADP-6oHbFuIvdku4p_uGDCc6xdpodWjrSPuA5C1x6zDKct1_F_zG0qB85VPj73gAGx3qzjWnoSP5cYcmOZ91bJ1RSGiCCR_m0L5GcN9ENgUvJo6QJho2Bi7RhOkHTph7lYv25BsIPWrPLCPxLayiyNRBQ0LA6Z7afRgQf1r8GiwxtYVa12Dd-mGkll_Logp3h44z5EU2YM0e5wZ4lY68HkK4mIsCXVb_hJLK4g30f3ZK7aWDGX5uEoIyYyDEE6SbdtYpPq8tOluHNoiQ81lELb2HGoDSxInOo35b0itDtFwO3pe2W472KijHCmyjHEIGOsHD58xQaGgKYbvzRiiRi3FCoB6sqwK1-8G7dj44IPAlGvvx4OkKeuVCM1KbOdAzPGofpm7cJJAUj22RemP53ZkVX_a1ielMcWF_9d2ujv7JEkFz6vUL4pymNp9eu-tOy_mHZgO4jWqrBbsLWRUZbUim30RB3wCIIOB_skVIh0rp402P4qpSE_CX5hxAI3fVMBKO-CSs05OkNzz_ibNWV_HxDsr3PF7XRc91K4F-oAfUTjRrFBflmEfcU9znbuIQNPlBGktpcdECQcdE_ZAJ_kgMR5dKb5WpEXOEWSf1KFdWZ0kkNcAfTbvOrN-R0DctPakiwZw5OXputSzL3GTIlaPKy8XJt6CkZtjbrsPfWvSZxFESb1ayXYruXgsEmw4fDZxAEHuYU6bvqmwTpD8YbAPsguuF6BXgL-zI9lVccqJU3EoLCn8QRm3A6H1NONStlI8tImjOSstZB2Vb1ViYuJHbUQAYvo_3eHLuQzATjjhtTZxM56uQYmU6CJnq1OJaSGlw3Nogd19dvxGtQuyp9oQXNDdxlrd8HDqkpajRpWUS-m9_7amRryLemW5al3ITXSztQvS1Oc3U9FqxmMCsGdyvQXo)

산업혁명 이후로 대량 생산 시대가 찾아오면서 쓰레기 처리문제, 매립지 문제 등은 끊임없이 인류를 괴롭히는 문제 중 하나가 되었습니다. 최근에는 코로나 이슈로 쓰레기 문제는 점점 더 심각해지고 있는 상황입니다. 이러한 상황 속에서 분리수거의 중요성은 더 커지고 있습니다. 쓰레기 양이 많아지는 만큼 분리수거가 제대로 되지 않는 쓰레기들도 많아지는 것은 당연한 결과입니다.이를 해결하기 위해 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어보려고 합니다. 모델에 필요한 데이터셋은 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진들이 제공됩니다.

- **Input :** 쓰레기 객체가 담긴 이미지와 bbox 정보(좌표, 카테고리)
- **Output :** 모델은 bbox 좌표, 카테고리, score 값을 리턴

|  이름      | 역할                                                         | github                         |
| :-------: | ------------------------------------------------------------ | ------------------------------ |
|김용환       | yolox, yolov5x6 실험                                         | https://github.com/ |
|김우진       | htc_Swin_L(22K), yolov5l6 + TTA 실험, EDA 수행               | https://github.com/    |
|신건희       | Swin_L_Cascade R-CNN 실험, Oversampling 실험                 | https://github.com/   |
|신중현       | Swin-T, L 기반 Cascade R-CNN, HTC 실험                       | https://github.com/    |
|이종휘       | Centernet2, DyHead, Universenet 실험Augmentation 실험, Ensenble(WBF) 코드 | https://github.com/    |

## Leaderboard score
#### public 1등 mAP 0.7265 , private 1등 mAP 0.7171

![](https://lh3.googleusercontent.com/u/0/docs/ADP-6oEcqXs1vmOCoOUVkeAP-7hTBjmnw1moRZa5xG0oXG7BLnlWaEUksq6J8uW3pMBWQ7OGxwf_VejsClgz9QWQZeyJEbONMojc171wtUXqxGiek3mLh632MRxeVmOxPsYpRJWGwvqobHCeuXq-zpU5uhzK3Y2WxJiH960QmPExQTkmF1Fntbsz9ResBRPRLgQKToAZ434-lk2G57nsbc0SqwBTyxtE_ChrtJGBYIkAYne986orUrYGI8wEHiLya3Xs-zoCvT1A_J_vUmIXYugfjxrQ09XwV6R771uz6-jmN9HGW9Iuxon2c8G9eRBTf7Oms0M-TXIwrttkhQKxqpHfthxyd0OZxPgVTAXFAz92urmvJVB7kq3dz4IFD1THnd_faAv1V31hhfPrk3CffgnH_5g3J1jSZHdQBrDVBECddrwXoMJKy-rw98RyzgN5Lslgkt9G07L0yrNf1X1VJZf3H2W5pLKEbq8akO6latygNx0oq8aC1u2R1ADD8CBRDlMZijniBU7V7eARlAd5-DQQH3YLFQKMTSbrxYGr4h83lCYlv4PxNbnqMAAOBB06rDZDiMzNbTwnrmvYYRptz-opDavQzIOOPlCMVE-lUbefIfS1vybSTonjZHbZpjA6ibErhF-kuzLoFZ2UNdG5Y4E8JEZ8VruPTe9QfFxvYRNtAS5dqx01SjNzJ6tuCk871_SSf4TTxHipkzFgLyqSfXETktySKUEg3oeLk0U-W2_pKWiwKqrk96O8LBHqwR-rTpfc5dTaqLEuhWncYegJ3jHSYOBxSc89iMaU8syGxZSd4EyfzW4iNzVWxE4Ev3eCXkTD2_DKgTUOnQGu_z9dpBtam1ro5Wl-EBui7oMGW0rhdyDfXGFq-h_5rAUGEwM175tbYpUiKIYmFiceauMr1DlFHIA)

![](https://lh3.googleusercontent.com/u/0/docs/ADP-6oEPwMPA9tweHgQFiLv_2oTu_Va49DqzsYyjJZdwBWg6RWP616qIesrV2940C6KN31AxWZxfNWKeX27LF42GyKGkcHcJIVnCA_lyjbt5TJ-jad8pDcTMiU-aPavU0ydpFzOGwQda0sNvutF_kOVZGvcJpHeD7Oj-8W-fu4ZhmWGqieMrFIgm26cpsiDIXr_LbrVvoffdJ25hkjXpRJSiRuI5a4NGoX0HBCKjPOGVps8azHk9HgvMneZPQ6wwhix98IV7ncvm67QoxY-U3TtW4Ln4rHYd90IW14kP_G0VkKi9Me1WENh2X6AaC92Q2650IOrjianjUiUmvQ4wYBBk_0XjhLPxrn2HxpvSCWv655tt8wxP0NrbYKUcg3qVnvh-wU_PihC2CY8CFCiyux_iZWkWR26aijbqCTesKD0k0Sji2U2hxvCFSuWosAD3aq1fA2--MtNAWACzrWhCo4vtGDHwPJmH0NLO0DaJZuxksF4eeCCqVGSOAMV_RSgRzoAhbShHMgWjG2DozGvLGUuFV282Ysj-Fqi5Sm28CHNLu8KKWvgQd_LZAAAjZB3zla8_EapBg3CRfLfqkN_C7JrHpj89VCH-P7ozwMuUcmsFlnp6JpvkOjCxv_-iHMeJQJY5Z-aVmzYfxjKhz5AI2zMKi914b_DHUwyJ0KzL6p037DFF5Lmxi9tmkvHFJAG6EnRy0wjOxCuHVGE8DxWuK47nX21EddrcYgVBF_oUHsWITzepZUk874PJCItTW0bMpS9ldRqhF9Rhr_EGtArXCLzjAsbbBMmcG-2G7mW4a8Mhx8DmKEofx7N6IaV2n23FMkm-6H_tD-dO3qYDP4pzm4DI6fuV3lVUbKneKAwo4slHfqLguZ-0hw5f78hb90nekvIID-RftFvRJgAMd5e04q4CK9A)

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

![image](https://github.com/boostcampaitech5/level2_objectdetection-cv-14/assets/78690390/f10a587c-f4bf-476a-9d1f-f7b3e90843ea)



# Dataset

- 전체 이미지 개수 : 9754장 (train : 4883 장 , test : 4871 장)
- 10 class : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기 : (1024, 1024)
- COCO Format

# 실험 세팅

피어셰션을 통해 매일 방향성을 정하고, 노션과 wandb를 이용해 실험 내용을 공유하였습니다.
데이터를 stratified k-fold로 분할하여 평가하였으며, 클래스 별 mAP, confusion matrix 등을 평가 방법으로 사용하였습니다.

<img src="https://lh3.googleusercontent.com/u/0/docs/ADP-6oG-hJb9xdP_I_IoicHvsrMW4PHn3YBsIqIByBQ-AEmQKSyOKtjQSc7Hs3VZI6r7Ploh7ytq5G1-FCuAL55TYCsKn-b7E442WwMOMU7IYezxiQPZI6SRb0zqbSvNPPKPXkPRF1-1p6vF48n7YSM3eNJfgC1-KQ4kHjUj5BGRork1CQ4wIHVmwuMUMMkahWvP6xUUmYItxEAKcQyJTjAhMYojlGQWq8-9pMk4kEhy0711aMlTjrR5vyEOJAcZQDNg1Q6_VGtlMcuDu6x94xOSYLwarowheV6m0tgbzWo3STlktd9up0qQD_7U9kkDd7sZevzCJNJ3iJ4bR9CG0CA9RHn2ORGfcHuc5kzY8JQP-Ex7nXR8Z0WaQRoogYPoM464zzs3RNyyTjL1NoRORQIrZ1sQpms5XqNBv6wD2jlY1zj68gvjR33K0f0IVBAz_8U5ameEjxKEcrgWsSO_6D_wMYieGglU21VqoFxAWDv4SfJVJo3pCLXF3GteJFfvhvZEDdQ9nvn4hzXFafB1nmuCGkV6-2i9XDWFrV-JPaRfZcrBY1bmtP_H2BzM4kW2uNdrj2cO_bCaA2gXVe6qBQDcuoCFs2pg3zOlqHqWm0fR7kVfHFMVXT8VAHgtInGxbu0Thk1YyKoqlnKi5WvkocoyoXtQ9NKYRfTtDf_pOU1-YzOPE-5f1mJdWzm9Cgg9x1BxH5_aD50ru0oJAf6J7i7zIE3qKnanlh5yUDv5s1o3BWiRA-FnbkYWkxWVlIw3jBKPdMj1OM43kKC2QeT8MRUoVeCMUSE1Q10HdZ_jEDMRpG6iWlIkzD-exHtBXLFlaCktoJkTTnxjKKhpKtdJVXqhHfF3pcJLLUSdWsB3Y7W-aS16IvEDWWDe_SizoxB1JsqPrQqcBw8S260aOiFMgnPaIZU" alt="Example Image" width="400px" height="300px">

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
![](https://lh3.googleusercontent.com/u/0/docs/ADP-6oFBVQGnM8N4yemNaiX80qTh-PAZaP_QVqqwqzHBMPrif1Fx-OL228G-sfFcuAtXenZ-CTnc1SZjLSQJplOePT7u96Xs-SzUCFkOx6fSGRtYMNfDmObO7Si5WMAhIbNtbHCPVoznxcxt9nLjdx6vtgyEECydGQSYupunEOuJzotqGeqKknvZ_e3lMLqZgYbflKt6sBiUq5Z3MHXpLOrc02wXjpeY3C7pHc9KVHfckEnZKKnqPz-nvV4cvurlEF2k-h_4a296AOoGbqks8pkwTeBNvX6DqGu15A75frvTdhjnwRxE-WpcM_BFRTRwjWIhDrE-LqZOjX-d3EWMBenB_zW9VBRMw9OiQRjeQmhi6Aqkhwp_UdpJFMEjnpMs2AmITaGmgkUWcXYZrty98aHsSCwHefYjssYfun-M-eBOE2MaS8VLPkuRcKVAPnv5e_rHqq0txdnvcVZAeIMhMaUnClOdDNXQF77PPxzudTH2L40sMRs_-73OX3IT2MuRgJNU0sKFnrKciY828alUYwq1wB2MHqWrYtaSJUGNwMWtK5R1C3LMp_cMRrF_leyaNXQpatqmu3pj7NFsfSGgNfELtSluJn-LOCw9X8URMHS4UACMAeKXrYUYhkxnX76Hz6y-S2DT_QYmfXL33n49wVXLjp_qFlAL2nr4FfBeqNhjyQEk-XBwgCMr4aafmyf2LRaW-rvEmAWBmTTDTI53kZdIAXfMlTIRggypJYgJW2aRiDvusJHo4iKGvot-mf5-FnHm-RTTWjTsPiKxQwgnpSsRDiYjrbU9_BLQqO5w3tA9ttHUROg2J0YW9OT-djDrEYFbW6JifFD1bMd10_Qtd3V6Hrhsx2HQCpv3KZkJ3-iXH20yiQ961lr2gNwJT4Uv-8cJNuJlxERp5rTGCConWC9LrKk)

wbf ensemble confidence calculation
![](https://lh3.googleusercontent.com/u/0/docs/ADP-6oHDHEEnoJnjfKUmQCCjk14R5YbB5oWJmU7981RUIQ2bli_itqYViOwh7BXH7kEr1NlLM2G1x7Bk_g2Ut4hZQjtdDiT-EY-WdbzAwFunyNqi-ggxyK2Tgh9ZS7fDvA9Rz74CpBD4zvmq4k4byJIt7L5dHJt3irSdo1YYtJaeM-4mDBtOumoPKurCVURmn5KqcnNb6i9UlULEKSSFapPAC6VqjVL8PpbPA9b9hc1IeJKwQyw5maqei-jgD0NUxAoJOi3e03zSXIC8OoSlR4kxK5Xz23x2YLVsFNKB6ruTNwedT9lJJzdedBU2n1i845ZJm00Yv11d2vR8Ov56AyNSLvoqN8AhuR7dOs1NbDFWB53xcf8rdCIvowE6_AV6GIILBHMe2T_FiCdy8FLSMPmY6lSaXdOIxFBwVgfUsNP2tpSa_rBr4HZ35Ea7dWvn7EcGtjERT1cKoH1xVimEebA065dHGcWOUQfjLczpoXowKqcTJGGsuGo7u6lvRUdi9g93VSabUavPSDZfbTl_fvphIaemm9QehODRWIlDWrHBkCLD1FBRsmmqgmUqDJxtzqLNmWBKsdVsJv8ITxgkaxyNya0txoMhnsh0kvycoXyXsjyNNjwBBEf73g7aJz2AfS-SnU-h6q48dWBT_5z2IkbOO1xlePhOYjQYYGXTEqMfdq4hrCmBKwJtNy_Prl5yHTm461CA7whYXkFctn4Bl9K33E7DadU8OhforSN6KCncFbc48e_4MlTm-N90hMmI8WOw-mttn4GEab-3CRJ_OBW-T2g7SoFWWk362C4g1jmkzzCGYBBOakAe-y_P8ZVMjsdOImIbftle0XXPIGO9uVdeW7zaFbhBR0A7-mI3wgScB8wYr7gGHGyXdnYQuq9p7CI--580hs2I48Cr34Gsb1tnRVQ)

# 전체 진행 과정
![image](https://github.com/boostcampaitech5/level2_objectdetection-cv-14/assets/78690390/945661ee-80c9-4ce2-9d26-8b9d1c6d6778)

# Reference
---
blog : https://comlini8-8.tistory.com/m/97

약초의 숲 blog : https://herbwood.tistory.com/2

Cascade RCNN : https://arxiv.org/abs/1712.00726

Swin Transformer : https://arxiv.org/abs/2103.14030

ConvNext : https://arxiv.org/abs/2201.03545

WBF paper : https://arxiv.org/abs/1910.13302

pretrained model Github : https://github.com/facebookresearch/ConvNeXt, https://github.com/microsoft/Swin-Transformer 

