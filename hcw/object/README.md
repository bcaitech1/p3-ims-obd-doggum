# object detection

## 목차

* [Object Detection 프로젝트 소개](#object-detection-프로젝트-소개)
* [Environment](#environment)
* [Models](#models)
    * [Swin-Transformer](#swin-transformer)
    * [Neck : FPN](#neck)
    * [### Loss](#loss)
* [Augmentation](#augmentation)
    * [Train augmentation)(#train-augmentation)
    * [Test augmentation](#test_augmentation)
* [Ensemble](#ensemble)
    * [Non-maximum Suppression](#non-maximum-suppression)
    * [Weighted Boxes Fusion](#weighted-boxes-fusion)
    * [적용과정](#적용과정)

* [Reference](#reference)
* [출처](#출처)

<br>

---

<br>

## 프로젝트 기간
- 2021년 05월 10일 ~ 05월 20일(11 days)

## Object Detection 프로젝트 소개
- **Object-Detection 프로젝트**는 위치와 해당 객체를 분류하는 Task 입니다.
- 쓰레기 데이터 4109장의 데이터로 학습을 진행했습니다.

<br>

---

<br>

## Environment

- mmdetection



## Models

### Swin-Transformer
- Swin-Transformer는 최근 Object Detection 분야에서 좋은 성능을 보여주는 SOTA 모델입
- mmdetection에서 tiny, small, base 모델을 지원하고 있으며, 가장 성능이 좋았던 base 모델 활용
- 보통 SOTA의 모델들은 Instance Segmentation을 활용하는 HTC모델이지만 Instance Segmentation을 활용하지 않는 방법 사용

### Neck : FPN
- Top-down 방식으로 특징을 추출하며, 각 추출된 결과들인 low-resolution 및 high-resolution 들을 묶는 방식
- 각 레벨에서 독립적으로 특징을 추출하여 객체를 탐지하게 되는데, 상위 레벨의 이미 계산 된 특징을 재사용 하므로 멀티 스케일 특징들을 효율적으로 사용 가능.

### Loss 
- Class Loss = CrossEntropyLoss
- Bbox Loss = GloUloss



## Augmentation

### Train augmentation
- RandomFlip
- AutoArgment : Resize, RandomCrop
- Normalize, Pad, DefaultFormatBundle, Collect

### Test augmentation
- MultiScaleFlipAug
- Transform - Resize, RandomFlip, Normalize, ImageToTensor, Collect

<br>

---

<br>

## Ensemble

### Non maximum Suppression(NMS)
- 동일한 Object에 여러개의 bbox가 있다면, 가장 스코어가 높은 박스만 남기고 나머지를 제거하는 것

### Weighted Boxes Fusion(WBF)
- 다양한 물체 감지(Object Detection) 모델의 예측을 앙상블하여 성능을 향상시키는 새로운 방법
- 예측의 일부를 단순히 제거하는 NMS, Soft-NMS extension과 달리, WBF는 모든 예측된 사각형을 사용하므로, 결합된 사각형의 품질이 크게 향상시킴

### 적용 과정
- Cascade mask swin base 기반 모델 K-fold 결과 각각에 NMS(=0.4) 적용하여 중복되는 Bbox 수 줄임
- HTC swin Transformer 모델과 NMS를 적용하여 나온 Cascade mask swin base 모델 결과에 각각 weight를 달리하여 WBF 앙상블 수행


<br>

---

<br>

## Reference
* [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030.pdf)
* [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)
* [DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution](https://arxiv.org/pdf/2006.02334.pdf)
* [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431v2.pdf)


### 출처
- FPN : https://eehoeskrap.tistory.com/300 [Enough is not enough]
- GIoULoss : https://www.tensorflow.org/addons/api_docs/python/tfa/losses/GIoULoss
- WBF : https://lv99.tistory.com/74
