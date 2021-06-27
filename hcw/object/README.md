# object detection

## 목차

* [Object Detection 프로젝트 소개](#object-detection-프로젝트-소개)
* [Environment](#environment)
* [Models](#models)
    * [Swin-Transformer](#swin-transformer)
    * [Faster-RCNN](#faster-rcnn)
    * [DetectoRS](#detectors)
* [Post-processing](#post-processing)
    * [Non-maximum Suppression](#non-maximum-suppression)
* [Reference](#reference)

## 프로젝트 기간
- 2021년 05월 10일 ~ 05월 20일(11 days)

## Object Detection 프로젝트 소개
- **Object-Detection 프로젝트**는 위치와 해당 객체를 분류하는 Task 입니다.
- 쓰레기 데이터 4109장의 데이터로 학습을 진행했습니다.


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


### 출처
- FPN : https://eehoeskrap.tistory.com/300 [Enough is not enough]
- GIoULoss : https://www.tensorflow.org/addons/api_docs/python/tfa/losses/GIoULoss

## Augmentation

### train augmentation
- RandomFlip
- AutoArgment : Resize, RandomCrop
- Normalize, Pad, DefaultFormatBundle, Collect

### Test augmentation
- MultiScaleFlipAug
- Transform - Resize, RandomFlip, Normalize, ImageToTensor, Collect

## Post-processing

### Non maximum Suppression
![image](https://user-images.githubusercontent.com/67626878/123540929-6e37cf80-d77c-11eb-8383-57cdc7c28ab6.png)
- NMS는 불 필요한 박스들을 IoU를 기준으로 합쳐주는 방법으로, 앙상블 이전에 최대한 후처리하고 앙상블을 진행했습니다.


## Reference
* [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030.pdf)
* [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)
* [DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution](https://arxiv.org/pdf/2006.02334.pdf)
* [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431v2.pdf)

