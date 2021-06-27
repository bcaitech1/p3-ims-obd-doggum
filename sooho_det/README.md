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

## 평가 방법
![image](https://user-images.githubusercontent.com/67626878/123540325-5dd22580-d779-11eb-90a8-a9f8990f1143.png)
- Test Dataset을 대상으로 **mAP50(Mean Average Precision)** 로 평가합니다.

## Environment

- mmdetection

## Models

### Swin-Transformer
- Swin-Transformer는 최근 Object Detection 분야에서 좋은 성능을 보여주는 SOTA 모델입니다.
- mmdetection에서 tiny, small, base 모델을 지원하고 있으며, 가장 성능이 좋았던 base 모델을 활용했습니다.
- 보통 SOTA의 모델들은 Instance Segmentation을 활용하는 HTC모델이지만 Instance Segmentation을 활용하지 않는 방법을 사용했습니다.


### Faster-RCNN
- Swin-Transformer에 비해 빠른 속도로 학습이 가능했습니다.
- Faster-RCNN을 통해 다양한 데이터 관련 실험을 진행했습니다.
- 다른 모델에 비해 성능이 현저히 떨어져서 최종 앙상블에는 활용하지 않았습니다.

### DetectoRS
- Backbone으로 ResNext101_32x4d 모델을 활용했으나, Swin-Transformer에 비해 성능이 좋지 않고 학습 속도도 느려 최종 모델로 활용하지 않았습니다.

## Post-processing

### Non maximum Suppression
![image](https://user-images.githubusercontent.com/67626878/123540929-6e37cf80-d77c-11eb-8383-57cdc7c28ab6.png)
- NMS는 불 필요한 박스들을 IoU를 기준으로 합쳐주는 방법으로, 앙상블 이전에 최대한 후처리하고 앙상블을 진행했습니다.


## Reference
* [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030.pdf)
* [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)
* [DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution](https://arxiv.org/pdf/2006.02334.pdf)
* [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431v2.pdf)
