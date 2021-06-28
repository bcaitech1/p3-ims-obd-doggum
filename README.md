# 부스트캠프 AI Tech - Team Project (Stage - 3)

## Summary
| Task | Date | Team | Public LB | Private LB |
|---:|---:|---:|---:|---:|
| Semantic segmentation  | 2021.04.26 ~ 2021.05.06 | 개껌 | 0.6803 | 0.6561 |
| Object Detection | 2021.05.10 ~ 2021.05.20 | 개껌 | 0.5878 | 0.4771 |

# Semantic Segmentation

## 목차

* [Semantic Segmentation 프로젝트 소개](#semantic-segmentation-프로젝트-소개)
* [Environment](#environment)
* [Augmentation](#augmentation)
* [Models](#models)
    * [DeepLabv3+](#deeplabv3+)
* [Reference](#reference)

## 프로젝트 기간
- 2021년 04월 26일 ~ 05월 06일(11 days)

## Semantic Segmentation 프로젝트 소개
- **Semantic Segmentation 프로젝트**는 픽셀이 어떤 클래스인지 예측하는 Task 입니다.
- 쓰레기 데이터 4109장의 데이터로 학습을 진행했습니다.

## 평가 방법
![image](https://user-images.githubusercontent.com/67626878/123541215-49445c00-d77e-11eb-871a-5bbe579a5146.png)
- Test Dataset을 대상으로 **mIoU(Mean Intersection over Union)** 로 평가합니다.

## Environment
- ?

## Augmentation
- Augmentation은 다양한 실험을 위해 비교적 학습이 빠른 DeepLabv3+, Mobilenetv2모델을 활용했습니다. 30에폭을 기준으로 잡아서 Val mIoU가 높은 Feature를 추출했습니다. 가장 좋은 효과가 있었던 것은 CLAHE, Horizontal Flip이었습니다.

|Augmentation|Epochs|설정|val loss|val mIoU|
|---|---|---|---|---|
|Augmentation 사용 X|17|Normalize, ToTensor|0.3371|0.5040|
|CLAHE|14|default|0.3099|0.5185|
|VerticalFlip|20|default|0.3757|0.4767|
|HorizontalFlip|20|default|0.3417|0.5039|
|RandomBrightnessConstrast|25|default|0.3594|0.4992|
|RandomContrast|24|(-0.2, 0.2)|0.3466|0.4970|
|rotate|19|(-10, 10)|0.3311|0.4993|
|RandomBrightness|30|(-0.2, 0.2)|0.3478|0.4970|
|Cutout|17|num_holes=30, max_h_size=6, max_w_size=6|0.3458|0.4986|
|CLAHE+HorizontalFlip|24|default|0.3286|0.5126|
|ToGray|21|default|0.3391|0.4894|
|Equalize|28|mode:cv, by_channels=True|0.3591|0.5005|
|CropNonEmptyMaskIfExists|28|(512, 512)|0.3569|0.4979|
|CropNonEmptyMaskIfExists + Resize|26|(256, 256), (256, 256)|0.3229|0.4671|
|invertImg|27|defalut|0.3697|0.4973|

## Models

### DeepLabv3+
- ?


## Reference
* [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030.pdf)



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
