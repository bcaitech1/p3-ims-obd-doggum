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
