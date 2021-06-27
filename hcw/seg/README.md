# Semantic Segmentation

## 목차

* [Semantic Segmentation 프로젝트 소개](#semantic-segmentation-프로젝트-소개)
* [Environment](#environment)
* [Augmentation](#augmentation)
* [Models](#models)
    * [DeepLabv3+](#deeplabv3+)
* [Reference](#reference)

<br>

---

<br>

## 프로젝트 기간
- 2021년 04월 26일 ~ 05월 06일(11 days)

## Semantic Segmentation 프로젝트 소개
- **Semantic Segmentation 프로젝트**는 픽셀이 어떤 클래스인지 예측하는 Task 입니다.
- 쓰레기 데이터 4109장의 데이터로 학습을 진행했습니다.

## 평가 방법
![image](https://user-images.githubusercontent.com/67626878/123541215-49445c00-d77e-11eb-871a-5bbe579a5146.png)
- Test Dataset을 대상으로 **mIoU(Mean Intersection over Union)** 로 평가합니다.

<br>

---

<br>

## Models

### DeepLabv3+
- Depthwise Separable Convolution과 Atrous Convolution을 결합한 Atrous Separable Convolution의 활용
![image](https://user-images.githubusercontent.com/77056802/123552899-6c3d3300-d7b3-11eb-82b1-6e5f598abf4b.png)

### Encoder 
- se_resnext101_32x4d : pretrained model 활용
- senet154
- efficientnet-b4


## Augmentation

### Train Augmentation
- Albumentation 라이브러리 활용
- 단일 모델 : CropNonEmptyMaskIfExists(256), Resize(512), CLAHE(), HorizontalFlip(), Normalize(), ToTensorV2()
- 단일 모델 : CropNonEmptyMaskIfExists(400), Resize(512), CLAHE(), HorizontalFlip(), Normalize(), ToTensorV2()
- K-fold : Rotate(), CropNonEmptyMaskIfExists(400), Resize(512), CLAHE(), HorizontalFlip(), Normalize(), ToTensorV2()

<br>

## Ensemble

### Soft voting
- 각 class별로 모델들이 예측한 probability를 합산해서 가장 높은 class를 선택

<br>

## 실험 결과

### Model 실험
![image](https://user-images.githubusercontent.com/77056802/123553098-5d0ab500-d7b4-11eb-8662-5f7d9fc2a6f6.png)

### Parameter 실험
![image](https://user-images.githubusercontent.com/77056802/123553154-8a576300-d7b4-11eb-841e-4f63cf07b392.png)


<br>

---

<br>

## Reference
* [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030.pdf)

### 출처
- Deeplab v3+ : https://kuklife.tistory.com/121
- soft voting : https://devkor.tistory.com/entry/Soft-Voting-%EA%B3%BC-Hard-Voting
