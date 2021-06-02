# 개요
- 재활용 쓰레기 사진으로 부터 각 픽셀의 쓰레기 class를 찾는 semantic segmentation model

# 데이터

- 전체 이미지 개수 : (512,512) 4109장

- class(12) : Background, UNKNOWN, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing

![image](https://user-images.githubusercontent.com/43736669/119310188-5dda9380-bcaa-11eb-844d-36f2707331d3.png)

# 모델
### 나열한 모델들의 soft ensemble

- DeepLabV3Plus
    - encoder : efficientnet-b3, efficientnet-b6, efficientnet-b0, senet154, se-resnext101_32x4d
- DeepLabV3
    - encoder : efficientnet-b0 

# 학습
### Data Augmentation
- 성능 향상을 이끌었던 Augmentation 적용 후 학습(Rotate, CLAHE, HorizontalFlip, Crop 등)

### Scheduling
- valid mIoU 계산 후 최고점을 계산하지 못 했으면 best state로 다시 돌아가 lr을 낮춘 후 재학습
- train loss와 valid loss의 동시 감소 유도

### 실험 관리
- wandb 및 notion 

# 결과
- 최종 LB mIoU : 0.6561, 60위


