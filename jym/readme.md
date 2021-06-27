# Semantic Segmentation

## 검증전략
DeepLab V3Plus 를 Architecture 로 활용하고 backbone 과 다른 하이퍼 파라미터를 변경해가며 다양한 실험을 했다. SOTA 모델인 HRnet 을 구현해보고자 하였으나, 2 주라는 짧은 시간에 논문을 보고 구현하고 이번 프로젝트에 최적화 시키는 것까지는 하지 못했다. 결국 DeepLab V3Plus 에 집중을 해서 끝까지 밀고 나갔다.

## 최고 단일 모델 성능
DeepLab V3Plus + se-resnext101_32x4d(Backbone)를 사용
Val mIOU : 0.5560, Public LB : 0.6331 처음에는 가벼운 모델 위주로 실험을 하면서 최고의 하이퍼 파라미터와 argument 를 찾았고, CLAHE 와 Flip 만이 성능이 올라가는 경향을 보여서 사용했다.

## 최고 모델 성능(앙상블)
Public LB : 0.6803(8th), Private LB : 0.6561(11th)
DeepLab V3Plus + se-resnext101_32x4d(Backbone)으로 5fold cross validation 을 진행했다. 또한, DeepLab V3Plus architecture 에 backbone 으로 efficientnet b4, efficientnet b7, senet154 사용한 모델도 앙상블을 진행하였다.

## 실험관리
개인적으로 해 본 모든 실험을 wandb 를 통해 관리했다. 본격적으로 wandb 를 사용한 것이 처음이었지만 사용법이 굉장히 간단하였고 눈에 보기도 편해서 금방 익숙해질 수 있었다. 그리고 wandb 를 사용하면서 가장 인상깊었던 점이 잘못된 실험을 하고 있음을 바로 파악할 수 있었다는 것이다. 아래 그림에서 분명 동일한 모델을 사용했음에도 불구하고 하이퍼 파라미터를 조금 수정했을 뿐인데 mIOU 가 대폭 상승하는(제일 상위에 있는 노란색)경우가 있었다. 처음에는 단순히 점수가 많이 올라서 좋았지만 제출을 하여 Public LB 를 확인하니 오히려 다른 모델에 비해 점수가 낮았다. 그래서 원인을 분석해본 결과 validation dataset 에 실수로 train dataset 의 20%정도를 추가해서 실험을 진행한 것이었다. 이전의
대회처럼 Wandb 를 사용하지 않았다면 발견하지 못했을 시행착오였는데 이번에는 실수를 바로 알아차리고 원인분석까지 할 수 있어서 도움이 많이 된 것 같다.


# Object Detection

## 검증전략
Mmdetection 라이브러리 사용법을 최대한 빨리 익히고 원하는 실험을 바로바로 할 수 있는데까지 꽤 오랜시간이 걸렸고 대회가 끝난 지금도 완벽하진 않은 것 같다. 그럼에도 불구하고, 현재 object detection SOTA 모델인 Swin-Transform 을 구현하고, optimizer, augmentation, lr scheduler 등을 조절해가면서 최고의 점수를 얻으려고 노력했다.

## 최고 단일 모델 성능
Mmdetection 라이브러리 사용법을 최대한 빨리 익히고 원하는 실험을 바로바로 할 수 있는데까지 꽤 오랜시간이 걸렸고 대회가 끝난 지금도 완벽하진 않은 것 같다. 그럼에도 불구하고, 현재 object detection SOTA 모델인 Swin-Transform 을 구현하고, optimizer, augmentation, lr scheduler 등을 조절해가면서 최고의 점수를 얻으려고 노력했다.

## 최고 모델 성능(앙상블)
Cascade mask rcnn with swin transform(5fold cross validation) + Cascade mask rcnn with HTC – Public LB : 0.5878(9th), Private LB : 0.4771(6th)
5fold cross validation 과 새로운 HTC 모델을 이용하여 앙상블 효과를 노렸고, Weighted Box Fusion, Non-Maximum suppression 등을 사용하여 최종적으로 결과를 산출했다. 단일모델에 비해 약 0.05 상승이 있었지만 결국 0.6 을 넘지 못해 다소 아쉬움이 남았다. Private 에서 약간의 shake-up 이 생겨 순위가 3 등 올랐다. 하지만, 일반화를 잘했다고 생각하지는 않는다. 왜 일반화가 잘 된 것인지 또는 private test set 에 fitting 이 잘 된 것인지 회고를 해보아야 할 것 같다.

## 실험관리
팀원이 하나의 Notion 채널에서 각자 실험한 log 를 남기면서 어떻게 해야 성능을 높일 수 있을지 고민했다. 또한, 개인적으로는 Wandb 를 사용하며 mAP, loss, learning rate 등이 어떻게 변하는지 살펴보았다. 지난 대회까지는 체계적으로 실험관리를 하지 않아서 이미 해 본 실험을 다른 팀원이 동일하게 하는 등의 비효율적인 모습이 있었는데, 이렇게 실험관리를 하면서 조금 더 꼼꼼하게 대회를 진행할 수 있었다.
