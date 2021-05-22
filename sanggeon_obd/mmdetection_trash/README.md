### 들어가기 전에..
사용한 라이브러리  
https://github.com/open-mmlab/mmdetection  
https://github.com/SwinTransformer/Swin-Transformer-Object-Detection

# ai tech boostcamp
# P-stage 3-2 (object detection)
## 재활용 품목 분류를 위한 Object Detection

![img.png](files/img.png)

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 만들어져 쏟아져 나오고, 그에 따라 대량으로 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 심각한 쓰레기 문제를 낳고 있습니다. '쓰레기 대란', '매립지 부족'과 같은 표현을 뉴스에서 듣는 빈도가 점점 늘어나고 있다는 것만으로도 그 문제가 얼마나 심각한지 알 수 있죠.

![img_1.png](files/img_1.png)

이러한 환경 부담을 조금이나마 줄일 수 있는 방법의 하나로 '분리수거'가 있습니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립, 소각되기 때문입니다. 우리나라의 분리 수거율은 굉장히 높은 것으로 알려져 있고, 또 최근 이러한 쓰레기 문제가 주목받으며 더욱 많은 사람이 분리수거에 동참하려 하고 있습니다. 하지만 '이 쓰레기가 어디에 속하는지', '어떤 것들을 분리해서 버리는 것이 맞는지' 등 정확한 분리수거 방법을 알기 어렵다는 문제점이 있습니다.

따라서, 우리는 쓰레기가 찍힌 사진에서 쓰레기를 Segmentation 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 11 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 올바른 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 성능을 올려 지구를 위기로부터 구해주세요! 🌎

## 학습데이터 개요
우리는 수많은 쓰레기를 배출하면서 지구의 환경파괴, 야생동물의 생계 위협 등 여러 문제를 겪고 있습니다. 이러한 문제는 쓰레기를 줍는 드론, 쓰레기 배출 방지 비디오 감시, 인간의 쓰레기 분류를 돕는 AR 기술과 같은 여러 기술을 통해서 조금이나마 개선이 가능합니다.

제공되는 이 데이터셋은 위의 기술을 뒷받침하는 쓰레기를 판별하는 모델을 학습할 수 있게 해줍니다.

### 데이터셋의 간략한 통계

* 전체 이미지 개수 : 4109장

* 12 class : Background, UNKNOWN, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing

    * 참고 : train_all.json/train.json/val.json에는 background에 대한 annotation이 존재하지 않으므로 background (0) class 추가 (baseline 참고)

* 이미지 크기 : (512, 512)

###예제) image, target 시각화 
![img_2.png](files/img_2.png)

### annotation file

annotation file은 coco format 으로 이루어져 있습니다.

coco format은 크게 2가지 (images, annotations)의 정보를 가지고 있습니다.

* images:

    * id: 파일 안에서 image 고유 id, ex) 1

    * height: 512

    * width: 512

    * filename: ex) batch01_vt/002.jpg

* annotations: (참고 : "bbox", "area"는 Segmentation 경진대회에서 활용하지 않습니다.)

    * id: 파일 안에 annotation 고유 id, ex) 1

    * segmentation: masking 되어 있는 고유의 좌표

    * bbox: 객체가 존재하는 박스의 좌표 (xmin, ymin, w, h)

    * area: 객체가 존재하는 영역의 크기

    * category_id: 객체가 해당하는 class의 id

    * image_id: annotation이 표시된 이미지 고유 id

### Segmentation competition에서 사용되는 데이터의 전체 구성
![img_3.png](files/img_3.png)

### 공개 데이터

* batch_01_vt / batch_02_vt / batch_03 → train / validation / test set 들의 이미지들이 섞여서 존재

* tarin_all.json / train.json / val.json / test.json → 아래 표 참고
![img_4.png](files/img_4.png)
즉, 전체 데이터의 80%에 해당되는 이미지는 학습하는데 활용이 되며, 20%는 test dataset 입니다.

teste dataset은 최종적 리더보드 점수에 활용이 되며, 이 중 50%는 public 리더보드 점수에 활용이 됩니다.

제공된 데이터 외의 외부 데이터를 활용하는 것은 금지됩니다. Test 데이터를 학습에 활용하시는 것은 가능합니다.

### 베이스라인 코드 설명
#### train
1. terminal 에서 mmdetection_trash folder로 이동
2. train 명령어  
   python tools/train.py `[config_file]` `[config_file]`: train 시킬 model의 config 파일
   ex) configs/trash/faster_rcnn/faster_rcnn_r50_fpn_1x_trash.py

ex) python tools/train.py configs/trash/faster_rcnn/faster_rcnn_r50_fpn_1x_trash.py

3. 로그 확인  
   tail -f work_dirs/`[config_filename]`/`[2xxx]`.log   
   `[config_filename]`: 실행시킨 config 파일의 이름 ex) faster_rcnn_r50_fpn1x_trash  
   `[2xxxx]`: log가 기록되기 시작한 시각 (폴더 안에서 파일 확인 가능)

ex) tail -f  
work_dirs/faster_rcnn_r50_fpn_1x_trash/2xxx.log

#### inference
1.inference 명령어  
python tools/test.py `[config_file]` `[model_wieght_path]` —out `[output_filepath]`  
`[config_file]`: inference 할 model의 config 파일  
ex) configs/trash/faster_rcnn/faster_rcnn_r50_fpn_1x_trash.py  
`[model_weight_path]`: 저장된 model의 weight [output_filepath]: model이 예측한 output file (.pkl) 을 저장할 경로  

ex) 
```
python tools/test.py configs/trash/faster_rcnn/faster_rcnn_r50_fpn_1x_trash.py \ 
work_dirs/faster_rcnn_r50_fpn_1x_trash/epoch12.pth \
--out work_dirs/faster_rcnn_r50_fpn_1x_trash/epoch12.pkl
```

위의 명령어 입력시 output file이 work_dirs/faster_rcnn_r50_fpn_1x_trash/epoch_12.pkl 에 저장됩니다.

2. make submission 명령어  
   python pkl_to_submission.py --pkl `[output_filepath]` --csv `[submission_filepath]`  
   `[output_filepath]`: 1번에서 저장한 pkl file path  
   `[submission_filepath]`: submission file을 저장할 위치

ex)
```
python pkl_to_submission.py --pkl work_dirs/faster_rcnn_r50_fpn_1x_trash/epoch_12.pkl --csv submission.csv
```
위의 명령어 입력 시 mmdetection_trash 폴더 안에 submission.csv 생성

3. submit!!!!!

### Components

* ``code/saved``

    * 설명 : ``FCN8s baseline (VGG imageNet weight).ipynb`` 에서 학습된 model (fcn8s_best_model(pretrained).pt)이 저장되는 폴더입니다.

* `code/submission/samplesubmission.csv`

    * 설명 : `code/submission` 는 학습된 model로부터 test set에 대한 예측된 값이 `submission.csv` 로 저장되는 폴더입니다.

    * `sample_submission.csv`는 `submission.csv` 을 생성하기 위해 참고하는 파일

* `code/FCN8s baseline (VGG imageNet weight).ipynb`

    * 설명 : baseline을 notebook 기반으로 작성하였으며, 순차적으로 실행하시면 `submission.csv` 을 생성할 수 있습니다.

* code/utils.py

    * 설명 : 평가지표인 mIoU를 계산하는 python script 입니다.

* code/requirements.txt

    * 설명 : baseline을 돌리기 위해 필요한 library 들이 기입되어 있습니다.

### How to install?
1. 데이터 다운로드
    * data 구성
    ![img_5.png](files/img_5.png)
      
2. 라이브러리 설치
    * 아래의 경로를 확인하여 코드 합축을 푼 후 `code` 폴더를 아래와 같은 위치에 구성해주세요.
    ![img_6.png](files/img_6.png)
      
    * `mmdetection_trash`로 들어가서 아래 명령어를 실행해 필요한 라이브러리를 설치해 줍니다.
    

```
conda install pytorch=1.6.0 cudatoolkit=10.1 torchvision -c pytorch

pip install mmcv-full -f <https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html>

pip install -r requirements.txt

pip install -v -e .
```

## 추가한 / 수정한 코드

* `configs/trash/ssd/`,
  `configs/_base_/models/ssd300.py`
    * 설명 : ssd300 모델 실행을 위해 만든 코드.
    

* `configs/trash/yolo/`
    * 설명 : yolo 모델 실행을 위해 만든 코드.
    

* `configs/trash/swin/cascade_mask_rcnn_swin_~~~.py`, 
  `configs/_base_/default_runtime_swin_base_final.py`,
  `configs/_base_/datasets/dataset_swin_final.py`,
  `configs/_base_/models/models/cascade_mask_rcnn_swin_fpn.py`
    * 설명 : cascade_mask_rcnn_swin 을 실행하기 위해 만든 코드.
    

* `configs/swin/hybrid_cascade.py`,
  `configs/trash/dataset_hybrid.py`
    * 설명 : 논문을 비슷하게 재현하기 위해 만든 코드.
    
    
## 실험 결과들
[Object Detection 실험관리(notion)](https://htmlpreview.github.io/?https://github.com/bcaitech1/p3-ims-obd-doggum/blob/main/sanggeon_obd/mmdetection_trash/test%20history.html)

[나의 개인적인 테스트 결과들(wandb)](https://wandb.ai/sglee487/Pstage4_object_detection)  
![img_7.png](files/img_7.png)  

[앙상블을 위한 결과(wandb)](https://wandb.ai/sglee487/Project4_object_detection)  
![img_8.png](files/img_8.png)

## 렙업 레포트
![files/pstage3%20obd%20랩업_Page_1.png](files/pstage3%20obd%20랩업_Page_1.png)  
![files/pstage3%20obd%20랩업_Page_2.png](files/pstage3%20obd%20랩업_Page_2.png)  
![files/pstage3%20obd%20랩업_Page_3.png](files/pstage3%20obd%20랩업_Page_3.png)  
![files/pstage3%20obd%20랩업_Page_4.png](files/pstage3%20obd%20랩업_Page_4.png)  

## 피어세션 발표
![files/P-stage3%20obd%20발표용_Page_1.png](files/P-stage3%20obd%20발표용_Page_1.png)  
![files/P-stage3%20obd%20발표용_Page_2.png](files/P-stage3%20obd%20발표용_Page_2.png)  
![files/P-stage3%20obd%20발표용_Page_3.png](files/P-stage3%20obd%20발표용_Page_3.png)  
![files/P-stage3%20obd%20발표용_Page_4.png](files/P-stage3%20obd%20발표용_Page_4.png)  
![files/P-stage3%20obd%20발표용_Page_5.png](files/P-stage3%20obd%20발표용_Page_5.png)  
![files/P-stage3%20obd%20발표용_Page_6.png](files/P-stage3%20obd%20발표용_Page_6.png)  
![files/P-stage3%20obd%20발표용_Page_7.png](files/P-stage3%20obd%20발표용_Page_7.png)  
![files/P-stage3%20obd%20발표용_Page_8.png](files/P-stage3%20obd%20발표용_Page_8.png)  
![files/P-stage3%20obd%20발표용_Page_9.png](files/P-stage3%20obd%20발표용_Page_9.png)  
![files/P-stage3%20obd%20발표용_Page_10.png](files/P-stage3%20obd%20발표용_Page_10.png)  