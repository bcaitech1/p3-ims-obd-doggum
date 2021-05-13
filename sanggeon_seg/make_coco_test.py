#%%

import time
import argparse
import random
import os
import warnings
from importlib import import_module
import cv2


warnings.filterwarnings('ignore')

import torch
import numpy as np
from tqdm import tqdm

from datasets.data_loader import setup_loader
from datasets.data_loader import CustomDataLoader
from config import Config
import matplotlib.pyplot as plt

#%%

config = Config(
    lr=0.0001,
    epochs=20,
    batch_size=8,
    seed=21,
    eval=False,
    augmentation='CustomAugmentation7',
    criterion='cross_entropy',
    optimizer='adam',
    model='unetmnv2',
    continue_load='',
    eval_load='',
    dataset_path='../input/data')

#%%

print(config)

print('pytorch version: {}'.format(torch.__version__))
print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())

device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 사용 가능 여부에 따라 device 정보 저장

sorted_df = setup_loader(config)

# train.json / validation.json / test.json 디렉토리 설정
# train_path = config.dataset_path + '/train.json'
train_path = 'split_sample.json'

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))
#%%
# import json
# with open(train_path, "r") as st_json:
#
#     st_python = json.load(st_json)
# st_python
#%%

augmentation_module = getattr(import_module("transforms.Augmentations"), config.augmentation)
train_transform = augmentation_module(mode='train')
train_dataset = CustomDataLoader(data_dir=train_path, sorted_df=sorted_df, mode='train', transform=train_transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=config.batch_size,
                                          num_workers=2,
                                          collate_fn=collate_fn)

#%%

print('Start extracting..')
for images, masks, informs in iter(train_loader):
    images = torch.stack(images)  # (batch, channel, height, width)
    masks = torch.stack(masks).long()  # (batch, channel, height, width)
    B = images.shape[0]
    for i in range(B):
        image = images[i]
        mask = masks[i]
        inform = informs[i]
        print(image.shape)
        print(mask.shape)
        print(inform)
        plt.imshow(mask)
        break
    break

#%%


