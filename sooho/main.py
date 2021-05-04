# 작성중

import os
import random
import time
import json
import warnings 
import argparse
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2

import numpy as np
import pandas as pd

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from train import train, validation
from inference import test
from dataloader import CustomDataLoader
from utils import collate_fn

def set_seed(random_seed):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def load_all_dataset(collate_fn, args):
    dataset_path = '/opt/ml/input/data'
    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'
    test_path = dataset_path + '/test.json'

    train_transform = A.Compose([
        A.CropNonEmptyMaskIfExists(height=256, width=256, p=0.5),
        A.Resize(height=512, width=512, p=1.0),
        A.CLAHE(p=0.5),
        A.HorizontalFlip(p=0.5),                         
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225), 
            max_pixel_value=255.0, p=1.0
        )
        ToTensorV2(transpose_mask=True)
    ])
    val_transform = A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225), 
            max_pixel_value=255.0, p=1.0
        )
        ToTensorV2(transpose_mask=True)
    ])
    train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform)
    # validation dataset
    val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=val_transform)
    # test dataset
    test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=args['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args['batch_size'],
        num_workers=4,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader

def load_test_dataset(collate_fn, args):
    dataset_path = '/opt/ml/input/data'
    test_path = dataset_path + '/test.json'
    val_transform = A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225), 
            max_pixel_value=255.0, p=1.0
        )
        ToTensorV2(transpose_mask=True)
    ])
    test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=val_transform)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args['batch_size'],
        num_workers=4,
        collate_fn=collate_fn
    )
    return test_loader

def load_model_submission(test_loader, device, args):
    save_file_name = args['file_name']
    model_path = '/opt/ml/code/saved/' + args['model_path'] + f'{save_file_name}.pth'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # sample_submisson.csv 열기
    submission = pd.read_csv('/opt/ml/code/submission/sample_submission.csv', index_col=None)

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(f"/opt/ml/code/submission/{save_file_name}.csv", index=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", default='smp_deeplabv3+_mobilenet_v2_54epoch_CropNonEmptyMaskIfExists_CLAHE_HorizontalFlip_Resize', type=str, help='model.pth file directory')
    parser.add_argument("--model_path", default='augmentation test/CropNonEmptyMaskIfExists_CLAHE_HorizontalFlip_Resize/', type=str, help='input your model_path after saved directory')
    parser.add_argument("--mode", default='train', type=str, help='select mode')
    parser.add_argument("--seed", default=42, type=int, help="insert your seed number")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--encoder_name", default='efficientnet-b7', type=str, help="backbone model")
    parser.add_argument("--num_classes", default=12, type=int, help="the number of class")
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--num_epochs", default=30, type=int)
    args = vars(parser.parse_args())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.DeepLabV3Plus(encoder_name=args["encoder_name"], encoder_weights="imagenet", in_channels=3, classes=args["num_classes"])
    # test
    x = torch.randn([2, 3, 512, 512])
    model = model.to(device)
    if args['mode'] == 'train':
        if not os.path.isdir(saved_dir):                                               
            os.mkdir(saved_dir)
        val_every = 1
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params = model.parameters(), lr = args["learning_rate"], weight_decay=1e-6)
        num_epochs = args["num_epochs"]
        train_loader, val_loader, test_loader = load_all_dataset(collate_fn, args)
        train(num_epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, val_every, device)

    if args['mode'] == 'test':
        test_loader = load_test_dataset(collate_fn, args)
    load_model_submission(test_loader, device, args)