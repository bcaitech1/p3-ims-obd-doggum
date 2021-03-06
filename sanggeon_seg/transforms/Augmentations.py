import os
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import cv2
import torch
from glob import glob


invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])


class TestAugmentation:
    def __init__(self, **kwargs):
        self.transform = A.Compose([
            A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
            ToTensorV2(),
        ])

    def __call__(self, *args, **kwargs):
        return self.transform(**kwargs)


class CustomAugmentation:
    def __init__(self, **kwargs):
        self.transform = A.Compose([
            A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
            ToTensorV2()
        ])

    def __call__(self, **kwargs):
        return self.transform(**kwargs)


class CustomAugmentation2:
    def __init__(self, **kwargs):
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
            ToTensorV2()
        ])

    def __call__(self, **kwargs):
        return self.transform(**kwargs)


class CannyAugmentation:
    def __init__(self, is_test:bool=False, test_size=256):
        if is_test:
            self.transform = A.Compose([
                A.Resize(test_size, test_size)
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
                ToTensorV2()
            ])

    def __call__(self, **kwargs):
        image = kwargs['image']
        image_canny = cv2.cvtColor(image, code=cv2.IMREAD_GRAYSCALE)
        image_canny = cv2.Canny(image_canny, 100, 200)
        images_canny = cv2.cvtColor(image_canny, cv2.COLOR_GRAY2RGB)
        output = (0.1 * images_canny) + (0.9 * image)
        kwargs['image'] = output
        return self.transform(**kwargs)


class CustomAugmentation3:
    def __init__(self, mode:str='train',  **kwargs):
        if mode == 'train':
            self.transform = A.Compose([
                A.CropNonEmptyMaskIfExists(height=256, width=256, p=0.5),
                A.Resize(height=512, width=512, p=1.0),
                A.CLAHE(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246),
                            max_pixel_value=255.0, p=1.0),
                ToTensorV2(transpose_mask=True)
            ])
        elif mode in ('val', 'test'):
            self.transform = A.Compose([
                A.Resize(height=512, width=512, p=1.0),
                A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246),
                            max_pixel_value=255.0, p=1.0),
                ToTensorV2(transpose_mask=True)
            ])


    def __call__(self, **kwargs):
        return self.transform(**kwargs)


class CustomAugmentation4:
    def __init__(self, mode:str='train',  **kwargs):
        if mode == 'train':
            self.transform = A.Compose([
                A.CropNonEmptyMaskIfExists(height=256, width=256, p=0.5),
                A.Resize(height=512, width=512, p=1.0),
                A.CLAHE(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                                     val_shift_limit=20,
                                     p=0.5),
                A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246),
                            max_pixel_value=255.0, p=1.0),
                ToTensorV2(transpose_mask=True)
            ])
        elif mode in ('val', 'test'):
            self.transform = A.Compose([
                A.Resize(height=512, width=512, p=1.0),
                A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246),
                            max_pixel_value=255.0, p=1.0),
                ToTensorV2(transpose_mask=True)
            ])


    def __call__(self, **kwargs):
        return self.transform(**kwargs)


class CustomAugmentation5:
    def __init__(self, mode:str='train',  **kwargs):
        if mode == 'train':
            self.transform = A.Compose([
                A.CropNonEmptyMaskIfExists(height=256, width=256, p=0.5),
                A.Resize(height=512, width=512, p=1.0),
                A.CLAHE(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30),
                A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246),
                            max_pixel_value=255.0, p=1.0),
                ToTensorV2(transpose_mask=True)
            ])
        elif mode in ('val', 'test'):
            self.transform = A.Compose([
                A.Resize(height=512, width=512, p=1.0),
                A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246),
                            max_pixel_value=255.0, p=1.0),
                ToTensorV2(transpose_mask=True)
            ])


    def __call__(self, **kwargs):
        return self.transform(**kwargs)


class CustomAugmentation6:
    def __init__(self, mode:str='train',  **kwargs):
        if mode == 'train':
            self.transform = A.Compose([
                A.Rotate(limit=30),
                A.CropNonEmptyMaskIfExists(height=256, width=256, p=0.5),
                A.Resize(height=512, width=512, p=1.0),
                A.CLAHE(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246),
                            max_pixel_value=255.0, p=1.0),
                ToTensorV2(transpose_mask=True)
            ])
        elif mode in ('val', 'test'):
            self.transform = A.Compose([
                A.Resize(height=512, width=512, p=1.0),
                A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246),
                            max_pixel_value=255.0, p=1.0),
                ToTensorV2(transpose_mask=True)
            ])


    def __call__(self, **kwargs):
        return self.transform(**kwargs)


class CustomAugmentation7:
    def __init__(self, mode:str='train',  **kwargs):
        self.transform = A.Compose([
            # A.Rotate(p=1.0, limit=(-90,90)),
            # A.VerticalFlip(p=0.5),
            # A.HorizontalFlip(p=0.8),
            ToTensorV2(transpose_mask=True)
        ])


    def __call__(self, **kwargs):
        return self.transform(**kwargs)


class CustomAugmentation8:
    def __init__(self, mode:str='train'):
        self.transform = A.Compose([
            A.Rotate(p=1.0, limit=(-90,90)),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.8),
            ToTensorV2(transpose_mask=True)
        ])
        self.backgrounds = glob(os.path.join('../input/data', 'results_splits/') + '*.jpg')
        self.backgrounds_len = len(self.backgrounds)


    def __call__(self, image, mask):
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        bgint = random.randint(0,self.backgrounds_len-1)
        background = cv2.cvtColor(cv2.imread(self.backgrounds[bgint]), cv2.COLOR_BGR2RGB)
        background = self.transform(image=background)['image']
        other_background = torch.where(mask > 0, image, background)
        return {'image': torch.tensor(other_background,dtype=torch.uint8), 'mask': mask}


class CustomAugmentation9:
    def __init__(self, mode:str='train',  **kwargs):
        self.transform = A.Compose([
            # A.Rotate(p=1.0, limit=(-90,90)),
            # A.VerticalFlip(p=0.5),
            # A.HorizontalFlip(p=0.8),
            A.ShiftScaleRotate(p=1.0, shift_limit=(-0.0,0.0), scale_limit=(0.05,0.05), rotate_limit=(0,0)),
            ToTensorV2(transpose_mask=True)
        ])


    def __call__(self, **kwargs):
        return self.transform(**kwargs)