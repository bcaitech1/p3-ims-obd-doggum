import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


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