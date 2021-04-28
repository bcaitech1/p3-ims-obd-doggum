import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomAugmentation:
    def __init__(self, **kwargs):
        self.transform = A.Compose([
            ToTensorV2()
        ])

    def __call__(self, **kwargs):
        return self.transform(**kwargs)