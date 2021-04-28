import albumentations as A
from albumentations.pytorch import ToTensorV2

class TestAugmentation:
    def __init__(self, **kwargs):
        self.transform = A.Compose([A.Resize(256, 256)])

    def __call__(self, *args, **kwargs):
        return self.transform(**kwargs)

class CustomAugmentation:
    def __init__(self, **kwargs):
        self.transform = A.Compose([
            ToTensorV2()
        ])

    def __call__(self, **kwargs):
        return self.transform(**kwargs)