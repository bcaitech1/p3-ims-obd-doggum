import albumentations as A
from albumentations.pytorch import ToTensorV2

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