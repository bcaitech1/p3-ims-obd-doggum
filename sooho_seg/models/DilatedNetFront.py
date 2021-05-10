import torch.nn as nn
from torchvision.models import vgg16
# vgg16 pretrained_model load
pretrained_model = vgg16(pretrained = True)
# cuda 설정
device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU 사용 가능 여부에 따라 device 정보 저장

class VGG16(nn.Module):
    def __init__(self, num_classes=12):
        super(VGG16, self).__init__()
        
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.features3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.features4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU()
        )

        self.features5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.features1(x)
        print(x.shape)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        return x

class DilatedNet_classifier(nn.Module):
    def __init__(self, num_classes=12):
        super(DilatedNet_classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, dilation=4, padding=12),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )
    def forward(self, x):
        out = self.classifier(x)
        return out

class DilatedNetFront(nn.Module):
    def __init__(self, backbone, classifier):
        super(DilatedNetFront, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.deconv = nn.ConvTranspose2d(in_channels=12, out_channels=12, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        x = self.deconv(x)

        return x
      
# 구현된 model에 임의의 input을 넣어 output이 잘 나오는지 test

# model 생성
model = VGG16(num_classes=12)

# weight 불러오기와서 model에 weight를 입력
model_dict = model.state_dict()
pretrained_model = vgg16(pretrained = True)
pretrained_dict = pretrained_model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# classifier
classifier = DilatedNet_classifier()

# DilatedNetFront
model = DilatedNetFront(model, classifier)

# 임의의 x값 넣어서 shape 확인
x = torch.randn([1, 3, 512, 512])
print("input shape : ", x.shape)
out = model(x)
print("output shape : ", out.size())

model = model.to(device)
