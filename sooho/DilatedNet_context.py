#DilatedNet(Front+Basic Context module) 
import torch
import torch.nn as nn
class VGG16(nn.Module):
    def __init__(self, num_classes=12):
        super(VGG16, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
       
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.features(x)
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
    def __init__(self, backbone, classifier, context_module):
        super(DilatedNetFront, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.context_module = context_module
        self.deconv = nn.ConvTranspose2d(in_channels=12, out_channels=12, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        x = self.context_module(x)
        x = self.deconv(x)

        return x

class BasicContextModule(nn.Module):
    def __init__(self, num_classes):
        super(BasicContextModule, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=1, padding=16, dilation=16),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU()
        )
        # no Truncation 여기 padding=1하면 528로 사이즈 커져서 0으로 넣어줘야 하는 것 같습니다.
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1, stride=1)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x
      
# 구현된 model에 임의의 input을 넣어 output이 잘 나오는지 test
# DilatedNetFront+Basic Context module
# model 생성
num_classes = 12
model = VGG16(num_classes=12)

#weight 불러오기
model_dict = model.state_dict()
pretrained_model = vgg16(pretrained = True)
pretrained_model_dict = pretrained_model.state_dict()

key = ['features.24.weight', 'features.24.bias', 'features.26.weight','features.26.bias','features.28.weight','features.28.bias']
new_pretrained_model_dict = {}
for k, v in pretrained_model_dict.items():
    if k in key:
        if k.split('.')[-1] == 'weight':
            number = int(k.split('.')[1])-1
            k = f'features.{number}.weight'
        else:
            number = int(k.split('.')[1])-1
            k = f'features.{number}.bias'
    new_pretrained_model_dict[k] = v
pretrained_model_dict = {k: v for k, v in new_pretrained_model_dict.items() if k in model_dict}
model_dict.update(pretrained_model_dict)
model.load_state_dict(model_dict)

# classifier
classifier = DilatedNet_classifier(num_classes)

# context_module
context_module = BasicContextModule(num_classes)

# DilatedNetFront
model = DilatedNetFront(model, classifier, context_module)

# 임의의 x값 넣어서 shape 확인
x = torch.randn([1, 3, 512, 512])
print("input shape : ", x.shape)
out = model(x)
print("output shape : ", out.size())

model = model.to(device)
