import torch
import torch.nn as nn
from torch.nn import functional as F

def conv3x3_relu(in_ch, out_ch, rate=1):
    conv3x3_relu = nn.Sequential(nn.Conv2d(in_ch, 
                                           out_ch, 
                                           kernel_size=3,
                                           stride=1,
                                           padding=rate,
                                           dilation=rate),
                                 nn.ReLU())
    return conv3x3_relu


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        
        self.conv1_1 = conv3x3_relu(3, 64)
        self.conv1_2 = conv3x3_relu(64, 64)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.conv2_1 = conv3x3_relu(64, 128)
        self.conv2_2 = conv3x3_relu(128, 128)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.conv3_1 = conv3x3_relu(128, 256)
        self.conv3_2 = conv3x3_relu(256, 256)
        self.conv3_3 = conv3x3_relu(256, 256)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.conv4_1 = conv3x3_relu(256, 512)
        self.conv4_2 = conv3x3_relu(512, 512)
        self.conv4_3 = conv3x3_relu(512, 512)
        self.maxpool4 = nn.MaxPool2d(3, stride=1, padding=1)
        
        self.conv5_1 = conv3x3_relu(512, 512, rate=2)
        self.conv5_2 = conv3x3_relu(512, 512, rate=2)
        self.conv5_3 = conv3x3_relu(512, 512, rate=2)
        self.maxpool5 = nn.MaxPool2d(3, stride=1, padding=1)

        
    def forward(self, x):
        '''
        [TODO]

        '''
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.maxpool1(x)
        
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.maxpool2(x)
        
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.maxpool3(x)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.maxpool4(x)
        
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        out = self.maxpool5(x)
        
        return out

    
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=1024, num_classes=21):
        super(ASPP, self).__init__()
        '''
        [TODO]

        ''' 
        self.fc6_6 = conv3x3_relu(in_channels, in_channels, rate=6)
        self.fc6_12 = conv3x3_relu(in_channels, in_channels, rate=12)
        self.fc6_18 = conv3x3_relu(in_channels, in_channels, rate=18)
        self.fc6_24 = conv3x3_relu(in_channels, in_channels, rate=24)
        
        self.fc7 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.fc8 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.out_1 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.out_2 = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        '''
        [TODO]

        '''
        x1 = self.fc6_6(x)
        x1 = self.fc7(x)
        x1 = self.fc8(x)
        
        x2 = self.fc6_12(x)
        x2 = self.fc7(x)
        x2 = self.fc8(x)
        
        x3 = self.fc6_18(x)
        x3 = self.fc7(x)
        x3 = self.fc8(x)
        
        x4 = self.fc6_24(x)
        x4 = self.fc7(x)
        x4 = self.fc8(x)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        x = self.out_1(x)
        out = self.out_2(x)

        return out

    
class DeepLabV2(nn.Module):
    ## VGG 위에 ASPP 쌓기
    def __init__(self, backbone, classifier, upsampling=8):
        super(DeepLabV2, self).__init__()
        '''
        [TODO]

        ''' 
        self.backbone = backbone
        self.classifier = classifier
        self.upsampling = upsampling

    def forward(self, x):
        '''
        [TODO]

        '''
        x = self.backbone(x)
        x = self.classifier(x)
        
        _, _, feature_map_h, feature_map_w = x.size()
        
        out = F.interpolate(x, size=(feature_map_h * self.upsampling, feature_map_w * self.upsampling), mode="bilinear")
        
        return out


# 구현된 model에 임의의 input을 넣어 output이 잘 나오는지 test
backbone = VGG16()
aspp_module = ASPP(in_channels=512, out_channels=256, num_classes=12)
model = DeepLabV2(backbone=backbone, classifier=aspp_module)

x = torch.randn([1, 3, 512, 512])
print("input shape : ", x.shape)
out = model(x).to(device)
print("output shape : ", out.size())

model = model.to(device)

