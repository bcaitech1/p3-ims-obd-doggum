import torch
import torch.nn as nn

from torchvision.models import vgg16

class SegNetVgg16(nn.Module):
    def __init__(self, num_classes):
        super(SegNetVgg16,self).__init__()

        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):

            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding),
                            nn.BatchNorm2d(num_features=out_channels),
                            nn.ReLU()
                            )

        self.vgg16_ori = vgg16(pretrained=True)
        features, classifiers = list(self.vgg16_ori.features.children()), list(self.vgg16_ori.classifier.children())

        self.features_map1 = nn.Sequential(*features[0:4])
        self.features_map2 = nn.Sequential(*features[5:9])
        self.features_map3 = nn.Sequential(*features[10:16])
        self.features_map4 = nn.Sequential(*features[17:23])
        self.features_map5 = nn.Sequential(*features[24:30])

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)

        # deconv5
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.dcbr5_3 = CBR(512, 512, 3, 1, 1)
        self.dcbr5_2 = CBR(512, 512, 3, 1, 1)
        self.dcbr5_1 = CBR(512, 512, 3, 1, 1)

        # deconv4
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.dcbr4_3 = CBR(512, 512, 3, 1, 1)
        self.dcbr4_2 = CBR(512, 512, 3, 1, 1)
        self.dcbr4_1 = CBR(512, 256, 3, 1, 1)

        # deconv3
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.dcbr3_3 = CBR(256, 256, 3, 1, 1)
        self.dcbr3_2 = CBR(256, 256, 3, 1, 1)
        self.dcbr3_1 = CBR(256, 128, 3, 1, 1)

        # deconv2
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.dcbr2_2 = CBR(128, 128, 3, 1, 1)
        self.dcbr2_1 = CBR(128, 64, 3, 1, 1)

        # deconv1
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv1_1 = CBR(64, 64, 3, 1, 1)

        # Score
        self.score_fr = nn.Conv2d(64, num_classes, kernel_size=3,
                                  stride=1, padding=1, dilation=1)

    def forward(self, x):
        '''
        [TODO]
        '''
        h = self.features_map1(x)
        h, pool1_indices = self.pool1(h)

        h = self.features_map2(h)
        h, pool2_indices = self.pool2(h)

        h = self.features_map3(h)
        h, pool3_indices = self.pool3(h)

        h = self.features_map4(h)
        h, pool4_indices = self.pool4(h)

        h = self.features_map5(h)
        h, pool5_indices = self.pool5(h)

        h = self.unpool5(h, pool5_indices)
        h = self.dcbr5_3(h)
        h = self.dcbr5_2(h)
        h = self.dcbr5_1(h)

        h = self.unpool4(h, pool4_indices)
        h = self.dcbr4_3(h)
        h = self.dcbr4_2(h)
        h = self.dcbr4_1(h)

        h = self.unpool3(h, pool3_indices)
        h = self.dcbr3_3(h)
        h = self.dcbr3_2(h)
        h = self.dcbr3_1(h)

        h = self.unpool2(h, pool2_indices)
        h = self.dcbr2_2(h)
        h = self.dcbr2_1(h)

        h = self.unpool1(h, pool1_indices)
        h = self.deconv1_1(h)

        h = self.score_fr(h)

        return h
