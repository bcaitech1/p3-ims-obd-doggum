import torch
import torch.nn as nn
class FCN8s(nn.Module):
    """
    FCN8s fc6 layer : kernel_size = 7 conv layer
    Base Model : VGG16
    """
    def __init__(self, num_classes=12):
        super(FCN8s, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d()
        
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1) 
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        
        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)
        
    def forward(self, x):
        h = x
        h = self.relu(self.conv1_1(h))
        h = self.relu(self.conv1_2(h))
        h = self.pool1(h)
        
        h = self.relu(self.conv2_1(h))
        h = self.relu(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3_1(h))
        h = self.relu(self.conv3_2(h))
        h = self.relu(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h

        h = self.relu(self.conv4_1(h))
        h = self.relu(self.conv4_2(h))
        h = self.relu(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h

        h = self.relu(self.conv5_1(h))
        h = self.relu(self.conv5_2(h))
        h = self.relu(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu(self.fc6(h))
        h = self.drop(h)

        h = self.relu(self.fc7(h))
        h = self.drop(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h
        h = upscore2 + score_pool4c
        h = self.upscore_pool4(h)
        upscore_pool4 = h

        h = self.score_pool3(pool3)
        h = h[:, :, 9:9+upscore_pool4.size()[2], 9:9+upscore_pool4.size()[3]]
        score_pool3c = h

        h = upscore_pool4 + score_pool3c

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        
        return h
      
                                  
