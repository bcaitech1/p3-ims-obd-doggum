import torch
import torch.nn as nn
class FCN8s(nn.Module):
    """
    FCN8s fc6 layer : kernel_size = 7 conv layer
    Base Model : VGG16
    """
    def __init__(self, num_classes=12):
        super(FCN8s, self).__init__())
        
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
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool1(x)
        
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.pool3(x)
        pool3 = x

        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.pool4(x)
        pool4 = x

        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.pool5(x)

        x = self.relu(self.fc6(x))
        x = self.drop(x)

        x = self.relu(self.fc7(x))
        x = self.drop(x)

        x = self.score_fr(x)
        x = self.upscore2(x)
        upscore2 = x

        x = self.score_pool4(pool4 * 0.01)
        x = x[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = x

        x = upscore2 + score_pool4c
        x = self.upscore_pool4(x)
        upscore_pool4 = x

        x = self.score_pool3(pool3 * 0.0001)
        x = x[:, :,
              9:9+upscore_pool4.size()[2],
              9:9+upscore_pool4.size()[3]]
        score_pool3c = x

        x = upscore_pool4 + score_pool3c

        x = sef.upscore8(x)
        x = x[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
      
                                  
