import torch
import torch.nn as nn
class UNet(nn.Module):
    """
    add zero padding
    """
    def __init__(self, num_classes=12):
        super(UNet, self).__init__()
        self.enc1_1 = self.CBR2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc1_2 = self.CBR2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = self.CBR2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc2_2 = self.CBR2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = self.CBR2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc3_2 = self.CBR2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = self.CBR2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc4_2 = self.CBR2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = self.CBR2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc5_2 = self.CBR2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.upconv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = self.CBR2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec4_1 = self.CBR2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.upconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = self.CBR2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec3_1 = self.CBR2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = self.CBR2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec2_1 = self.CBR2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.upconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = self.CBR2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec1_1 = self.CBR2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.score_fr = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True)           

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        enc5_2 = self.enc5_2(enc5_1)
        upconv4 = self.upconv4(enc5_2)

        # crop_enc4_2 = self.crop_img(enc4_2, upconv4.size()[2])
        # cat4 = torch.cat([upconv4, crop_enc4_2], dim=1)
        cat4 = torch.cat([upconv4, enc4_2], dim=1)


        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        upconv3 = self.upconv3(dec4_1)

        # crop_enc3_2 = self.crop_img(enc3_2, upconv3.size()[2])
        # cat3 = torch.cat([upconv3, crop_enc3_2], dim=1)
        cat3 = torch.cat([upconv3, enc3_2], dim=1)


        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        upconv2 = self.upconv2(dec3_1)

        # crop_enc2_2 = self.crop_img(enc2_2, upconv2.size()[2])
        # cat2 = torch.cat([upconv2, crop_enc2_2], dim=1)
        cat2 = torch.cat([upconv2, enc2_2], dim=1)


        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        upconv1 = self.upconv1(dec2_1)

        # crop_enc1_2 = self.crop_img(enc1_2, upconv1.size()[2])
        # cat1 = torch.cat([upconv1, crop_enc1_2], dim=1)
        cat1 = torch.cat([upconv1, enc1_2], dim=1)

        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        x = self.score_fr(dec1_1)

        return x

    def CBR2d(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                        kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
            )
        
    def crop_img(self, in_tensor, out_size):
        dim1, dim2 = in_tensor.size()[2:]
        out_tensor = in_tensor[:,
                                :,
                                int((dim1-out_size)/2):int((dim1+out_size)/2),
                                int((dim2-out_size)/2):int((dim2+out_size)/2),
                                ]
        return out_tensor

# 구현된 model에 임의의 input을 넣어 output이 잘 나오는지 test

model = UNet(num_classes=12)
x = torch.randn([1, 3, 512, 512])
print("input shape : ", x.shape)
out = model(x)
print("output shape : ", out.size())