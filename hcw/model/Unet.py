import torch
import torch.nn as nn
class UNet(nn.Module):
    def __init__(self, num_classes=12):
        super(UNet, self).__init__()
        
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            return nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          bias=bias),
                                nn.BatchNorm2d(num_features=out_channels),
                                nn.ReLU()
                                )
        
        def ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True):
            return nn.ConvTranspose2d(in_channels=in_channels, 
                                      out_channels=out_channels, 
                                      kernel_size=kernel_size, 
                                      stride=stride, 
                                      padding=padding, 
                                      bias=bias)
        
        
        # Contracting path
        # Encoder 1
        self.enc1_1 = CBR2d(1, 64, 3, 1, 0, bias=True)
        self.enc1_2 = CBR2d(64, 64, 3, 1, 0, bias=True)
        self.pool1 =  nn.MaxPool2d(kernel_size=2)
        
        # Encoder 2
        self.enc2_1 = CBR2d(64, 128, 3, 1, 0, bias=True)
        self.enc2_2 = CBR2d(128, 128, 3, 1, 0, bias=True)
        self.pool2 =  nn.MaxPool2d(kernel_size=2)
        
        # Encoder 3
        self.enc3_1 = CBR2d(128, 256, 3, 1, 0, bias=True)
        self.enc3_2 = CBR2d(256, 256, 3, 1, 0, bias=True)
        self.pool3 =  nn.MaxPool2d(kernel_size=2)
        
        # Encoder 4
        self.enc4_1 = CBR2d(256, 512, 3, 1, 0, bias=True)
        self.enc4_2 = CBR2d(512, 512, 3, 1, 0, bias=True)
        self.pool4 =  nn.MaxPool2d(kernel_size=2)
        
        # Encoder 5 and Decoder 5
        self.enc5_1 = CBR2d(512, 1024, 3, 1, 0, bias=True)
        self.enc5_2 = CBR2d(1024, 1024, 3, 1, 0, bias=True)
        self.upconv4 = ConvTranspose2d(1024, 512, 2, 2, 0, bias=True)
        
        # Decoder 4
        self.dec4_2 = CBR2d(1024, 512, 3, 1, 0, bias=True)
        self.dec4_1 = CBR2d(512, 512, 3, 1, 0, bias=True)
        self.upconv3 = ConvTranspose2d(512, 256, 2, 2, 0, bias=True)
        
        # Decoder 3
        self.dec3_2 = CBR2d(512, 256, 3, 1, 0, bias=True)
        self.dec3_1 = CBR2d(256, 256, 3, 1, 0, bias=True)
        self.upconv2 = ConvTranspose2d(256, 128, 2, 2, 0, bias=True)
        
        # Decoder 2
        self.dec2_2 = CBR2d(256, 128, 3, 1, 0, bias=True)
        self.dec2_1 = CBR2d(128, 128, 3, 1, 0, bias=True)
        self.upconv1 = ConvTranspose2d(128, 64, 2, 2, 0, bias=True)

        
        # Decoder 1
        self.dec1_2 = CBR2d(128, 64, 3, 1, 0, bias=True)
        self.dec1_1 = CBR2d(64, 64, 3, 1, 0, bias=True)
        self.score_fr = nn.Conv2d(in_channels=64, 
                                  out_channels=12,
                                  kernel_size=1,
                                  stride=1,
                                  padding=1,
                                  bias=True)
        

    def crop_img(in_tensor, out_size) :
            """
            Args :
                in_tensor(tensor) : tensor to be cut
                out-size(int) : sie of cut
            """
            dim1, idm2 = in_tensor.size()[2:]
            out_tnesor = in_tensor[:,
                                   :,
                                  int((dim1-out_size)/2):int((dim1+out_size)/2),
                                  int((dim1-out_size)/2):int((dim1+out_size)/2)]
            return out_tensor   
        
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
        
        # Encoder 5 (Decoder 5)
        enc5_1 = self.enc5_1(pool4)
        enc5_2 = self.enc5_2(enc5_1)
        upconv4 = self.upconv4(enc5_2)
        
        # concat
#         crop_enc4_2 = crop_img(enc4_2, upconv4.size()[2])
        cat4 = torch.cat([upconv4, enc4_2], dim=1)
        
        # decoder 4
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        upconv3 = self.upconv3(dec4_1)
        
        # concat
#         crop_enc3_2 = crop_img(enc3_2, upconv3.size()[2])
        cat3 = torch.cat([upconv3, enc3_2], dim=1)
        
        # decoder 3
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        upconv2 = self.upconv2(dec3_1)
        
        # concat
#         crop_enc2_2 = crop_img(enc2_2, upconv2.size()[2])
        cat2 = torch.cat([upconv2, enc2_2], dim=1)
        
        # decoder 2
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        upconv1 = self.upconv1(dec2_1)
        
        # concat
#         crop_enc1_2 = crop_img(enc1_2, upconv1.size()[2])
        cat1 = torch.cat([upconv1, enc1_2], dim=1)
        
        # decoder 1
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        out = self.score_fr(dec1_1)
        
        return out # data가 모든 layer를 거쳐서 나온 output 값
        
