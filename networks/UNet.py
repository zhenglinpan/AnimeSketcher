# _*_ coding: utf-8 _*_
# @Time    :   2021/09/03 13:58:44
# @FileName:   UNet.py
# @Author  :   Boyang
# @Descript:   UNet model
import torch
import torch.nn as nn
from math import sqrt

class UNet(nn.Module):
    
    def __init__(self,chan_in, chan_out, long_skip,nf=32):
        super(UNet, self).__init__()
        self.long_skip = long_skip
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.relu = nn.ReLU()
        self.with_bn = True
        self.conv1_1 = nn.Conv2d(self.chan_in, nf, (3,3),(1,1),(1,1))
        self.bn1_1   = nn.BatchNorm2d(nf)#,affine=False)# input of (n,n,1), output of (n-2,n-2,64)
        self.conv1_2 = nn.Conv2d(nf, nf, 3,1,1)
        self.bn1_2   = nn.BatchNorm2d(nf)#,affine=False)
        self.conv2_1 = nn.Conv2d(nf, nf*2, 3,1,1)
        self.bn2_1   = nn.BatchNorm2d(nf*2)#,affine=False)
        self.conv2_2 = nn.Conv2d(nf*2, nf*2, 3,1,1)
        self.bn2_2   = nn.BatchNorm2d(nf*2)#,affine=False)
        self.conv3_1 = nn.Conv2d(nf*2, nf*4, 3,1,1)
        self.bn3_1   = nn.BatchNorm2d(nf*4)#,affine=False)
        self.conv3_2 = nn.Conv2d(nf*4, nf*4, 3,1,1)
        self.bn3_2   = nn.BatchNorm2d(nf*4)#,affine=False)
        
        self.dc2     =nn.ConvTranspose2d(nf*4, nf*2, 4, stride=2, padding=1,bias=False)

        self.conv4_1 = nn.Conv2d(nf*4, nf*2, 3,1,1)
        self.bn4_1   = nn.BatchNorm2d(nf*2)#,affine=False)
        self.conv4_2 = nn.Conv2d(nf*2, nf*2, 3,1,1)
        self.bn4_2   = nn.BatchNorm2d(nf*2)#,affine=False)
        
        self.dc1     =nn.ConvTranspose2d(nf*2, nf, 4, stride=2, padding=1,bias=False)
        
        self.conv5_1 = nn.Conv2d(nf*2, nf, 3,1,1)
        self.bn5_1   = nn.BatchNorm2d(nf)#,affine=False)
        self.conv5_2 = nn.Conv2d(nf, nf, 3,1,1)
        self.bn5_2   = nn.BatchNorm2d(nf)#,affine=False)
        self.conv5_3 = nn.Conv2d(nf, self.chan_out, 3,1,1)


        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self._initialize_weights()
        print('initialization weights is done')

    def forward(self, x1):
        if self.with_bn:
            x1_ = self.relu(self.bn1_2(self.conv1_2(self.relu(self.bn1_1(self.conv1_1(x1))))))
            x2 = self.relu(self.bn2_2(self.conv2_2(self.relu(self.bn2_1(self.conv2_1(self.maxpool(x1_)))))))
            x3 = self.relu(self.bn3_2(self.conv3_2(self.relu(self.bn3_1(self.conv3_1(self.maxpool(x2)))))))
            x4 = self.relu(self.dc2(x3))  
            x4_2 = torch.cat((x4, x2), 1)
            x5 = self.relu(self.bn4_2(self.conv4_2(self.relu(self.bn4_1(self.conv4_1(x4_2))))))
            x6 = self.relu(self.dc1(x5))  
            x6_1 = torch.cat((x6, x1_), 1)
            x7 = self.relu(self.bn5_2(self.conv5_2(self.relu(self.bn5_1(self.conv5_1(x6_1))))))
        else:
            x1_ = self.relu(self.conv1_2(self.relu(self.conv1_1(x1))))
            x2 = self.relu(self.conv2_2(self.relu(self.conv2_1(self.maxpool(x1_)))))
            x3 = self.relu(self.conv3_2(self.relu(self.conv3_1(self.maxpool(x2)))))
            x4 = self.relu(self.dc2(x3))  
            x4_2 = torch.cat((x4, x2), 1)
            x5 = self.relu(self.conv4_2(self.relu(self.conv4_1(x4_2))))
            x6 = self.relu(self.dc1(x5))  
            x6_1 = torch.cat((x6, x1_), 1)
            x7 = self.relu(self.conv5_2(self.relu(self.conv5_1(x6_1))))
        x8 = self.conv5_3(x7)
        if self.long_skip == True:        
            return x8 + x1[:,0:self.chan_out,:,:]
        else:
            return x8


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                # m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                # m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()