#!C:\ProgramData\Anaconda3
# coding: utf-8
# ST-CGANの論文そのまま
import math
import torch
import torch.nn as nn

# Light Estimation network for 3ch images + Masked 3ch images
class LightEstimator(nn.Module):
    def __init__(self, len_ch=3, len_features=64):
        super(LightEstimator, self).__init__()
        self.conv0 = Cvi(len_ch, len_features)
        self.conv1 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            Cvi(len_features, len_features*2),
            nn.BatchNorm2d(len_features*2),
        )
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            Cvi(len_features*2, len_features*4),
            nn.BatchNorm2d(len_features*4),
        )
        self.conv3 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            Cvi(len_features*4, len_features*8),
            nn.BatchNorm2d(len_features*8),
        )
        self.conv4 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            Cvi(len_features*8, len_features*8),
            nn.BatchNorm2d(len_features*8),
        )
        self.conv5 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            Cvi(len_features*8, len_features*8),
        )
        self.deconv5 = nn.Sequential(
            nn.ReLU(inplace=True),
            CvTi(len_features*8, len_features*8),
            nn.BatchNorm2d(len_features*8),
        )
        self.deconv4 = nn.Sequential(
            nn.ReLU(inplace=True),
            CvTi(len_features*16, len_features*8),
            nn.BatchNorm2d(len_features*8),
        )
        self.deconv3 = nn.Sequential(
            nn.ReLU(inplace=True),
            CvTi(len_features*16, len_features*4),
            nn.BatchNorm2d(len_features*4),
        )
        self.deconv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            CvTi(len_features*8, len_features*2),
            nn.BatchNorm2d(len_features*2),
        )
        self.deconv1 = nn.Sequential(
            nn.ReLU(inplace=True),
            CvTi(len_features*4, len_features),
            nn.BatchNorm2d(len_features),
        )
        self.deconv0 = nn.Sequential(
            nn.ReLU(inplace=True),
            CvTi(len_features*2, 3),
            # nn.Tanh(),
            nn.ReLU(inplace=True)
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(3,3,3,padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(3,3,3,padding="same"),
            nn.ReLU(inplace=True),
        )
        
    ### end __init__

    def forward(self, inputs):
        x0 = self.conv0(inputs)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        y5 = self.deconv5(x5)
        y4 = self.deconv4(torch.cat([x4, y5], dim=1))
        y3 = self.deconv3(torch.cat([x3, y4], dim=1))
        y2 = self.deconv2(torch.cat([x2, y3], dim=1))
        y1 = self.deconv1(torch.cat([x1, y2], dim=1))
        outputs = self.deconv0(torch.cat([x0, y1], dim=1))
        outputs2 = self.conv2d(outputs)
        return outputs2
    ### end forward
### end class

# Convolutional layer with weight initilization
class Cvi(nn.Module):
    def __init__(
            self, in_ch, out_ch,
            kernel_size=4, stride=2, padding=1, dilation=1,
            groups=1, bias=False
        ):
        super(Cvi, self).__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias
        )
        self.conv.apply(weight_init('gaussian'))
    ### end __init__

    def forward(self, inputs):
        return self.conv(inputs)
    ### end forward
### end class

# Convolutional layer with weight initilization
class CvTi(nn.Module):
    def __init__(
            self, in_ch, out_ch,
            kernel_size=4, stride=2, padding=1, dilation=1,
            groups=1, bias=False
        ):
        super(CvTi, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size, stride, padding,
            dilation=dilation, groups=groups, bias=bias
        )
        self.deconv.apply(weight_init('gaussian'))
    ### end __init__

    def forward(self, inputs):
        return self.deconv(inputs)
    ### end forward
### end class

# Weight initilization function
def weight_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    ### end def

    return init_fun
### end def