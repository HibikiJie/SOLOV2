from torch import nn
from torch.nn import Module
import torch
import torch.nn.functional as func


class Swish(Module):
    """Swish激活函数"""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, input_):
        return input_ * torch.sigmoid(input_)


class Mish(Module):

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, input_):
        return input_ * torch.tanh(func.softplus(input_))


class ChannelAttention(Module):
    """通道注意力"""

    def __init__(self, num_channels, r=8):
        """
        注意力
        :param num_channels: 通道数
        :param r: 下采样倍率
        """
        super(ChannelAttention, self).__init__()
        self.num_channels = num_channels
        self.layer = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_channels // r, 1),
            Swish(),
            nn.Conv2d(self.num_channels // r, self.num_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, input_):
        return self.layer(torch.mean(input_, dim=[2, 3], keepdim=True)) * input_


class SpatialAttention(Module):
    """空间注意力"""

    def __init__(self, kernel_size=7):
        """
        注意力
        :param num_channels: 通道数
        :param r: 下采样倍率
        """
        super(SpatialAttention, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_):
        avg_out = torch.mean(input_, dim=1, keepdim=True)
        max_out, _ = torch.max(input_, dim=1, keepdim=True)
        return self.layer(torch.cat([avg_out, max_out], dim=1)) * input_


class ConvolutionLayer(Module):
    """卷积、批归一化、激活"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, action=Mish()):
        """
        :param in_channels: 输入通道
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 步长
        :param padding:填充
        :param bias:偏置
        :param action:激活函数
        """
        super(ConvolutionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            action
        )

    def forward(self, input_):
        return self.layer(input_)


class Pool(Module):
    '''多尺度池化，借鉴Inception网络结构'''

    def __init__(self):
        super(Pool, self).__init__()
        self.max1 = nn.MaxPool2d(5, 1, 2)
        self.max2 = nn.MaxPool2d(9, 1, 4)
        self.max3 = nn.MaxPool2d(13, 1, 6)

    def forward(self, input_):
        return torch.cat((self.max1(input_), self.max2(input_), self.max3(input_), input_), dim=1)


class ResBlock(Module):
    """残差块,带注意力"""

    def __init__(self, channels, r=4):
        """
        :param channels: 通道数
        :param r: 通道缩减倍数，默认：2。
        """
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            ConvolutionLayer(channels, channels // r),
            ChannelAttention(channels // r),
            SpatialAttention(),
            ConvolutionLayer(channels // r, channels)
        )

    def forward(self, input_):
        return self.layer(input_) + input_


class Block(Module):

    def __init__(self, in_channels, out_channels, num_res):
        super(Block, self).__init__()
        res = []
        for i in range(num_res):
            res.append(ResBlock(in_channels))
        self.c1 = ConvolutionLayer(in_channels, out_channels, 3, 2)
        self.c2 = nn.Sequential(
            ConvolutionLayer(out_channels, in_channels, kernel_size=1, padding=0),
            *res,
            ConvolutionLayer(in_channels, in_channels, kernel_size=1, padding=0)
        )
        self.c3 = ConvolutionLayer(out_channels, in_channels, kernel_size=1, padding=0)
        self.c4 = ConvolutionLayer(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, input_):
        input_ = self.c1(input_)
        input_ = torch.cat((self.c2(input_), self.c3(input_)), dim=1)
        return self.c4(input_)


class FPN(Module):

    def __init__(self, f_channels):
        super(FPN, self).__init__()
        self.c1 = ConvolutionLayer(f_channels, f_channels // 2, kernel_size=1, padding=0, action=nn.LeakyReLU(0.1))
        self.c2 = ConvolutionLayer(f_channels, f_channels // 2, kernel_size=1, padding=0, action=nn.LeakyReLU(0.1))
        self.c3 = nn.Sequential(
            ConvolutionLayer(f_channels, f_channels // 2, kernel_size=1, padding=0, action=nn.LeakyReLU(0.1)),
            ConvolutionLayer(f_channels // 2, f_channels, action=nn.LeakyReLU(0.1)),
            ConvolutionLayer(f_channels, f_channels // 2, kernel_size=1, padding=0, action=nn.LeakyReLU(0.1)),
            ConvolutionLayer(f_channels // 2, f_channels, action=nn.LeakyReLU(0.1)),
            ConvolutionLayer(f_channels, f_channels // 2, kernel_size=1, padding=0, action=nn.LeakyReLU(0.1)),
        )

    def forward(self, f1, f2):
        f1 = func.interpolate(self.c1(f1), scale_factor=2, mode='nearest')
        f2 = self.c2(f2)
        return self.c3(torch.cat([f1, f2], dim=1))


class Backbone(Module):
    """Mask RCNN"""

    def __init__(self):
        super(Backbone, self).__init__()
        self.block0 = nn.Sequential(
            nn.BatchNorm2d(3),
            ConvolutionLayer(3, 64)
        )
        self.block1 = Block(64, 128, 1)  # 64
        self.block2 = Block(128, 256, 2)  # 128
        self.block3 = Block(256, 512, 8)  # 256
        self.block4 = Block(512, 1024, 8)  # 512
        self.block5 = Block(1024, 2048, 4)  # 1024

    def forward(self, image):
        f_512 = self.block0(image)  # out 32 channels
        f_256 = self.block1(f_512)  # out 64 channels
        f_128 = self.block2(f_256)  # out 128 channels
        f_64 = self.block3(f_128)  # out 256 channels
        f_32 = self.block4(f_64)  # out 512 channels
        f_16 = self.block5(f_32)  # out 1024 channels
        return f_128, f_64, f_32, f_16


if __name__ == '__main__':
    m = Backbone()
    # torch.save(m.state_dict(), 'mask_rcnn.pt')
    x = torch.randn(1, 3, 640, 640)
    y = m(x)
    print(y[0].shape, y[1].shape, y[2].shape, y[3].shape, y[4].shape, y[5].shape)
    exit()
