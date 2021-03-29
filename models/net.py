from torchvision.models.resnet import resnet50
from torch import nn
import torch.nn.functional as F
import torch
import warnings
# warnings.filterwarnings("ignore")


def coord_feat(f_map):
    x_range = torch.linspace(-1, 1, f_map.shape[-1],
                             device=f_map.device,
                             dtype=f_map.dtype)
    y_range = torch.linspace(-1, 1, f_map.shape[-2],
                             device=f_map.device,
                             dtype=f_map.dtype)
    y, x = torch.meshgrid(y_range, x_range)
    y = y.expand([f_map.shape[0], 1, -1, -1])
    x = x.expand([f_map.shape[0], 1, -1, -1])
    c_f = torch.cat([x, y], 1)
    return torch.cat([f_map, c_f], 1)


class Backbone(nn.Module):

    def __init__(self):
        super(Backbone, self).__init__()
        backbone = list(resnet50().children())
        self.stage1 = nn.Sequential(
            nn.BatchNorm2d(3),
            *backbone[0:3]
        )
        self.stage2 = nn.Sequential(
            *backbone[3:5]
        )
        self.stage3 = backbone[5]
        self.stage4 = backbone[6]
        self.stage5 = backbone[7]

    def forward(self, input_):
        f_2 = self.stage1(input_)
        f_4 = self.stage2(f_2)
        f_8 = self.stage3(f_4)
        f_16 = self.stage4(f_8)
        f_32 = self.stage5(f_16)
        return f_4, f_8, f_16, f_32


class FPN(nn.Module):

    def __init__(self):
        super(FPN, self).__init__()
        self.conv_4 = nn.Conv2d(256, 256, 1)
        self.conv_8 = nn.Conv2d(512, 256, 1)
        self.conv_16 = nn.Conv2d(1024, 256, 1)
        self.conv_32 = nn.Conv2d(2048, 256, 1)

        self.conv_4_out = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv_8_out = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv_16_out = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv_32_out = nn.Conv2d(256, 256, 3, 1, 1)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down_sample = nn.MaxPool2d(1, 2)

    def forward(self, f_4, f_8, f_16, f_32):
        f_4 = self.conv_4(f_4)
        f_8 = self.conv_8(f_8)
        f_16 = self.conv_16(f_16)
        f_32 = self.conv_32(f_32)

        f_32 = self.conv_32_out(f_32)
        f_16 = self.conv_16_out(f_16 + self.up_sample(f_32))
        f_8 = self.conv_8_out(f_8 + self.up_sample(f_16))
        f_4 = self.conv_4_out(f_4 + self.up_sample(f_8))
        f_64 = self.down_sample(f_32)
        return f_4, f_8, f_16, f_32, f_64


class InsHead(nn.Module):

    def __init__(self):
        super(InsHead, self).__init__()
        self.cate_tower = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(True)
        )
        self.kernel_tower = nn.Sequential(
            nn.Conv2d(258, 512, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(True)
        )
        self.cate_pred = nn.Sequential(
            nn.Conv2d(512, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.kernel_pred = nn.Conv2d(512, 256, 3, 1, 1)

    def forward(self, f_4, f_8, f_16, f_32, f_64):
        f_4 = F.interpolate(f_4,size=80, mode='bilinear', align_corners=True)
        f_64 = F.interpolate(f_64, scale_factor=2, mode='bilinear', align_corners=True)
        f_4_c, f_4_k = self.pre_cate_kernel(f_4, 40)
        f_8_c, f_8_k = self.pre_cate_kernel(f_8, 32)
        f_16_c, f_16_k = self.pre_cate_kernel(f_16, 24)
        f_32_c, f_32_k = self.pre_cate_kernel(f_32, 16)
        f_64_c, f_64_k = self.pre_cate_kernel(f_64, 12)
        return (f_4_c, f_4_k), (f_8_c, f_8_k), (f_16_c, f_16_k), (f_32_c, f_32_k), (f_64_c, f_64_k)

    def pre_cate_kernel(self, f_m, grid_nums):
        f_m = coord_feat(f_m)
        kernel_feat = F.interpolate(f_m, size=grid_nums, mode='bilinear', align_corners=True)
        cate_feat = kernel_feat[:, :-2, :, :]
        kernel_feat = self.kernel_tower(kernel_feat)
        cate_feat = self.cate_tower(cate_feat)
        kernel_pred = self.kernel_pred(kernel_feat)
        cate_pred = self.cate_pred(cate_feat)
        return cate_pred.permute(0, 2, 3, 1), kernel_pred.permute(0, 2, 3, 1)


class MaskHead(nn.Module):

    def __init__(self):
        super(MaskHead, self).__init__()
        self.f_4_to_4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU()
        )
        self.f_8_to_4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.f_16_to_4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.f_32_to_4 = nn.Sequential(
            nn.Conv2d(258, 128, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.to_mask_feature = nn.Sequential(
            nn.Conv2d(128, 256, 1, 1, 0, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU()
        )

    def forward(self, f_4, f_8, f_16, f_32):
        f_4 = self.f_4_to_4(f_4)
        f_8 = self.f_8_to_4(f_8)
        f_16 = self.f_16_to_4(f_16)
        f_32 = self.f_32_to_4(coord_feat(f_32))
        return self.to_mask_feature(f_4 + f_8 + f_16 + f_32)


class Solo(nn.Module):

    def __init__(self):
        super(Solo, self).__init__()
        self.backbone = Backbone()
        self.fpn = FPN()
        self.ins_head = InsHead()
        self.mask_head = MaskHead()

    def forward(self, input_):
        f_4, f_8, f_16, f_32 = self.backbone(input_)
        f_4, f_8, f_16, f_32, f_64 = self.fpn(f_4, f_8, f_16, f_32)
        (f_4_c, f_4_k), (f_8_c, f_8_k), (f_16_c, f_16_k), (f_32_c, f_32_k), (f_64_c, f_64_k) = self.ins_head(f_4,
                                                                                                             f_8,
                                                                                                             f_16,
                                                                                                             f_32,
                                                                                                             f_64)
        mask_feature = self.mask_head(f_4, f_8, f_16, f_32)
        return (f_4_c, f_4_k), (f_8_c, f_8_k), (f_16_c, f_16_k), (f_32_c, f_32_k), (f_64_c, f_64_k), mask_feature

if __name__ == '__main__':
    m = Solo()
    # print(net)
    image = torch.randn(1, 3, 640, 640)
    (f_4_c, f_4_k), (f_8_c, f_8_k), (f_16_c, f_16_k), (f_32_c, f_32_k), (f_64_c, f_64_k), mask_feature = m(image)

    for i in [f_4_c, f_4_k, f_8_c, f_8_k, f_16_c, f_16_k, f_32_c, f_32_k, f_64_c, f_64_k, mask_feature]:
        print(i.shape)



