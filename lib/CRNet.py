import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.res2net import res2net50_v1b

class StarkIndustriesProduction(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, groups=1, dilation=1, bias=True):
        super(StarkIndustriesProduction, self).__init__()
        assert dilation in [1, 2], "dilation设置错误！"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels // self.groups, 3, 3))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mask1 = torch.cuda.FloatTensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]]).to("cuda:0")
        self.mask2 = torch.cuda.FloatTensor([[0, 0, 0], [1, 1, 1], [0, 0, 0]]).to("cuda:0")
        self.mask3 = torch.cuda.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).to("cuda:0")
        self.mask4 = torch.cuda.FloatTensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).to("cuda:0")

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, _type_='stark1'):
        if _type_ == 'stark1':
            weight = torch.mul(self.weight, self.mask1)
        elif _type_ == 'stark2':
            weight = torch.mul(self.weight, self.mask2)
        elif _type_ == 'stark3':
            weight = torch.mul(self.weight, self.mask3)
        elif _type_ == 'stark4':
            weight = torch.mul(self.weight, self.mask4)

        out = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return out
class MR(nn.Module):
    def     __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.inconv1 = StarkIndustriesProduction(in_channels, out_channels)
        self.inconv2 = StarkIndustriesProduction(in_channels, out_channels)
        self.inconv3 = StarkIndustriesProduction(in_channels, out_channels)
        self.inconv4 = StarkIndustriesProduction(in_channels, out_channels)
    def forward(self, x):
        h1 = self.in_layers(x)
        dConv11 = self.inconv1(h1, 'stark1')
        dConv12 = self.inconv1(h1, 'stark2')
        dConv13 = self.inconv1(h1, 'stark3')
        dConv14 = self.inconv1(h1, 'stark4')
        d_layer1 = dConv11 + dConv12 + dConv13 + dConv14
        return d_layer1

class conv_3_7(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_3_7, self).__init__()
        self.conv_3 = conv_block_3(ch_in, ch_out)
        self.conv_7 = conv_block_7(ch_in, ch_out)
        self.conv = nn.Conv2d(ch_out * 2, ch_out, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x3 = self.conv_3(x)
        x7 = self.conv_7(x)
        x = torch.cat((x3, x7), dim=1)
        x = self.conv(x)

        return x


class conv_block_3(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_7(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#采用双线性插值上采用

            self.mr = MR(out_channels, out_channels)
            self.heconv = conv_3_7(out_channels,out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
        self.eg_conv1 = nn.Sequential(*[nn.Conv2d(128, out_channels, kernel_size=1),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace=True)])

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,make_eg: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        # 输入图片如果不是16的整数倍，防止上采样的x1和要拼接的x2的图片的长和宽不一致，因为再unet网络中下采样4此，所以图像缩小了16倍
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        if self.in_channels != self.out_channels:
            x1 = self.conv1(x1)
        x = x2+x1
        mr = self.mr(x)

        target_h, target_w = mr.size(2), mr.size(3)
        make_eg = F.interpolate(make_eg, size=(target_h, target_w), mode='bilinear', align_corners=False)
        if self.in_channels != self.out_channels:
            make_eg = self.eg_conv1(make_eg)
        em = make_eg * mr
        em =self.heconv(em)
        x = mr + em
        return x

#z最后的1×1卷积层
class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
class conv_3_7(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_3_7, self).__init__()
        self.conv_3 = conv_block_3(ch_in, ch_out)
        self.conv_7 = conv_block_7(ch_in, ch_out)
        self.conv = nn.Conv2d(ch_out * 2, ch_out, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x3 = self.conv_3(x)
        x7 = self.conv_7(x)
        x = torch.cat((x3, x7), dim=1)
        x = self.conv(x)

        return x


class conv_block_3(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_7(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class EgOut(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(EgOut, self).__init__(
            nn.Conv2d(in_channels, 320, kernel_size=3,padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            nn.Conv2d(320, num_classes, kernel_size=1),
        )
class DepthWisePatchEmbedding(nn.Sequential):
    def __init__(self, in_channels, patch_size, embed_dim):
        super(DepthWisePatchEmbedding, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=patch_size, stride=patch_size, groups=in_channels, bias=False),
            nn.InstanceNorm2d(num_features=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=1)
        )
class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)
class DualPathChannelBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(DualPathChannelBlock, self).__init__()
        self.conv1 = nn.Sequential(
            channel_attention(in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,out_channels,1))
        self.sa = spatial_attention()
        self.action = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        )
        self.conv33 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        )


        self.depthwise = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 7, groups=out_channels, padding= 7 // 2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channels))
        self.pointwise = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channels))
        self.conv1x1 = nn.Conv2d(3*out_channels,out_channels,1)
    def forward(self, x):
        x1 = self.conv1(x)
        x1  = self.action(x1)
        sa = self.sa(x1)
        att1 = x1 + sa
        conv3_1 = self.conv3(att1)
        conv3_1 = conv3_1 +att1
        conv3_2 = self.conv33(att1)
        conv3_2 = conv3_2 +sa

        dw = self.depthwise(x1)
        dw = dw +x1
        dw = self.pointwise(dw)

        att2 = torch.concat([conv3_1,conv3_2,dw],dim=1)
        att2 = self.conv1x1(att2)
        att2 = self.action(att2)
        return att2
class up_adap_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_adap_conv, self).__init__()
        self.in_channels =in_channels
        self.out_channels =out_channels
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                    nn.InstanceNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
        self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, x1,x2):
        target_h, target_w = x2.size(2), x2.size(3)
        x1 = F.interpolate(x1, size=(target_h, target_w), mode='bilinear', align_corners=False)
        if self.in_channels != self.out_channels:
            x1 = self.conv(x1) * self.weight.sigmoid()
        x = x1 + x2
        return x
class adap_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(adap_conv, self).__init__()
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                    nn.InstanceNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
        self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, x):
        x = self.conv(x) * self.weight.sigmoid()
        return x
class Refine_block2_1(nn.Module):
    def __init__(self, in_channel, out_channel, factor, require_grad=False):
        super(Refine_block2_1, self).__init__()
        self.pre_conv1 = adap_conv(in_channel[0], out_channel)
        self.pre_conv2 = adap_conv(in_channel[1], out_channel)
        self.factor = factor

    def forward(self, *input):

        x1 = self.pre_conv1(input[0])
        x2 = self.pre_conv2(input[1])
        if self.factor >= 2:
            x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=self.factor, padding=int(self.factor/2),
                                output_padding=(0, 0))
        if x1.shape != x2.shape:
            target_h, target_w = x1.size(2), x1.size(3)
            x2 = F.interpolate(x2, size=(target_h, target_w), mode='bilinear',align_corners=False)

        return x1 + x2
class CRNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 bilinear: bool = True,
                 base_c: int = 16):
        super(CRNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = 1
        self.bilinear = bilinear

        # self.in_conv = DoubleConv(in_channels, base_c)
        res2net = res2net50_v1b(pretrained=True)
        # Encoder
        self.encoder1_conv = res2net.conv1
        self.encoder1_bn = res2net.bn1
        self.encoder1_relu = res2net.relu
        self.maxpool = res2net.maxpool
        self.encoder2 = res2net.layer1
        self.encoder3 = res2net.layer2
        self.encoder4 = res2net.layer3
        self.encoder5 = res2net.layer4

        self.reduce1 = nn.Sequential(nn.Conv2d(64, 16, 1),
                                     )
        self.reduce2 = nn.Sequential(nn.Conv2d(256, 32, 1),

                                     )
        self.reduce3 = nn.Sequential(nn.Conv2d(512, 64, 1),

                                     )
        self.reduce4 = nn.Sequential(nn.Conv2d(1024, 128, 1),

                                     )
        self.reduce5 = nn.Sequential(nn.Conv2d(2048, 128, 1),

                                     )
        self.uac4 = up_adap_conv(128, 128)
        self.uac3 = up_adap_conv(128, 64)
        self.uac2 = up_adap_conv(64, 32)
        self.uac1 = up_adap_conv(32, 16)

        self.dpc1 = DualPathChannelBlock(in_channels=16, out_channels=16)
        self.dpc2 = DualPathChannelBlock(in_channels=32, out_channels=32)
        self.dpc3 = DualPathChannelBlock(in_channels=64, out_channels=64)
        self.dpc4 = DualPathChannelBlock(in_channels=128, out_channels=128)

        self.t4 = Refine_block2_1(in_channel=(128, 128), out_channel=128, factor=0)
        self.t3 = Refine_block2_1(in_channel=(64, 64), out_channel=64, factor=0)
        self.t2 = Refine_block2_1(in_channel=(32, 32), out_channel=32, factor=0)
        self.t1 = Refine_block2_1(in_channel=(16, 16), out_channel=16, factor=0)

        self.up1 = Up(base_c * 8, base_c * 8 , bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 , bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 , bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        embed_dim = 128
        self.patch_embeddings = nn.ModuleList(
            [
                DepthWisePatchEmbedding(in_channels=ic, patch_size=16 // 2 ** i, embed_dim=embed_dim)
                for i, ic in enumerate([16, 32, 64, 128, 128])
            ]
        )
        self.conv_3_7 = conv_3_7(640, 128)
        self.egout = EgOut(640, num_classes)
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]: #[8, 3, 352, 352]
        x0 = x
        e1_ = self.encoder1_conv(x)  # H/2*W/2*64 #[8, 64, 176, 176]
        e1_ = self.encoder1_bn(e1_)
        e1_ = self.encoder1_relu(e1_) #[8, 64, 176, 176]
        e1_pool_ = self.maxpool(e1_)  # H/4*W/4*64 [8, 64, 88, 88]
        e2_ = self.encoder2(e1_pool_)  # H/4*W/4*64 [8, 256, 88, 88]
        e3_ = self.encoder3(e2_)  # H/8*W/8*128 [8, 512, 44, 44]
        e4_ = self.encoder4(e3_)  # H/16*W/16*256 [8, 1024, 22, 22]
        e5_ = self.encoder5(e4_)  # H/32*W/32*512 [8, 2048, 11, 11]

        x1 = self.reduce1(e1_)
        x2 = self.reduce2(e2_)
        x3 = self.reduce3(e3_)
        x4 = self.reduce4(e4_)
        x5 = self.reduce5(e5_)

        uac1 = self.uac1(x2, x1)
        uac2 = self.uac2(x3, x2)
        uac3 = self.uac3(x4, x3)
        uac4 = self.uac4(x5, x4)

        dpc1 = self.dpc1(uac1)
        dpc2 = self.dpc2(uac2)
        dpc3 = self.dpc3(uac3)
        dpc4 = self.dpc4(uac4)

        t4 = self.t4(dpc4, x4)
        t3 = self.t3(dpc3, x3)
        t2 = self.t2(dpc2, x2)
        t1 = self.t1(dpc1, x1)

        sides = [x1, x2, x3, x4, x5]

        side_embeddings = []
        for s, embd in zip(sides, self.patch_embeddings):
            side_embeddings.append(embd(s))
        side_embeddings = torch.cat(side_embeddings, dim=1)
        egout = self.egout(side_embeddings)
        make_eg = self.conv_3_7(side_embeddings)

        up1 = self.up1(x5, t4, make_eg)
        up2 = self.up2(up1, t3, make_eg)
        up3 = self.up3(up2, t2, make_eg)
        up4 = self.up4(up3, t1, make_eg)

        target_h, target_w = x0.size(2), x0.size(3)
        x = F.interpolate(up4, size=(target_h, target_w), mode='bilinear', align_corners=False)
        out = self.out_conv(x)
        egout = F.interpolate(egout, size=(target_h, target_w), mode='bilinear', align_corners=False)

        return out, egout


from thop import profile

def cal_param(net):
    # model = torch.nn.DataParallel(net)
    inputs = torch.randn([1, 3, 352, 352]).cuda()
    flop, para = profile(net, inputs=(inputs,), verbose=False)
    return 'Flops：' + str(2 * flop / 1000 ** 3) + 'G', 'Params：' + str(para / 1000 ** 2) + 'M'


if __name__ == "__main__":
    net = CRNet().cuda()
    print(cal_param(net))
