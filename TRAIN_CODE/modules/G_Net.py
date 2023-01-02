# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# ------------------------------------------------------------------------------
import logging
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

InstanceNorm3d = nn.InstanceNorm3d
InstanceNorm3d_class = nn.InstanceNorm3d
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def spectral_norm(module, mode=False):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class Fuse_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fuse_Block, self).__init__()

        self.expan_channels = out_channels * 2

        self.expan = nn.Sequential(
            nn.Conv3d(in_channels, self.expan_channels, 1),
            nn.InstanceNorm3d(self.expan_channels),
            nn.LeakyReLU(),
            nn.Conv3d(self.expan_channels, self.expan_channels, 3, 1, 1)
        )

        self.w = nn.Sequential(
            nn.Conv3d(self.expan_channels, self.expan_channels // 2, 3, padding=2, dilation=2),
            nn.Conv3d(self.expan_channels // 2, self.expan_channels, 1),
            nn.Sigmoid()
        )

        self.skip_conv = nn.Conv3d(in_channels, out_channels, 1)
        self.zoom_conv = nn.Conv3d(out_channels * 2, out_channels, 1)

    def forward(self, input):
        x = self.expan(input)
        x = x * self.w(x)
        x = self.zoom_conv(x)
        skip_x = self.skip_conv(input)
        out = x + skip_x
        return out


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride, padding=0):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(input_dim, output_dim, kernel_size=kernel, stride=stride, padding=padding),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.upsample(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, padding_mode='reflect')


class Bottleneck(nn.Module):
    expansion = 8

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = InstanceNorm3d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Sequential(
            nn.ReflectionPad3d(1),
            nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                      padding=0, bias=False))
        self.bn2 = InstanceNorm3d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = InstanceNorm3d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, BN=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = spectral_norm(conv3x3(inplanes, planes, stride), False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = spectral_norm(conv3x3(planes, planes), False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.bn = BN
        self.skip_conv = None
        if not BN:
            self.skip_conv = nn.Conv3d(inplanes, planes, 1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        # if self.bn:
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        if not self.bn:
            identity = self.skip_conv(identity)
        out += identity
        out = self.relu(out)
        return out


class HighResolutionModule3D(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule3D, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.LeakyReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(

                nn.Conv3d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                InstanceNorm3d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index], stride=1))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv3d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        InstanceNorm3d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.ReflectionPad3d(1),
                                nn.Conv3d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 0, bias=False),
                                InstanceNorm3d(num_outchannels_conv3x3,
                                               momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.ReflectionPad3d(1),
                                nn.Conv3d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 0, bias=False),
                                InstanceNorm3d(num_outchannels_conv3x3,
                                               momentum=BN_MOMENTUM),
                                nn.LeakyReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    deep_output = x[i].shape[-3]
                    temp = self.fuse_layers[i][j](x[j])

                    y = y + F.interpolate(
                        temp,
                        size=[deep_output, height_output, width_output],
                        mode='trilinear', align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class G_Net(nn.Module):
    def __init__(self, base=48):
        super(G_Net, self).__init__()
        self.base = base
        num_channels = base
        self.skip_conv = nn.Sequential(nn.ReflectionPad3d(1),
                                       nn.Conv3d(1, num_channels, kernel_size=3, stride=1, padding=0),
                                       InstanceNorm3d(num_channels, momentum=BN_MOMENTUM),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv3d(num_channels, num_channels, kernel_size=1, stride=1, padding=0),
                                       InstanceNorm3d(num_channels, momentum=BN_MOMENTUM),
                                       )
        self.conv1 = nn.Sequential(
            nn.ReflectionPad3d(1),
            nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=2, padding=0),
            InstanceNorm3d(num_channels, momentum=BN_MOMENTUM),
            nn.LeakyReLU(inplace=True),
        )
        self.conv1_0 = nn.Sequential(BasicBlock(num_channels, num_channels, 1))

        self.conv1_1 = nn.Sequential(
            nn.ReflectionPad3d(1),
            nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=2, padding=0, bias=False),
            InstanceNorm3d(num_channels, momentum=BN_MOMENTUM),
            nn.LeakyReLU(inplace=True)
        )

        self.layer1 = self._make_layer(Bottleneck, num_channels, num_channels, 1)
        stage1_out_channel = Bottleneck.expansion * num_channels

        num_channels = [base, base * 2]
        block = BasicBlock
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(num_inchannels=num_channels, num_branches=2, num_modules=6)

        num_channels = [base, base * 2]
        block = BasicBlock
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(num_inchannels=num_channels, num_branches=2, num_modules=8)

        self.fuse1 = nn.Conv3d(base, base, 1)
        self.fuse2 = nn.Conv3d(base * 2, base, 1)

        self.fuse_layer1 = nn.Conv3d(base * 2, base, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

        self.fuse_layer2 = Fuse_Block(base * 2, base)
        self.fuse_layer3 = Fuse_Block(base * 2, base)

        self.upsample_1 = Upsample(base, base, kernel=4, stride=2, padding=1)
        self.upsample_2 = Upsample(base, base, kernel=4, stride=2, padding=1)

        self.layer2 = nn.Sequential(
            block(base, base, 1),
            #block(base, base, 1)
        )
        self.last_layer = nn.Sequential(
            block(base, base, 1, BN=True),
            block(base, base, 1, BN=False),
            nn.Conv3d(
                in_channels=base,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            padding_mode='replicate'),
            nn.Tanh(),
        )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv3d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        InstanceNorm3d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.LeakyReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv3d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        InstanceNorm3d(outchannels, momentum=BN_MOMENTUM),
                        nn.LeakyReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                InstanceNorm3d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, num_inchannels, num_branches, multi_scale_output=True, num_modules=2):

        num_blocks = [2] * num_branches
        num_channels = [self.base * i for i in range(1, num_branches + 1)]
        block = BasicBlock
        fuse_method = 'SUM'
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule3D(num_branches,
                                       block,
                                       num_blocks,
                                       num_inchannels,
                                       num_channels,
                                       fuse_method,
                                       reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x_in):
        skip_x_0 = self.skip_conv(x_in)
        x = self.conv1(skip_x_0)
        # passby = self.passby(x_in)
        # x = x + passby
        skip_x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.layer1(x)
        x_list = []
        for i in range(2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(2):
            if self.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage3(x_list)
        x1 = self.fuse1(x[0])
        b, c, w, h, d = x1.shape
        x2 = self.fuse2(x[1])
        x2 = F.interpolate(x2, size=(w, h, d), mode='trilinear', align_corners=True)
        x = torch.cat([x1, x2], dim=1)
        x = self.fuse_layer1(x)

        x = self.upsample_1(x)
        x = torch.cat((x, skip_x), dim=1)

        x = self.fuse_layer2(x)
        x = self.layer2(x)
        x = self.upsample_2(x)
        x = torch.cat((x, skip_x_0), dim=1)
        x = self.fuse_layer3(x)
        x = self.last_layer(x)
        return x
