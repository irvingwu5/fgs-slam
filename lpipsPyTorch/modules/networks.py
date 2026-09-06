# 文件作用：modules；隶属于 FGS-SLAM 的modules模块。
from typing import Sequence

from itertools import chain

import torch
import torch.nn as nn
from torchvision import models

from .utils import normalize_activation


# 功能：获取与 get_network 对应的数据或状态。
# 输入：
#   - net_type：net_type 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def get_network(net_type: str):
    if net_type == 'alex':
        return AlexNet()
    elif net_type == 'squeeze':
        return SqueezeNet()
    elif net_type == 'vgg':
        return VGG16()
    else:
        raise NotImplementedError('choose net_type from [alex, squeeze, vgg].')


class LinLayers(nn.ModuleList):
    # 功能：初始化对象状态、配置和所需资源。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - n_channels_list：n_channels_list 所表示的计算输入，具体类型、形状和约束见调用上下文。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def __init__(self, n_channels_list: Sequence[int]):
        super(LinLayers, self).__init__([
            nn.Sequential(
                nn.Identity(),
                nn.Conv2d(nc, 1, 1, 1, 0, bias=False)
            ) for nc in n_channels_list
        ])

        for param in self.parameters():
            param.requires_grad = False


class BaseNet(nn.Module):
    # 功能：初始化对象状态、配置和所需资源。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def __init__(self):
        super(BaseNet, self).__init__()

        # register buffer
        self.register_buffer(
            'mean', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer(
            'std', torch.Tensor([.458, .448, .450])[None, :, None, None])

    # 功能：设置与 set_requires_grad 对应的数据或状态。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - state：state 所表示的计算输入，具体类型、形状和约束见调用上下文。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def set_requires_grad(self, state: bool):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    # 功能：执行 z_score 对应的计算或状态操作。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - x：x 所表示的计算输入，具体类型、形状和约束见调用上下文。
    # 输出：返回计算、读取或组装得到的结果。
    def z_score(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    # 功能：执行网络前向计算与 forward 对应的数据或状态。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - x：x 所表示的计算输入，具体类型、形状和约束见调用上下文。
    # 输出：返回计算、读取或组装得到的结果。
    def forward(self, x: torch.Tensor):
        x = self.z_score(x)

        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output


class SqueezeNet(BaseNet):
    # 功能：初始化对象状态、配置和所需资源。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.layers = models.squeezenet1_1(True).features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]

        self.set_requires_grad(False)


class AlexNet(BaseNet):
    # 功能：初始化对象状态、配置和所需资源。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def __init__(self):
        super(AlexNet, self).__init__()

        self.layers = models.alexnet(True).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]

        self.set_requires_grad(False)


class VGG16(BaseNet):
    # 功能：初始化对象状态、配置和所需资源。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]

        self.set_requires_grad(False)
