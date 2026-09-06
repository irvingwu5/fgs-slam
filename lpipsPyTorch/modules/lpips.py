# 文件作用：modules；隶属于 FGS-SLAM 的modules模块。
import torch
import torch.nn as nn

from .networks import get_network, LinLayers
from .utils import get_state_dict


class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    # 功能：初始化对象状态、配置和所需资源。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - net_type：net_type 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - version：version 所表示的计算输入，具体类型、形状和约束见调用上下文。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type)

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    # 功能：执行网络前向计算与 forward 对应的数据或状态。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - x：x 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - y：y 所表示的计算输入，具体类型、形状和约束见调用上下文。
    # 输出：返回计算、读取或组装得到的结果。
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0), 0, True)
