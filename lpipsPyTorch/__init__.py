# 文件作用：lpipsPyTorch；隶属于 FGS-SLAM 的lpipsPyTorch模块。
import torch

from .modules.lpips import LPIPS


# 功能：执行 lpips 对应的计算或状态操作。
# 输入：
#   - x：x 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - y：y 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - net_type：net_type 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - version：version 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    device = x.device
    criterion = LPIPS(net_type, version).to(device)
    return criterion(x, y)
