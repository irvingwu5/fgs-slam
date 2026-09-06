# 文件作用：modules；隶属于 FGS-SLAM 的modules模块。
from collections import OrderedDict

import torch


# 功能：执行 normalize_activation 对应的计算或状态操作。
# 输入：
#   - x：x 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - eps：eps 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


# 功能：获取与 get_state_dict 对应的数据或状态。
# 输入：
#   - net_type：net_type 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - version：version 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    # build url
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
        + f'master/lpips/weights/v{version}/{net_type}.pth'

    # download
    old_state_dict = torch.hub.load_state_dict_from_url(
        url, progress=True,
        map_location=None if torch.cuda.is_available() else torch.device('cpu')
    )

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict
