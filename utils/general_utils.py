# 文件作用：utils；隶属于 FGS-SLAM 的utils模块。
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
import PIL

# 功能：执行 inverse_sigmoid 对应的计算或状态操作。
# 输入：
#   - x：x 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def inverse_sigmoid(x):
    return torch.log(x/(1-x))

# 功能：执行 PILtoTorch 对应的计算或状态操作。
# 输入：
#   - pil_image：pil_image 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - resolution：resolution 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def PILtoTorch(pil_image, resolution):
    if len(pil_image.split()) == 3:
        resized_image_PIL = pil_image.resize(resolution, PIL.Image.LANCZOS)
        resized_image = torch.from_numpy(np.array(resized_image_PIL))
        resized_image = resized_image / 255.0
        return resized_image.permute(2, 0, 1)
    else:
        resized_image_PIL = pil_image.resize(resolution, PIL.Image.NEAREST)
        resized_image = torch.from_numpy(np.array(resized_image_PIL))
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

# 功能：获取与 get_expon_lr_func 对应的数据或状态。
# 输入：
#   - lr_init：学习率参数。
#   - lr_final：学习率参数。
#   - lr_delay_steps：学习率参数。
#   - lr_delay_mult：学习率参数。
#   - max_steps：max_steps 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    # 功能：执行 helper 对应的计算或状态操作。
    # 输入：
    #   - step：step 所表示的计算输入，具体类型、形状和约束见调用上下文。
    # 输出：返回计算、读取或组装得到的结果。
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

# 功能：执行 strip_lowerdiag 对应的计算或状态操作。
# 输入：
#   - L：L 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

# 功能：执行 strip_symmetric 对应的计算或状态操作。
# 输入：
#   - sym：sym 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def strip_symmetric(sym):
    return strip_lowerdiag(sym)

# 功能：执行 build_rotation 对应的计算或状态操作。
# 输入：
#   - r：r 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    # 수정
    # xyzw
    # r = q[:, 0]
    # x = q[:, 1]
    # y = q[:, 2]
    # z = q[:, 3]
    
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    r = q[:, 3]
    
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

# 功能：执行 build_scaling_rotation 对应的计算或状态操作。
# 输入：
#   - s：s 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - r：r 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

# 功能：执行 safe_state 对应的计算或状态操作。
# 输入：
#   - silent：silent 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：无显式返回值；输出体现为状态或外部资源更新。
def safe_state(silent):
    old_f = sys.stdout
    class F:
        # 功能：初始化对象状态、配置和所需资源。
        # 输入：
        #   - self：当前类实例，提供并更新对象状态。
        #   - silent：silent 所表示的计算输入，具体类型、形状和约束见调用上下文。
        # 输出：无显式返回值；输出体现为状态或外部资源更新。
        def __init__(self, silent):
            self.silent = silent

        # 功能：执行 write 对应的计算或状态操作。
        # 输入：
        #   - self：当前类实例，提供并更新对象状态。
        #   - x：x 所表示的计算输入，具体类型、形状和约束见调用上下文。
        # 输出：无显式返回值；输出体现为状态或外部资源更新。
        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        # 功能：执行 flush 对应的计算或状态操作。
        # 输入：
        #   - self：当前类实例，提供并更新对象状态。
        # 输出：无显式返回值；输出体现为状态或外部资源更新。
        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
