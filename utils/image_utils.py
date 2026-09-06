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

# 功能：执行 mse 对应的计算或状态操作。
# 输入：
#   - img1：图像、图像路径或尺寸。
#   - img2：图像、图像路径或尺寸。
# 输出：返回计算、读取或组装得到的结果。
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

# 功能：执行 psnr 对应的计算或状态操作。
# 输入：
#   - img1：图像、图像路径或尺寸。
#   - img2：图像、图像路径或尺寸。
# 输出：返回计算、读取或组装得到的结果。
def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
