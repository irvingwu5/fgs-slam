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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import cv2
import numpy as np

# 功能：执行 l1_loss 对应的计算或状态操作。
# 输入：
#   - network_output：network_output 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - gt：gt 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def l1_loss(network_output, gt):
    loss = torch.abs((network_output - gt))
    loss = torch.where(gt!=0, loss, 0.)
    return loss, loss.mean()

# 功能：执行 l2_loss 对应的计算或状态操作。
# 输入：
#   - network_output：network_output 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - gt：gt 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def l2_loss(network_output, gt):
    loss = ((network_output - gt) ** 2)
    loss = torch.where(gt!=0, loss, 0.)
    return loss.mean()

# 功能：执行 gaussian 对应的计算或状态操作。
# 输入：
#   - window_size：window_size 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - sigma：sigma 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

# 功能：创建与 create_window 对应的数据或状态。
# 输入：
#   - window_size：window_size 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - channel：channel 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

# 功能：执行 ssim 对应的计算或状态操作。
# 输入：
#   - img：图像、图像路径或尺寸。
#   - gt：gt 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - window_size：window_size 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - size_average：size_average 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def ssim(img, gt, window_size=11, size_average=True):
    img = torch.where(gt!=0, img, 0.)
    channel = img.size(-3)
    window = create_window(window_size, channel)

    if img.is_cuda:
        window = window.cuda(img.get_device())
    window = window.type_as(img)

    return _ssim(img, gt, window, window_size, channel, size_average)

# 功能：执行 _ssim 对应的计算或状态操作。
# 输入：
#   - img1：图像、图像路径或尺寸。
#   - img2：图像、图像路径或尺寸。
#   - window：window 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - window_size：window_size 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - channel：channel 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - size_average：size_average 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map, ssim_map.mean()
    else:
        return ssim_map, ssim_map.mean(1).mean(1).mean(1)

# 功能：执行 frequency 对应的计算或状态操作。
# 输入：
#   - frame：frame 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def frequency(frame):
    gray = 0.144 * frame[0, :, :] + 0.587 * frame[1, :, :] + 0.299 * frame[2, :, :]
    # gray = cv2.cvtColor(np.round(((frame*255).cpu().detach().numpy().transpose(1, 2, 0))) , cv2.COLOR_RGB2GRAY)  # RGB or BGR ?
    # gray = torch.tensor(gray).cuda()

    freq = torch.fft.fft2(gray, dim=(-2,-1), norm='ortho')
    freq = torch.fft.fftshift(freq, dim=(-2,-1))  # 中心化

    phase = torch.angle(freq)
    magnitude = torch.abs(freq)

    return phase, magnitude