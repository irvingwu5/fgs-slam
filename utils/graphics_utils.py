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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

# 功能：执行 geom_transform_points 对应的计算或状态操作。
# 输入：
#   - points：三维点云或点属性。
#   - transf_matrix：transf_matrix 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

# 功能：获取与 getWorld2View 对应的数据或状态。
# 输入：
#   - R：旋转矩阵。
#   - t：t 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

# 功能：获取与 getWorld2View2 对应的数据或状态。
# 输入：
#   - R：旋转矩阵。
#   - t：t 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - translate：translate 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - scale：尺度参数或倍率。
# 输出：返回计算、读取或组装得到的结果。
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

# 功能：获取与 getProjectionMatrix 对应的数据或状态。
# 输入：
#   - znear：znear 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - zfar：zfar 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - fovX：fovX 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - fovY：fovY 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

# 功能：执行 fov2focal 对应的计算或状态操作。
# 输入：
#   - fov：fov 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - pixels：pixels 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

# 功能：执行 focal2fov 对应的计算或状态操作。
# 输入：
#   - focal：相机焦距。
#   - pixels：pixels 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))