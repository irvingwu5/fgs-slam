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

from errno import EEXIST
from os import makedirs, path
import os

# 功能：执行 mkdir_p 对应的计算或状态操作。
# 输入：
#   - folder_path：folder_path 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：无显式返回值；输出体现为状态或外部资源更新。
def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

# 功能：执行 searchForMaxIteration 对应的计算或状态操作。
# 输入：
#   - folder：folder 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)
