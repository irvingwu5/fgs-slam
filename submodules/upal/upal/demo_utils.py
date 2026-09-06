# 文件作用：提供演示图像与绘制工具相关实现；隶属于演示图像与绘制工具模块。
"""Shared image loading and drawing helpers for the runnable demos."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


# 函数作用：读取、缩放图像并转换为 RGB 批张量。
# 所属模块：演示图像与绘制工具。
# 输入：
#   - path：输入或输出文件路径。
#   - max_size：图像最长边允许的最大尺寸。
# 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
def load_image(path: Path, max_size: int) -> tuple[np.ndarray, torch.Tensor]:
    """Load an RGB image and its normalized ``1 x 3 x H x W`` tensor."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    scale = min(1.0, max_size / max(rgb.shape[:2]))
    if scale < 1.0:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0)
    return rgb, tensor


# 函数作用：在图像上绘制关键点和线段。
# 所属模块：演示图像与绘制工具。
# 输入：
#   - image：输入图像张量或 NumPy 图像。
#   - keypoints：像素坐标顺序为 (x,y) 的关键点。
#   - lines：端点形状为 [L,2,2] 的二维线段。
# 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
def draw_features(
    image: np.ndarray,
    *,
    keypoints: np.ndarray | None = None,
    lines: np.ndarray | None = None,
) -> np.ndarray:
    """Draw model features on an RGB image and return a BGR visualization."""
    output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if lines is not None:
        for line in lines:
            p0, p1 = np.rint(line).astype(int)
            cv2.line(output, tuple(p0), tuple(p1), (54, 224, 88), 1, cv2.LINE_AA)
    if keypoints is not None:
        for point in np.rint(keypoints).astype(int):
            cv2.circle(output, tuple(point), 2, (39, 90, 245), -1, cv2.LINE_AA)
    return output


# 函数作用：将两幅图像补齐高度后水平拼接。
# 所属模块：演示图像与绘制工具。
# 输入：
#   - left：left 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
#   - right：right 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
# 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
def side_by_side(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, int]:
    """Join BGR images horizontally and return the join plus the right x offset."""
    height = max(left.shape[0], right.shape[0])
    if left.shape[0] != height:
        left = cv2.copyMakeBorder(left, 0, height - left.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    if right.shape[0] != height:
        right = cv2.copyMakeBorder(right, 0, height - right.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    return np.concatenate((left, right), axis=1), left.shape[1]
