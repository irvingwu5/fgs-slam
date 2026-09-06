# 文件作用：提供共享编码器与点线预测网络相关实现；隶属于共享编码器与点线预测网络模块。
"""Standalone inference architecture for the UPAL joint point-line network."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import torchvision
from torch import nn


MODEL_CONFIG = {
    "c1": 16,
    "c2": 32,
    "c3": 64,
    "c4": 128,
    "dim": 128,
    "descriptor_kernel_size": 3,
    "descriptor_samples": 16,
}


class InputPadder:
    """Pad and unpad images to a multiple of a requested divisor."""

    # 函数作用：初始化对象配置、网络层和所需状态。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - height：height 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    #   - width：width 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    #   - divisor：divisor 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    # 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
    def __init__(self, height: int, width: int, divisor: int = 32):
        pad_h = (((height // divisor) + 1) * divisor - height) % divisor
        pad_w = (((width // divisor) + 1) * divisor - width) % divisor
        self.padding = [
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
        ]

    # 函数作用：填充 pad 对应的数据或状态。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - tensor：tensor 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    # 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
    def pad(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.pad(tensor, self.padding, mode="replicate")

    # 函数作用：移除填充 unpad 对应的数据或状态。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - tensor：tensor 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    # 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
    def unpad(self, tensor: torch.Tensor) -> torch.Tensor:
        left, right, top, bottom = self.padding
        return tensor[..., top : tensor.shape[-2] - bottom, left : tensor.shape[-1] - right]


class DeformableConv2d(nn.Module):
    # 函数作用：初始化对象配置、网络层和所需状态。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - in_channels：in_channels 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    #   - out_channels：out_channels 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    #   - kernel_size：尺寸或数量上限。
    #   - padding：图像边界填充量。
    #   - bias：bias 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    # 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.padding = padding
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.regular_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    # 函数作用：执行当前网络层或完整 UPAL 模型的前向计算。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - tensor：tensor 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    # 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        max_offset = max(tensor.shape[-2:]) / 4.0
        offset = self.offset_conv(tensor).clamp(-max_offset, max_offset)
        return torchvision.ops.deform_conv2d(
            input=tensor,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
        )


# 函数作用：执行 _make_conv 对应的计算或状态操作。
# 所属模块：共享编码器与点线预测网络。
# 输入：
#   - in_channels：in_channels 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
#   - out_channels：out_channels 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
#   - conv_type：conv_type 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
#   - bias：bias 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
# 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
def _make_conv(
    in_channels: int,
    out_channels: int,
    conv_type: str,
    bias: bool = False,
) -> nn.Module:
    if conv_type == "conv":
        return nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias)
    if conv_type == "dcn":
        return DeformableConv2d(in_channels, out_channels, bias=bias)
    raise ValueError(f"Unknown convolution type: {conv_type}")


class ConvBlock(nn.Module):
    # 函数作用：初始化对象配置、网络层和所需状态。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - in_channels：in_channels 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    #   - out_channels：out_channels 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    #   - gate：gate 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    #   - norm：norm 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    #   - conv_type：conv_type 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    # 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gate: nn.Module,
        norm: Callable[[int], nn.Module],
        conv_type: str,
    ):
        super().__init__()
        self.gate = gate
        self.conv1 = _make_conv(in_channels, out_channels, conv_type)
        self.bn1 = norm(out_channels)
        self.conv2 = _make_conv(out_channels, out_channels, conv_type)
        self.bn2 = norm(out_channels)

    # 函数作用：执行当前网络层或完整 UPAL 模型的前向计算。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - tensor：tensor 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    # 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.gate(self.bn1(self.conv1(tensor)))
        return self.gate(self.bn2(self.conv2(tensor)))


class ResBlock(nn.Module):
    # 函数作用：初始化对象配置、网络层和所需状态。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - in_channels：in_channels 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    #   - out_channels：out_channels 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    #   - downsample：downsample 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    #   - gate：gate 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    #   - norm：norm 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    #   - conv_type：conv_type 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    # 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: nn.Module,
        gate: nn.Module,
        norm: Callable[[int], nn.Module],
        conv_type: str,
    ):
        super().__init__()
        self.gate = gate
        self.conv1 = _make_conv(in_channels, out_channels, conv_type)
        self.bn1 = norm(out_channels)
        self.conv2 = _make_conv(out_channels, out_channels, conv_type)
        self.bn2 = norm(out_channels)
        self.downsample = downsample

    # 函数作用：执行当前网络层或完整 UPAL 模型的前向计算。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - tensor：tensor 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    # 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(tensor)
        output = self.gate(self.bn1(self.conv1(tensor)))
        output = self.bn2(self.conv2(output))
        return self.gate(output + identity)


class AlikedEncoder(nn.Module):
    """ALIKED-style full-resolution, multi-scale feature encoder."""

    # 函数作用：初始化对象配置、网络层和所需状态。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - config：config 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    # 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
    def __init__(self, config: dict[str, int]):
        super().__init__()
        c1, c2, c3, c4, dim = (config[k] for k in ("c1", "c2", "c3", "c4", "dim"))
        gate = nn.SELU(inplace=True)
        norm = nn.BatchNorm2d

        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool4 = nn.AvgPool2d(4, 4)
        self.norm = norm
        self.gate = gate
        self.block1 = ConvBlock(3, c1, gate, norm, "conv")
        self.block2 = ResBlock(c1, c2, nn.Conv2d(c1, c2, 1), gate, norm, "conv")
        self.block3 = ResBlock(c2, c3, nn.Conv2d(c2, c3, 1), gate, norm, "dcn")
        self.block4 = ResBlock(c3, c4, nn.Conv2d(c3, c4, 1), gate, norm, "dcn")
        self.conv1 = nn.Conv2d(c1, dim // 4, 1, bias=False)
        self.conv2 = nn.Conv2d(c2, dim // 4, 1, bias=False)
        self.conv3 = nn.Conv2d(c3, dim // 4, 1, bias=False)
        self.conv4 = nn.Conv2d(dim, dim // 4, 1, bias=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.upsample32 = nn.Upsample(scale_factor=32, mode="bilinear", align_corners=True)

    # 函数作用：执行当前网络层或完整 UPAL 模型的前向计算。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - image：输入图像张量或 NumPy 图像。
    # 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(image)
        x2 = self.block2(self.pool2(x1))
        x3 = self.block3(self.pool4(x2))
        x4 = self.block4(self.pool4(x3))
        return torch.cat(
            [
                self.gate(self.conv1(x1)),
                self.upsample2(self.gate(self.conv2(x2))),
                self.upsample8(self.gate(self.conv3(x3))),
                self.upsample32(self.gate(self.conv4(x4))),
            ],
            dim=1,
        )


class ScoreHead(nn.Module):
    # 函数作用：初始化对象配置、网络层和所需状态。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - input_dim：input_dim 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    # 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
    def __init__(self, input_dim: int):
        super().__init__()
        gate = nn.SELU(inplace=True)
        self.gate = gate
        self.score_head = nn.Sequential(
            nn.Conv2d(input_dim, 8, 1, bias=False),
            gate,
            nn.Conv2d(8, 4, 3, padding=1, bias=False),
            gate,
            nn.Conv2d(4, 4, 3, padding=1, bias=False),
            gate,
            nn.Conv2d(4, 1, 3, padding=1, bias=False),
        )

    # 函数作用：执行当前网络层或完整 UPAL 模型的前向计算。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - tensor：tensor 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    # 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.score_head(tensor))


# 函数作用：从特征图采样关键点邻域补丁。
# 所属模块：共享编码器与点线预测网络。
# 输入：
#   - tensor：tensor 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
#   - required_corners：required_corners 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
#   - patch_size：尺寸或数量上限。
# 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
def _get_patches(
    tensor: torch.Tensor, required_corners: torch.Tensor, patch_size: int
) -> torch.Tensor:
    channels, height, width = tensor.shape
    corner = (required_corners - patch_size / 2 + 1).long()
    corner[:, 0] = corner[:, 0].clamp(min=0, max=width - 1 - patch_size)
    corner[:, 1] = corner[:, 1].clamp(min=0, max=height - 1 - patch_size)
    offset = torch.arange(patch_size, device=corner.device)
    x, y = torch.meshgrid(offset, offset, indexing="ij")
    patches = torch.stack((x, y)).permute(2, 1, 0).unsqueeze(2)
    patches = patches + corner[None, None]
    points = patches.reshape(-1, 2)
    sampled = tensor.permute(1, 2, 0)[tuple(points.T)[::-1]]
    sampled = sampled.reshape(patch_size, patch_size, -1, channels)
    return sampled.permute(2, 3, 0, 1)


class DescriptorHead(nn.Module):
    """Sparse deformable descriptor head (SDDH)."""

    # 函数作用：初始化对象配置、网络层和所需状态。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - dims：dims 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    #   - kernel_size：尺寸或数量上限。
    #   - num_samples：num_samples 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    # 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
    def __init__(self, dims: int, kernel_size: int = 3, num_samples: int = 16):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_pos = num_samples
        self.offset_conv = nn.Sequential(
            nn.Conv2d(dims, 2 * num_samples, kernel_size, bias=True),
            nn.SELU(inplace=True),
            nn.Conv2d(2 * num_samples, 2 * num_samples, 1, bias=True),
        )
        self.sf_conv = nn.Conv2d(dims, dims, 1, bias=False)
        self.agg_weights = nn.Parameter(torch.rand(num_samples, dims, dims))

    # 函数作用：执行当前网络层或完整 UPAL 模型的前向计算。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - feature_map：feature_map 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
    #   - keypoints：像素坐标顺序为 (x,y) 的关键点。
    # 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
    def forward(
        self, feature_map: torch.Tensor, keypoints: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        batch, channels, height, width = feature_map.shape
        wh = torch.tensor([[width - 1, height - 1]], device=feature_map.device)
        max_offset = max(height, width) / 4.0
        descriptors = []

        for index in range(batch):
            features = feature_map[index]
            points = keypoints[index]
            points_pixel = (points / 2 + 0.5) * wh
            count = len(points)
            patches = _get_patches(features, points_pixel.long(), self.kernel_size)
            offsets = self.offset_conv(patches).clamp(-max_offset, max_offset)
            offsets = offsets[:, :, 0, 0].view(count, 2, self.n_pos).permute(0, 2, 1)

            positions = points_pixel.unsqueeze(1) + offsets
            positions = 2.0 * positions / wh[None] - 1
            positions = positions.reshape(1, count * self.n_pos, 1, 2)
            sampled = F.grid_sample(
                features.unsqueeze(0), positions, mode="bilinear", align_corners=True
            )
            sampled = sampled.reshape(channels, count, self.n_pos, 1).permute(1, 0, 2, 3)
            sampled = torch.selu_(self.sf_conv(sampled)).squeeze(-1)
            desc = torch.einsum("ncp,pcd->nd", sampled, self.agg_weights)
            descriptors.append(F.normalize(desc, p=2.0, dim=1))
        return descriptors


# 函数作用：对关键点热力图执行局部非极大值抑制。
# 所属模块：共享编码器与点线预测网络。
# 输入：
#   - scores：关键点、候选或匹配分数。
#   - radius：邻域或线段半径。
# 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
def _simple_nms(scores: torch.Tensor, radius: int) -> torch.Tensor:
    zeros = torch.zeros_like(scores)
    max_mask = scores == F.max_pool2d(scores, 2 * radius + 1, stride=1, padding=radius)
    for _ in range(2):
        suppression = F.max_pool2d(
            max_mask.float(), 2 * radius + 1, stride=1, padding=radius
        ) > 0
        suppressed_scores = torch.where(suppression, zeros, scores)
        new_max = suppressed_scores == F.max_pool2d(
            suppressed_scores, 2 * radius + 1, stride=1, padding=radius
        )
        max_mask = max_mask | (new_max & ~suppression)
    return torch.where(max_mask, scores, zeros)


class KeypointDetector(nn.Module):
    """Non-maximum suppression and differentiable sub-pixel refinement."""

    # 函数作用：初始化对象配置、网络层和所需状态。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - radius：邻域或线段半径。
    #   - max_keypoints：关键点坐标或索引。
    # 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
    def __init__(self, radius: int = 3, max_keypoints: int = 1024):
        super().__init__()
        self.radius = radius
        self.max_keypoints = max_keypoints
        self.temperature = 0.1
        kernel_size = 2 * radius + 1
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=radius)
        axis = torch.linspace(-radius, radius, kernel_size)
        self.register_buffer(
            "hw_grid",
            torch.stack(torch.meshgrid(axis, axis, indexing="ij")).view(2, -1).t()[:, [1, 0]],
            persistent=False,
        )

    # 函数作用：执行当前网络层或完整 UPAL 模型的前向计算。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - scores：关键点、候选或匹配分数。
    # 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
    def forward(
        self, scores: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        batch, _, height, width = scores.shape
        detached = scores.detach()
        nms_scores = _simple_nms(detached, self.radius)
        nms_scores[..., : self.radius, :] = 0
        nms_scores[..., -self.radius :, :] = 0
        nms_scores[..., :, : self.radius] = 0
        nms_scores[..., :, -self.radius :] = 0
        count = min(self.max_keypoints, height * width)
        indices = torch.topk(nms_scores.reshape(batch, -1), count).indices
        patches = self.unfold(scores)
        wh = torch.tensor([width - 1, height - 1], device=scores.device)

        keypoints, keypoint_scores, dispersities = [], [], []
        for batch_index in range(batch):
            point_indices = indices[batch_index]
            patch_scores = patches[batch_index].t()[point_indices]
            integer_points = torch.stack(
                [
                    point_indices % width,
                    torch.div(point_indices, width, rounding_mode="trunc"),
                ],
                dim=1,
            )
            maximum = patch_scores.max(dim=1).values.detach()[:, None]
            weights = ((patch_scores - maximum) / self.temperature).exp()
            residual = weights @ self.hw_grid / weights.sum(dim=1, keepdim=True)
            grid_distance = (
                torch.norm(
                    (self.hw_grid[None] - residual[:, None]) / self.radius, dim=-1
                )
                ** 2
            )
            dispersity = (weights * grid_distance).sum(dim=1) / weights.sum(dim=1)
            points = (integer_points + residual) / wh * 2 - 1
            point_scores = F.grid_sample(
                scores[batch_index].unsqueeze(0),
                points.view(1, 1, -1, 2),
                mode="bilinear",
                align_corners=True,
            )[0, 0, 0]
            keypoints.append(points)
            keypoint_scores.append(point_scores)
            dispersities.append(dispersity)
        return keypoints, keypoint_scores, dispersities


class UPAL(nn.Module):
    """Joint keypoint, descriptor, and line-distance-field inference network."""

    # 函数作用：初始化对象配置、网络层和所需状态。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - max_num_keypoints：每幅图像最多输出的关键点数。
    #   - nms_radius：邻域或线段半径。
    #   - line_neighborhood：线段、线端点或线索引。
    # 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
    def __init__(
        self,
        max_num_keypoints: int = 1024,
        nms_radius: int = 3,
        line_neighborhood: float = 5.0,
    ):
        super().__init__()
        dim = MODEL_CONFIG["dim"]
        self.line_neighborhood = line_neighborhood
        self.encoder_backbone = AlikedEncoder(MODEL_CONFIG)
        self.keypoint_and_junction_branch = ScoreHead(dim)
        self.descriptor_branch = DescriptorHead(
            dim,
            MODEL_CONFIG["descriptor_kernel_size"],
            MODEL_CONFIG["descriptor_samples"],
        )
        self.distance_field_branch = nn.Sequential(
            nn.Conv2d(dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(),
        )
        self.keypoint_detector = KeypointDetector(nms_radius, max_num_keypoints)

    # 函数作用：执行当前网络层或完整 UPAL 模型的前向计算。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - image：输入图像张量或 NumPy 图像。
    # 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from an image tensor shaped ``B x C x H x W`` in [0, 1]."""
        if image.ndim != 4 or image.shape[1] not in (1, 3):
            raise ValueError("image must have shape B x 1 x H x W or B x 3 x H x W")
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        _, _, height, width = image.shape
        padder = InputPadder(height, width, divisor=32)
        raw_features = self.encoder_backbone(padder.pad(image))
        score_map = padder.unpad(self.keypoint_and_junction_branch(raw_features))
        features = padder.unpad(F.normalize(raw_features, p=2, dim=1))
        distance_field = torch.exp(-self.distance_field_branch(features)) * self.line_neighborhood

        keypoints_normalized, scores, dispersities = self.keypoint_detector(score_map)
        descriptors = self.descriptor_branch(features, keypoints_normalized)
        wh = torch.tensor([width - 1, height - 1], device=image.device)
        keypoints = [wh * (points + 1.0) / 2.0 for points in keypoints_normalized]

        return {
            "keypoints": torch.stack(keypoints),
            "keypoint_scores": torch.stack(scores),
            "keypoint_dispersity": torch.stack(dispersities),
            "descriptors": torch.stack(descriptors),
            "keypoint_heatmap": score_map[:, 0],
            "line_distance_field": distance_field[:, 0],
        }

    # 函数作用：在指定像素坐标采样并归一化描述子。
    # 所属模块：共享编码器与点线预测网络。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - image：输入图像张量或 NumPy 图像。
    #   - keypoints：像素坐标顺序为 (x,y) 的关键点。
    # 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
    def describe_keypoints(self, image: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """Extract descriptors at pixel-space coordinates shaped ``B x N x 2``.

        This is useful for describing externally supplied locations, such as
        endpoints of line segments detected from the model outputs.
        """
        if image.ndim != 4 or image.shape[1] not in (1, 3):
            raise ValueError("image must have shape B x 1 x H x W or B x 3 x H x W")
        if keypoints.ndim != 3 or keypoints.shape[:1] != image.shape[:1] or keypoints.shape[-1] != 2:
            raise ValueError("keypoints must have shape B x N x 2 matching the image batch")
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        _, _, height, width = image.shape
        padder = InputPadder(height, width, divisor=32)
        features = padder.unpad(F.normalize(self.encoder_backbone(padder.pad(image)), p=2, dim=1))
        wh = keypoints.new_tensor((width - 1, height - 1))
        normalized_keypoints = 2.0 * keypoints / wh - 1.0
        return torch.stack(self.descriptor_branch(features, list(normalized_keypoints)))


# 函数作用：构建 UPAL、加载检查点并切换到推理模式。
# 所属模块：共享编码器与点线预测网络。
# 输入：
#   - checkpoint_path：UPAL 模型检查点路径。
#   - device：模型和张量所在设备。
#   - max_num_keypoints：每幅图像最多输出的关键点数。
# 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
def load_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    max_num_keypoints: int = 1024,
) -> UPAL:
    """Build UPAL and load the validated inference subset of a training checkpoint."""
    device = torch.device(device)
    model = UPAL(max_num_keypoints=max_num_keypoints)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    state = {key.split("extractor.")[-1]: value for key, value in state.items()}
    expected = model.state_dict()
    inference_state = {key: value for key, value in state.items() if key in expected}

    mismatched = {
        key: (tuple(value.shape), tuple(expected[key].shape))
        for key, value in inference_state.items()
        if value.shape != expected[key].shape
    }
    if mismatched:
        raise RuntimeError(f"Checkpoint contains incompatible tensor shapes: {mismatched}")
    missing = sorted(set(expected) - set(inference_state))
    if missing:
        raise RuntimeError(f"Checkpoint is missing inference weights: {missing}")

    model.load_state_dict(inference_state, strict=True)
    return model.to(device).eval()
