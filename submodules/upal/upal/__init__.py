# 文件作用：提供包接口导出相关实现；隶属于包接口导出模块。
"""UPAL point-and-line feature extraction."""

from .model import UPAL, load_model

__all__ = ["UPAL", "load_model"]
