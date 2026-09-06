# 文件作用：utils；隶属于 FGS-SLAM 的utils模块。
"""
PyTorch dataset classes for GradSLAM v1.0.

The base dataset class now loads one sequence at a time
(opposed to v0.1.0 which loads multiple sequences).

A few parts of this code are adapted from NICE-SLAM
https://github.com/cvg/nice-slam/blob/645b53af3dc95b4b348de70e759943f7228a61ca/src/utils/datasets.py
"""

import abc
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import imageio
import numpy as np
import torch
import yaml
# from natsort import natsorted

from .geometryutils import relative_transformation
from . import datautils


# 功能：执行 to_scalar 对应的计算或状态操作。
# 输入：
#   - inp：inp 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def to_scalar(inp: Union[np.ndarray, torch.Tensor, float]) -> Union[int, float]:
    """
    Convert the input to a scalar
    """
    if isinstance(inp, float):
        return inp

    if isinstance(inp, np.ndarray):
        assert inp.size == 1
        return inp.item()

    if isinstance(inp, torch.Tensor):
        assert inp.numel() == 1
        return inp.item()


# 功能：执行 as_intrinsics_matrix 对应的计算或状态操作。
# 输入：
#   - intrinsics：intrinsics 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


# 功能：执行 from_intrinsics_matrix 对应的计算或状态操作。
# 输入：
#   - K：K 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def from_intrinsics_matrix(K):
    """
    Get fx, fy, cx, cy from the intrinsics matrix

    return 4 scalars
    """
    fx = to_scalar(K[0, 0])
    fy = to_scalar(K[1, 1])
    cx = to_scalar(K[0, 2])
    cy = to_scalar(K[1, 2])
    return fx, fy, cx, cy


# 功能：读取与 readEXR_onlydepth 对应的数据或状态。
# 输入：
#   - filename：filename 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header["dataWindow"]
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header["channels"]:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if "Y" not in header["channels"] else channelData["Y"]

    return Y


class GradSLAMDataset(torch.utils.data.Dataset):
    # 功能：初始化对象状态、配置和所需资源。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - config_dict：config_dict 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - stride：stride 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - start：start 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - end：end 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - desired_height：图像高度。
    #   - desired_width：图像宽度。
    #   - channels_first：channels_first 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - normalize_color：颜色值或颜色张量。
    #   - device：张量计算设备。
    #   - dtype：dtype 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - load_embeddings：load_embeddings 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - embedding_dir：输入或输出目录。
    #   - embedding_dim：embedding_dim 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - relative_pose：relative_pose 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - kwargs：额外关键字参数。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def __init__(
        self,
        config_dict,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        device="cuda:0",
        dtype=torch.float,
        load_embeddings: bool = False,
        embedding_dir: str = "feat_lseg_240_320",
        embedding_dim: int = 512,
        relative_pose: bool = True,  # If True, the pose is relative to the first frame
        **kwargs,
    ):
        super().__init__()
        self.name = config_dict["dataset_name"]
        self.device = device
        self.png_depth_scale = config_dict["camera_params"]["png_depth_scale"]

        self.orig_height = config_dict["camera_params"]["image_height"]
        self.orig_width = config_dict["camera_params"]["image_width"]
        self.fx = config_dict["camera_params"]["fx"]
        self.fy = config_dict["camera_params"]["fy"]
        self.cx = config_dict["camera_params"]["cx"]
        self.cy = config_dict["camera_params"]["cy"]

        self.dtype = dtype

        self.desired_height = desired_height
        self.desired_width = desired_width
        self.height_downsample_ratio = float(self.desired_height) / self.orig_height
        self.width_downsample_ratio = float(self.desired_width) / self.orig_width
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        self.load_embeddings = load_embeddings
        self.embedding_dir = embedding_dir
        self.embedding_dim = embedding_dim
        self.relative_pose = relative_pose

        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(stride))
        if not (end == -1 or end > start):
            raise ValueError("end ({0}) must be -1 (use all images) or greater than start ({1})".format(end, start))

        self.distortion = (
            np.array(config_dict["camera_params"]["distortion"])
            if "distortion" in config_dict["camera_params"]
            else None
        )
        self.crop_size = (
            config_dict["camera_params"]["crop_size"] if "crop_size" in config_dict["camera_params"] else None
        )

        self.crop_edge = None
        if "crop_edge" in config_dict["camera_params"].keys():
            self.crop_edge = config_dict["camera_params"]["crop_edge"]

        self.color_paths, self.depth_paths, self.embedding_paths = self.get_filepaths()
        if len(self.color_paths) != len(self.depth_paths):
            raise ValueError("Number of color and depth images must be the same.")
        if self.load_embeddings:
            if len(self.color_paths) != len(self.embedding_paths):
                raise ValueError("Mismatch between number of color images and number of embedding files.")
        self.num_imgs = len(self.color_paths)
        self.poses = self.load_poses()

        if self.end == -1:
            self.end = self.num_imgs

        self.color_paths = self.color_paths[self.start : self.end : stride]
        self.depth_paths = self.depth_paths[self.start : self.end : stride]
        if self.load_embeddings:
            self.embedding_paths = self.embedding_paths[self.start : self.end : stride]
        self.poses = self.poses[self.start : self.end : stride]
        # Tensor of retained indices (indices of frames and poses that were retained)
        self.retained_inds = torch.arange(self.num_imgs)[self.start : self.end : stride]
        # Update self.num_images after subsampling the dataset
        self.num_imgs = len(self.color_paths)

        # self.transformed_poses = datautils.poses_to_transforms(self.poses)
        self.poses = torch.stack(self.poses)
        if self.relative_pose:
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses

    # 功能：执行 __len__ 对应的计算或状态操作。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    # 输出：返回计算、读取或组装得到的结果。
    def __len__(self):
        return self.num_imgs

    # 功能：获取与 get_filepaths 对应的数据或状态。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def get_filepaths(self):
        """Return paths to color images, depth images. Implement in subclass."""
        raise NotImplementedError

    # 功能：加载与 load_poses 对应的数据或状态。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def load_poses(self):
        """Load camera poses. Implement in subclass."""
        raise NotImplementedError

    # 功能：执行 _preprocess_color 对应的计算或状态操作。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - color：颜色值或颜色张量。
    # 输出：返回计算、读取或组装得到的结果。
    def _preprocess_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Retruns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.normalize_color:
            color = datautils.normalize_image(color)
        if self.channels_first:
            color = datautils.channels_first(color)
        return color

    # 功能：执行 _preprocess_depth 对应的计算或状态操作。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - depth：输入深度数据。
    # 输出：返回计算、读取或组装得到的结果。
    def _preprocess_depth(self, depth: np.ndarray):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = datautils.channels_first(depth)
        return depth / self.png_depth_scale

    # 功能：执行 _preprocess_poses 对应的计算或状态操作。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - poses：相机位姿序列。
    # 输出：返回计算、读取或组装得到的结果。
    def _preprocess_poses(self, poses: torch.Tensor):
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )

    # 功能：获取与 get_cam_K 对应的数据或状态。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    # 输出：返回计算、读取或组装得到的结果。
    def get_cam_K(self):
        """
        Return camera intrinsics matrix K

        Returns:
            K (torch.Tensor): Camera intrinsics matrix, of shape (3, 3)
        """
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        return K

    # 功能：读取与 read_embedding_from_file 对应的数据或状态。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - embedding_path：embedding_path 所表示的计算输入，具体类型、形状和约束见调用上下文。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def read_embedding_from_file(self, embedding_path: str):
        """
        Read embedding from file and process it. To be implemented in subclass for each dataset separately.
        """
        raise NotImplementedError

    # 功能：执行 __getitem__ 对应的计算或状态操作。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - index：index 所表示的计算输入，具体类型、形状和约束见调用上下文。
    # 输出：返回计算、读取或组装得到的结果。
    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)
        if ".png" in depth_path:
            # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
        elif ".exr" in depth_path:
            depth = readEXR_onlydepth(depth_path)

        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, K, self.distortion)

        color = torch.from_numpy(color)
        K = torch.from_numpy(K)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        K = datautils.scale_intrinsics(K, self.height_downsample_ratio, self.width_downsample_ratio)
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                embedding.to(self.device),  # Allow embedding to be another dtype
                # self.retained_inds[index].item(),
            )

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
            # self.retained_inds[index].item(),
        )
