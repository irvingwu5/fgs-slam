# 文件作用：跟踪/建图进程通信；隶属于 FGS-SLAM 的跟踪/建图进程通信模块。
import torch
import numpy as np
import cv2
import torch.nn as nn
import copy
import math


# 功能：获取与 getWorld2View2 对应的数据或状态。
# 输入：
#   - R：旋转矩阵。
#   - t：t 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - translate：translate 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - scale：尺度参数或倍率。
# 输出：返回计算、读取或组装得到的结果。
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.t()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = Rt.inverse()
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = C2W.inverse()
    return Rt


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


class SharedPoints(nn.Module):
    # 功能：初始化对象状态、配置和所需资源。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - num_points：三维点云或点属性。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def __init__(self, num_points):
        super().__init__()
        self.points = torch.zeros((num_points, 3)).float()
        self.colors = torch.zeros((num_points, 3)).float()
        self.z_values = torch.zeros((num_points)).float()
        self.filter = torch.zeros((num_points)).int()
        self.using_idx = torch.zeros((1)).int()
        self.filter_size = torch.zeros((1)).int()

    # 功能：执行 input_values 对应的计算或状态操作。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - new_points：三维点云或点属性。
    #   - new_colors：颜色值或颜色张量。
    #   - new_z_values：new_z_values 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - new_filter：筛选索引、掩码或阈值。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def input_values(self, new_points, new_colors, new_z_values, new_filter):
        self.using_idx[0] = new_points.shape[0]
        self.points[:self.using_idx[0], :] = new_points
        self.colors[:self.using_idx[0], :] = new_colors
        self.z_values[:self.using_idx[0]] = new_z_values

        self.filter_size[0] = new_filter.shape[0]
        self.filter[:self.filter_size[0]] = new_filter

    # 功能：获取与 get_values 对应的数据或状态。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    # 输出：返回计算、读取或组装得到的结果。
    def get_values(self):
        return copy.deepcopy(self.points[:self.using_idx[0], :].numpy()), \
            copy.deepcopy(self.colors[:self.using_idx[0], :].numpy()), \
            copy.deepcopy(self.z_values[:self.using_idx[0]].numpy()), \
            copy.deepcopy(self.filter[:self.filter_size[0]].numpy())


class SharedGaussians(nn.Module):
    # 功能：初始化对象状态、配置和所需资源。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - num_points：三维点云或点属性。
    #   - H：H 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - W：W 所表示的计算输入，具体类型、形状和约束见调用上下文。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def __init__(self, num_points, H, W):
        super().__init__()
        self.xyz = torch.zeros((num_points, 3)).float().cuda()
        self.colors = torch.zeros((num_points, 3)).float().cuda()
        self.rots = torch.zeros((num_points, 4)).float().cuda()
        self.scales = torch.zeros((num_points, 3)).float().cuda()
        self.opacities = torch.zeros((num_points), 1).float().cuda()
        self.z_values = torch.zeros((num_points)).float().cuda()
        self.tracking_mask = torch.zeros((num_points)).bool().cuda()
        self.zero_filter = torch.zeros((num_points)).long().cuda()
        self.using_idx = torch.zeros((1)).int().cuda()
        self.mask_size = torch.zeros((1)).int().cuda()
        self.zero_filter_size = torch.zeros((1)).int().cuda()
        self.current_pose = torch.ones((4, 4)).float().cuda()
        self.opacity_mask = torch.zeros((H * W)).bool().cuda()

    # 功能：执行 input_values_tracking 对应的计算或状态操作。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - new_xyz：new_xyz 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - new_colors：颜色值或颜色张量。
    #   - new_rots：旋转参数或矩阵。
    #   - new_scales：尺度参数或倍率。
    #   - new_z_values：new_z_values 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - new_tracking_mask：new_tracking_mask 所表示的计算输入，具体类型、形状和约束见调用上下文。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def input_values_tracking(self, new_xyz, new_colors, new_rots, new_scales, new_z_values, new_tracking_mask):
        # on CPU memory
        self.using_idx[0] = new_xyz.shape[0]
        self.xyz[:self.using_idx[0], :] = new_xyz
        self.colors[:self.using_idx[0], :] = new_colors
        self.rots[:self.using_idx[0], :] = new_rots
        self.scales[:self.using_idx[0], :] = new_scales
        self.z_values[:self.using_idx[0]] = new_z_values

        self.mask_size[0] = new_tracking_mask.shape[0]
        self.tracking_mask[:self.mask_size[0]] = new_tracking_mask

    # 功能：获取与 get_values_tracking 对应的数据或状态。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    # 输出：返回计算、读取或组装得到的结果。
    def get_values_tracking(self):
        return copy.deepcopy(self.xyz[:self.using_idx[0], :]), \
            copy.deepcopy(self.colors[:self.using_idx[0], :]), \
            copy.deepcopy(self.rots[:self.using_idx[0], :]), \
            copy.deepcopy(self.scales[:self.using_idx[0], :]), \
            copy.deepcopy(self.z_values[:self.using_idx[0]]), \
            copy.deepcopy(self.tracking_mask[:self.mask_size[0]])

    # 功能：执行 input_values 对应的计算或状态操作。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - new_xyz：new_xyz 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - new_colors：颜色值或颜色张量。
    #   - new_rots：旋转参数或矩阵。
    #   - new_scales：尺度参数或倍率。
    #   - new_opacity：高斯不透明度。
    #   - new_z_values：new_z_values 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - current_pose：current_pose 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - new_tracking_mask：new_tracking_mask 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - zero_filter：筛选索引、掩码或阈值。
    #   - opacity_mask：高斯不透明度。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def input_values(self, new_xyz, new_colors, new_rots, new_scales, new_opacity, new_z_values, current_pose,
                     new_tracking_mask, zero_filter, opacity_mask):
        # on CPU memory
        self.using_idx[0] = new_xyz.shape[0]
        self.xyz[:self.using_idx[0], :] = new_xyz
        self.colors[:self.using_idx[0], :] = new_colors
        self.rots[:self.using_idx[0], :] = new_rots
        self.scales[:self.using_idx[0], :] = new_scales
        self.opacities[:self.using_idx[0], :] = new_opacity
        self.z_values[:self.using_idx[0]] = new_z_values
        self.current_pose[:] = current_pose

        self.mask_size[0] = new_tracking_mask.shape[0]
        self.zero_filter_size[0] = zero_filter.shape[0]
        self.tracking_mask[:self.mask_size[0]] = new_tracking_mask
        self.zero_filter[:self.zero_filter_size[0]] = zero_filter
        self.opacity_mask[:] = opacity_mask
        a = 1

    # 功能：获取与 get_values 对应的数据或状态。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    # 输出：返回计算、读取或组装得到的结果。
    def get_values(self):
        return copy.deepcopy(self.xyz[:self.using_idx[0], :]), \
            copy.deepcopy(self.colors[:self.using_idx[0], :]), \
            copy.deepcopy(self.rots[:self.using_idx[0], :]), \
            copy.deepcopy(self.scales[:self.using_idx[0], :]), \
            copy.deepcopy(self.opacities[:self.using_idx[0], :]), \
            copy.deepcopy(self.z_values[:self.using_idx[0]]), \
            copy.deepcopy(self.current_pose), \
            copy.deepcopy(self.tracking_mask[:self.mask_size[0]]), \
            copy.deepcopy(self.zero_filter[:self.zero_filter_size[0]]), \
            copy.deepcopy(self.opacity_mask)


class SharedTargetPoints(nn.Module):
    # 功能：初始化对象状态、配置和所需资源。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - num_points：三维点云或点属性。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points
        self.xyz = torch.zeros((num_points, 3)).float()
        self.using_idx = torch.zeros((1)).int()

    # 功能：执行 input_values 对应的计算或状态操作。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - new_xyz：new_xyz 所表示的计算输入，具体类型、形状和约束见调用上下文。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def input_values(self, new_xyz):
        self.using_idx[0] = new_xyz.shape[0]
        if self.using_idx[0] > self.num_points:
            print("Too many target points")
        self.xyz[:self.using_idx[0], :] = new_xyz

    # 功能：获取与 get_values_tensor 对应的数据或状态。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    # 输出：返回计算、读取或组装得到的结果。
    def get_values_tensor(self):
        return copy.deepcopy(self.xyz[:self.using_idx[0], :])


    # 功能：获取与 get_values_np 对应的数据或状态。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    # 输出：返回计算、读取或组装得到的结果。
    def get_values_np(self):
        return copy.deepcopy(self.xyz[:self.using_idx[0], :].numpy())


class SharedCam(nn.Module):
    # 功能：初始化对象状态、配置和所需资源。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - FoVx：FoVx 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - FoVy：FoVy 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - image：输入图像。
    #   - depth_image：depth_image 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - cx：cx 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - cy：cy 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - fx：fx 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - fy：fy 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - trans：trans 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - scale：尺度参数或倍率。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def __init__(self, FoVx, FoVy, image, depth_image,
                 cx, cy, fx, fy,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        super().__init__()
        self.cam_idx = torch.zeros((1)).int()
        self.R = torch.eye(3, 3).float()
        self.t = torch.zeros((3)).float()
        self.FoVx = torch.tensor([FoVx])
        self.FoVy = torch.tensor([FoVy])
        self.image_width = torch.tensor([image.shape[1]])
        self.image_height = torch.tensor([image.shape[0]])
        self.cx = torch.tensor([cx])
        self.cy = torch.tensor([cy])
        self.fx = torch.tensor([fx])
        self.fy = torch.tensor([fy])
        self.current_pose = torch.eye(4, 4).float().cuda()

        self.original_image = torch.from_numpy(image).float().permute(2, 0, 1) / 255

        self.original_depth_image = torch.from_numpy(depth_image).float().unsqueeze(0)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = getWorld2View2(self.R, self.t, trans, scale).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    # 功能：更新与 update_matrix 对应的数据或状态。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def update_matrix(self):
        self.world_view_transform[:, :] = getWorld2View2(self.R, self.t, self.trans, self.scale).transpose(0, 1)
        self.full_proj_transform[:, :] = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center[:] = self.world_view_transform.inverse()[3, :3]

    # 功能：设置与 setup_cam 对应的数据或状态。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - R：旋转矩阵。
    #   - t：t 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - rgb_img：图像、图像路径或尺寸。
    #   - depth_img：图像、图像路径或尺寸。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def setup_cam(self, R, t, rgb_img, depth_img):
        # Set pose, projection matrix
        self.R[:, :] = torch.from_numpy(R)
        self.t[:] = torch.from_numpy(t)
        self.update_matrix()
        # Update image
        self.original_image[:, :, :] = torch.from_numpy(rgb_img).float().permute(2, 0, 1) / 255
        self.original_depth_image[:, :, :] = torch.from_numpy(depth_img).float().unsqueeze(0)

    # 功能：执行 on_cuda 对应的计算或状态操作。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def on_cuda(self):
        self.world_view_transform = self.world_view_transform.cuda()
        self.projection_matrix = self.projection_matrix.cuda()
        self.full_proj_transform = self.full_proj_transform.cuda()
        self.camera_center = self.camera_center.cuda()

        self.original_image = self.original_image.cuda()
        self.original_depth_image = self.original_depth_image.cuda()


class MappingCam(nn.Module):
    # 功能：初始化对象状态、配置和所需资源。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - cam_idx：cam_idx 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - R：旋转矩阵。
    #   - t：t 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - FoVx：FoVx 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - FoVy：FoVy 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - image：输入图像。
    #   - depth_image：depth_image 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - cx：cx 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - cy：cy 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - fx：fx 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - fy：fy 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - trans：trans 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - scale：尺度参数或倍率。
    #   - data_device：data_device 所表示的计算输入，具体类型、形状和约束见调用上下文。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def __init__(self, cam_idx, R, t, FoVx, FoVy, image, depth_image,
                 cx, cy, fx, fy,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"
                 ):
        super().__init__()
        self.cam_idx = cam_idx
        self.R = R
        self.t = t
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = image.shape[1]
        self.image_height = image.shape[0]
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.last_loss = 0.

        self.original_image = torch.from_numpy(image).float().cuda().permute(2, 0, 1) / 255

        self.original_depth_image = torch.from_numpy(depth_image).float().unsqueeze(0).cuda()

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, t, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    # 功能：更新与 update 对应的数据或状态。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def update(self):
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.t, self.trans, self.scale)).transpose(0,
                                                                                                                   1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]