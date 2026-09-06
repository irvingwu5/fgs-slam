# 文件作用：scene；隶属于 FGS-SLAM 的scene模块。
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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch import nn

class Scene(nn.Module):

    gaussians : GaussianModel

    # 功能：初始化对象状态、配置和所需资源。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - args：命令行参数或调用参数集合。
    #   - load_iteration：load_iteration 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - shuffle：shuffle 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - resolution_scales：尺度参数或倍率。
    #   - slam_trigger：slam_trigger 所表示的计算输入，具体类型、形状和约束见调用上下文。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def __init__(self, args : ModelParams, load_iteration=None, shuffle=True, resolution_scales=[1.0], slam_trigger=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        super().__init__()
        self.model_path = args.model_path
        self.loaded_iter = None
        self.slam = slam_trigger
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        scene_info = None
        
        if self.slam:
            print("SLAM mode")
            scene_info = sceneLoadTypeCallbacks["SLAM"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            if not self.slam:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            # if scene_info.test_cameras:
            #     camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            # for id, cam in enumerate(camlist):
            #     json_cams.append(camera_to_JSON(id, cam))
            # with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            #     json.dump(json_cams, file)


        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            # print("Loading Test Cameras")
            # self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        del scene_info
        
    # 功能：保存与 save 对应的数据或状态。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - iteration：当前迭代编号。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    # 功能：获取与 getTrainCameras 对应的数据或状态。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - scale：尺度参数或倍率。
    # 输出：返回计算、读取或组装得到的结果。
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    # 功能：获取与 getTestCameras 对应的数据或状态。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - scale：尺度参数或倍率。
    # 输出：返回计算、读取或组装得到的结果。
    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]