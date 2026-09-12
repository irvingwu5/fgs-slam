# 文件作用：arguments；隶属于 FGS-SLAM 的arguments模块。
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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    # 功能：初始化对象状态、配置和所需资源。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - parser：parser 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - name：name 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - fill_none：fill_none 所表示的计算输入，具体类型、形状和约束见调用上下文。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    # 功能：执行 extract 对应的计算或状态操作。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - args：命令行参数或调用参数集合。
    # 输出：返回计算、读取或组装得到的结果。
    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    # 功能：初始化对象状态、配置和所需资源。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - parser：parser 所表示的计算输入，具体类型、形状和约束见调用上下文。
    #   - sentinel：sentinel 所表示的计算输入，具体类型、形状和约束见调用上下文。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = 0    # 4
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    # 功能：执行 extract 对应的计算或状态操作。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - args：命令行参数或调用参数集合。
    # 输出：返回计算、读取或组装得到的结果。
    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    # 功能：初始化对象状态、配置和所需资源。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - parser：parser 所表示的计算输入，具体类型、形状和约束见调用上下文。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    # 功能：初始化对象状态、配置和所需资源。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    #   - parser：parser 所表示的计算输入，具体类型、形状和约束见调用上下文。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0000016 # 0.000016
        self.position_lr_final = 0.0000016 # 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 10_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05  # 0.05
        self.scaling_lr = 0.001 # 0.005
        self.rotation_lr = 0.001 # 0.001
        self.percent_dense = 0.01 # 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100 # 100
        self.opacity_reset_interval = 600 # 3000
        self.densify_from_iter = 300    #500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002 # 0.0002

        # for slam
        self.per_frame_iteration = 1
        self.downsample_rate = 10
        self.viewer_fps = 10.0
        self.max_correspondence_distance = 0.05
        self.keyframe_freq = 30
        self.train = True

        super().__init__(parser, "Optimization Parameters")

# 功能：获取与 get_combined_args 对应的数据或状态。
# 输入：
#   - parser：parser 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

class SLAMParameters():
    # 功能：初始化对象状态、配置和所需资源。
    # 输入：
    #   - self：当前类实例，提供并更新对象状态。
    # 输出：无显式返回值；输出体现为状态或外部资源更新。
    def __init__(self):
        ## Model parameters ##
        self.sh_degree = 0  # 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = 0    # 4
        self.white_background = False
        self.data_device = "cuda"
        self.eval = False

        ## Pipeline parameters ##
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

        ## Optimization parameters ##
        self.iterations = 30_000
        self.position_lr_init = 0.000001 # 0.0000016
        self.position_lr_final = 0.000001# 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 10_000
        self.feature_lr = 0.0035    # 0.0025
        self.opacity_lr = 0.05  # 0.05
        self.scaling_lr = 0.005 # 0.005 / best : 0.01(32.66)
        self.rotation_lr = 0.01 # 0.001
        self.percent_dense = 0.01 # 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100 # 100
        self.opacity_reset_interval = 600 # 3000
        self.densify_from_iter = 300    #500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002 # 0.0002

        # for slam
        self.per_frame_iteration = 1
        # self.downsample_rate = 10       # tum:5, replica:10
        self.viewer_fps = 10.0
        # self.max_correspondence_distance = 0.25
        self.keyframe_freq = 10 # replica : 10, tum : 10
        self.train = True
        self.training_stage=0
        self.loop_closure_config = None
