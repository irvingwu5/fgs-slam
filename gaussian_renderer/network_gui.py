# 文件作用：gaussian_renderer；隶属于 FGS-SLAM 的gaussian_renderer模块。
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
import traceback
import socket
import json
from scene.cameras import MiniCam

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 功能：执行 init 对应的计算或状态操作。
# 输入：
#   - wish_host：wish_host 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - wish_port：wish_port 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：无显式返回值；输出体现为状态或外部资源更新。
def init(wish_host, wish_port):
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)

# 功能：执行 try_connect 对应的计算或状态操作。
# 输入：无。
# 输出：无显式返回值；输出体现为状态或外部资源更新。
def try_connect():
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        conn.settimeout(None)
    except Exception as inst:
        pass
            
# 功能：读取与 read 对应的数据或状态。
# 输入：无。
# 输出：返回计算、读取或组装得到的结果。
def read():
    global conn
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, 'little')
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))

# 功能：发送与 send 对应的数据或状态。
# 输入：
#   - message_bytes：message_bytes 所表示的计算输入，具体类型、形状和约束见调用上下文。
#   - verify：verify 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：无显式返回值；输出体现为状态或外部资源更新。
def send(message_bytes, verify):
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, 'little'))
    conn.sendall(bytes(verify, 'ascii'))

# 功能：接收并解析与 receive 对应的数据或状态。
# 输入：无。
# 输出：返回计算、读取或组装得到的结果。
def receive():
    message = read()

    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        try:
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            do_shs_python = bool(message["shs_python"])
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        except Exception as e:
            print("")
            traceback.print_exc()
            raise e
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        return None, None, None, None, None, None
    
# 功能：接收并解析与 receive_with_initial_pose 对应的数据或状态。
# 输入：
#   - init_pose：init_pose 所表示的计算输入，具体类型、形状和约束见调用上下文。
# 输出：返回计算、读取或组装得到的结果。
def receive_with_initial_pose(init_pose):
    message = read()
    init_pose = torch.tensor(init_pose).float().cuda()
    init_pose = init_pose.inverse()
    init_pose[:3,:3] = init_pose[:3,:3].t()
    
    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        try:
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            do_shs_python = bool(message["shs_python"])
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        except Exception as e:
            print("")
            traceback.print_exc()
            raise e
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        return None, None, None, None, None, None