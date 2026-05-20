

import math
import numpy as np
import torch
from rclpy.node import Node
from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessorLegacy as VuerPreprocessor
from constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from pathlib import Path
import argparse
import time
import yaml
from multiprocessing import (
    Array,
    Process,
    shared_memory,
    Queue,
    Manager,
    Event,
    Semaphore,
)
import json

cwidth = 640
cheight = 480


class X100_Sim_BraincoV2:
    def __init__(self, print_freq=False):
        self.print_freq = print_freq

        # initialize gym
        self.gym = gymapi.acquire_gym()

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        asset_root = "./assets"
        robot_asset_path = "x100_urdf_ros1/x100_38_brainco_fixed.urdf"
        # robot_asset_path = "h1_inspire/urdf/h1_inspire.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, robot_asset_path, asset_options
        )
        self.dof = self.gym.get_asset_dof_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        for i in range(len(self.dof_names)):
            print(f"{i}: {self.dof_names[i]}")

        # 获取关节名称，并筛选左右手臂索引
        self.left_arm_indices = [
            i for i, name in enumerate(self.dof_names) if "left" in name.lower()
        ]
        self.right_arm_indices = [
            i for i, name in enumerate(self.dof_names) if "right" in name.lower()
        ]

        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed(0)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0, 0.18)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.robot_handle = self.gym.create_actor(
            self.env, robot_asset, pose, "x100", 1, 1
        )
        kp = 40.0
        kd = 4.0
        dof_props = self.gym.get_actor_dof_properties(self.env, self.robot_handle)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"].fill(kp)
        dof_props["damping"].fill(kd)
        dof_props["effort"].fill(200.0)
        dof_props["hasLimits"] = True

        self.gym.set_actor_dof_properties(self.env, self.robot_handle, dof_props)
        self.gym.set_actor_dof_states(
            self.env,
            self.robot_handle,
            np.zeros(self.dof, gymapi.DofState.dtype),
            gymapi.STATE_ALL,
        )
        robot_idx = self.gym.get_actor_index(
            self.env, self.robot_handle, gymapi.DOMAIN_SIM
        )

        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)
        self.robot_root_states = self.root_states[robot_idx]

        # 获取手腕和头部的全局位置
        # create default viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(1, 1, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.cam_lookat_offset = np.array([1, 0, 0])
        self.left_cam_offset = np.array([0, 0.033, 0])
        self.right_cam_offset = np.array([0, -0.033, 0])
        self.cam_pos = np.array([0.05, 0, 1.38])

        # create left 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = cwidth
        camera_props.height = cheight
        self.left_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(
            self.left_camera_handle,
            self.env,
            gymapi.Vec3(*(self.cam_pos + self.left_cam_offset)),
            gymapi.Vec3(
                *(self.cam_pos + self.left_cam_offset + self.cam_lookat_offset)
            ),
        )

        # create right 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = cwidth
        camera_props.height = cheight
        self.right_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(
            self.right_camera_handle,
            self.env,
            gymapi.Vec3(*(self.cam_pos + self.right_cam_offset)),
            gymapi.Vec3(
                *(self.cam_pos + self.right_cam_offset + self.cam_lookat_offset)
            ),
        )

    def step(self, qpos):
        # print("qpos:", qpos)
        if self.print_freq:
            start = time.time()

        robot_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        robot_states["pos"] = qpos
        self.gym.set_actor_dof_states(
            self.env, self.robot_handle, robot_states, gymapi.STATE_POS
        )

        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        left_image = self.gym.get_camera_image(
            self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR
        )
        right_image = self.gym.get_camera_image(
            self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR
        )
        left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
        right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]

        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        if self.print_freq:
            end = time.time()
            print("Frequency:", 1 / (end - start))

        return left_image, right_image

    def _step(self, head_rmat, left_pose, right_pose, left_qpos, right_qpos):

        if self.print_freq:
            start = time.time()

        self.left_root_states[0:7] = torch.tensor(left_pose, dtype=float)
        self.right_root_states[0:7] = torch.tensor(right_pose, dtype=float)
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states)
        )

        left_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)

        left_states["pos"] = left_qpos
        self.gym.set_actor_dof_states(
            self.env, self.left_handle, left_states, gymapi.STATE_POS
        )

        right_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        right_states["pos"] = right_qpos
        self.gym.set_actor_dof_states(
            self.env, self.right_handle, right_states, gymapi.STATE_POS
        )

        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        curr_lookat_offset = self.cam_lookat_offset @ head_rmat.T
        curr_left_offset = self.left_cam_offset @ head_rmat.T
        curr_right_offset = self.right_cam_offset @ head_rmat.T

        self.gym.set_camera_location(
            self.left_camera_handle,
            self.env,
            gymapi.Vec3(*(self.cam_pos + curr_left_offset)),
            gymapi.Vec3(*(self.cam_pos + curr_left_offset + curr_lookat_offset)),
        )

        self.gym.set_camera_location(
            self.right_camera_handle,
            self.env,
            gymapi.Vec3(*(self.cam_pos + curr_right_offset)),
            gymapi.Vec3(*(self.cam_pos + curr_right_offset + curr_lookat_offset)),
        )

        left_image = self.gym.get_camera_image(
            self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR
        )
        right_image = self.gym.get_camera_image(
            self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR
        )
        left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
        right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]

        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        if self.print_freq:
            end = time.time()
            print("Frequency:", 1 / (end - start))

        return left_image, right_image

    def get_arm_state(self):
        # 获取特定机器人的关节状态
        dof_states = self.gym.get_actor_dof_states(
            self.env, self.robot_handle, gymapi.STATE_ALL
        )

        # 提取左臂关节状态
        arm_indices = np.concatenate([self.left_arm_indices, self.right_arm_indices])
        joint_positions = dof_states["pos"][arm_indices]
        joint_velocities = dof_states["vel"][arm_indices]

        # print("Joint Positions:", joint_positions)
        # print("Joint Velocities:", joint_velocities)

        return joint_positions, joint_velocities

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


class X100_Sim:
    def __init__(self, print_freq=False):
        self.print_freq = print_freq

        # initialize gym
        self.gym = gymapi.acquire_gym()

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        asset_root = "./assets"
        # robot_asset_path = "x100_urdf_ros1/x100_38.urdf"
        # robot_asset_path = "h1_inspire/urdf/h1_inspire.urdf"
        robot_asset_path = "k100_description/k100.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, robot_asset_path, asset_options
        )
        self.dof = self.gym.get_asset_dof_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        print(f"dof: {self.dof}")
        print(f"{self.dof_names}")

        # 获取关节名称，并筛选左右手臂索引
        self.left_arm_indices = [
            i for i, name in enumerate(self.dof_names) if "left" in name.lower()
        ]
        self.right_arm_indices = [
            i for i, name in enumerate(self.dof_names) if "right" in name.lower()
        ]

        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed(0)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0, 0.18)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.robot_handle = self.gym.create_actor(
            self.env, robot_asset, pose, "x100", 1, 1
        )
        kp = 40.0
        kd = 4.0
        dof_props = self.gym.get_actor_dof_properties(self.env, self.robot_handle)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"].fill(kp)
        dof_props["damping"].fill(kd)
        dof_props["effort"].fill(200.0)
        dof_props["hasLimits"] = True

        self.gym.set_actor_dof_properties(self.env, self.robot_handle, dof_props)
        self.gym.set_actor_dof_states(
            self.env,
            self.robot_handle,
            np.zeros(self.dof, gymapi.DofState.dtype),
            gymapi.STATE_ALL,
        )
        robot_idx = self.gym.get_actor_index(
            self.env, self.robot_handle, gymapi.DOMAIN_SIM
        )

        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)
        self.robot_root_states = self.root_states[robot_idx]
        # 获取手腕和头部的全局位置
        # wrist_global_pos = self.gym.get_actor_rigid_body_states(self.env, wrist_handle, gymapi.STATE_POS)
        # head_global_pos = self.gym.get_actor_rigid_body_states(self.env, headHandle, gymapi.STATE_POS)

        # 计算相对位置
        # relative_position = wrist_global_pos - head_global_pos
        # print("relative_position:",relative_position)

        # create default viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(1, 1, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.cam_lookat_offset = np.array([1, 0, 0])
        self.left_cam_offset = np.array([0, 0.033, 0])
        self.right_cam_offset = np.array([0, -0.033, 0])
        self.cam_pos = np.array([0.05, 0, 1.38])

        # create left 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = cwidth
        camera_props.height = cheight
        self.left_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(
            self.left_camera_handle,
            self.env,
            gymapi.Vec3(*(self.cam_pos + self.left_cam_offset)),
            gymapi.Vec3(
                *(self.cam_pos + self.left_cam_offset + self.cam_lookat_offset)
            ),
        )

        # create right 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = cwidth
        camera_props.height = cheight
        self.right_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(
            self.right_camera_handle,
            self.env,
            gymapi.Vec3(*(self.cam_pos + self.right_cam_offset)),
            gymapi.Vec3(
                *(self.cam_pos + self.right_cam_offset + self.cam_lookat_offset)
            ),
        )

    def step(self, qpos):

        if self.print_freq:
            start = time.time()

        robot_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        robot_states["pos"] = qpos
        self.gym.set_actor_dof_states(
            self.env, self.robot_handle, robot_states, gymapi.STATE_POS
        )

        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        left_image = self.gym.get_camera_image(
            self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR
        )
        right_image = self.gym.get_camera_image(
            self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR
        )
        left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
        right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]

        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        if self.print_freq:
            end = time.time()
            print("Frequency:", 1 / (end - start))

        return left_image, right_image

    def _step(self, head_rmat, left_pose, right_pose, left_qpos, right_qpos):

        if self.print_freq:
            start = time.time()

        self.left_root_states[0:7] = torch.tensor(left_pose, dtype=float)
        self.right_root_states[0:7] = torch.tensor(right_pose, dtype=float)
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states)
        )

        left_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)

        left_states["pos"] = left_qpos
        self.gym.set_actor_dof_states(
            self.env, self.left_handle, left_states, gymapi.STATE_POS
        )

        right_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        right_states["pos"] = right_qpos
        self.gym.set_actor_dof_states(
            self.env, self.right_handle, right_states, gymapi.STATE_POS
        )

        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        curr_lookat_offset = self.cam_lookat_offset @ head_rmat.T
        curr_left_offset = self.left_cam_offset @ head_rmat.T
        curr_right_offset = self.right_cam_offset @ head_rmat.T

        self.gym.set_camera_location(
            self.left_camera_handle,
            self.env,
            gymapi.Vec3(*(self.cam_pos + curr_left_offset)),
            gymapi.Vec3(*(self.cam_pos + curr_left_offset + curr_lookat_offset)),
        )
        self.gym.set_camera_location(
            self.right_camera_handle,
            self.env,
            gymapi.Vec3(*(self.cam_pos + curr_right_offset)),
            gymapi.Vec3(*(self.cam_pos + curr_right_offset + curr_lookat_offset)),
        )
        left_image = self.gym.get_camera_image(
            self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR
        )
        right_image = self.gym.get_camera_image(
            self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR
        )
        left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
        right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]

        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        if self.print_freq:
            end = time.time()
            print("Frequency:", 1 / (end - start))

        return left_image, right_image

    def get_arm_state(self):
        # 获取特定机器人的关节状态
        dof_states = self.gym.get_actor_dof_states(
            self.env, self.robot_handle, gymapi.STATE_ALL
        )
        # print("dof_states:", dof_states)
        # 提取左臂关节状态
        arm_indices = np.concatenate([self.left_arm_indices, self.right_arm_indices])
        joint_positions = dof_states["pos"][arm_indices]
        joint_velocities = dof_states["vel"][arm_indices]

        # print("Joint Positions:", joint_positions)
        # print("Joint Velocities:", joint_velocities)

        return joint_positions, joint_velocities

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


import time
import threading
import socket
import json

class X100_SocketClient:
    # 172.27.60.12
    def __init__(self, server_host='192.168.1.105', server_port=5556):
        # Socket客户端设置
        self.server_host = server_host
        self.server_port = server_port
        self.socket_client = None
        self.is_connected = False
        
        # 存储接收到的机器人状态
        self.robot_state = {
            'left_arm': None,
            'right_arm': None,
            'timestamp': None
        }
        
        # 存储控制命令
        self.control_command = {
            'arm_positions': None,  # 14个关节位置
            'left_hand_positions': None,  # 5个左手关节位置
            'right_hand_positions': None  # 5个右手关节位置
        }
        
        # 连接到服务端
        self.connect_to_server()
        
        # 启动状态接收线程
        self.receive_thread = threading.Thread(target=self.receive_state_data)
        self.receive_thread.daemon = True
        self.receive_thread.start()
        
        # 启动控制命令发送线程
        self.send_thread = threading.Thread(target=self.send_control_data)
        self.send_thread.daemon = True
        self.send_thread.start()
        
        print('机械臂控制客户端已启动')

    def connect_to_server(self):
        """连接到服务端"""
        try:
            self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_client.connect((self.server_host, self.server_port))
            self.socket_client.settimeout(1.0)  # 设置超时
            self.is_connected = True
            print(f'已连接到服务端 {self.server_host}:{self.server_port}')
        except Exception as e:
            print(f'连接服务端失败: {e}')
            self.is_connected = False
            # 尝试重新连接
            threading.Timer(5.0, self.connect_to_server).start()

    def receive_state_data(self):
        """接收服务端发送的状态数据"""
        buffer = ""
        while True:
            if not self.is_connected:
                time.sleep(1)
                continue
                
            try:
                data = self.socket_client.recv(4096).decode('utf-8')
                if not data:
                    print('服务端断开连接')
                    self.is_connected = False
                    self.connect_to_server()  # 尝试重连
                    continue
                    
                buffer += data
                
                # 处理完整的数据包（以换行符分隔）
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self.process_state_message(line.strip())
                        
            except socket.timeout:
                continue  # 超时是正常的，继续循环
            except Exception as e:
                print(f'接收数据错误: {e}')
                self.is_connected = False
                time.sleep(1)
                self.connect_to_server()  # 尝试重连

    def process_state_message(self, message):
        """处理状态消息"""
        try:
            state_data = json.loads(message)
            self.robot_state = state_data
            self.robot_state['timestamp'] = time.time()
            
            # 打印状态信息
            self.print_robot_state()
            
        except json.JSONDecodeError as e:
            print(f'JSON解析错误: {e}, 数据: {message}')
        except Exception as e:
            print(f'处理状态消息错误: {e}')

    def get_robot_joint_positions(self):
        """获取机器人关节位置，返回14个值的列表（左臂7个 + 右臂7个）"""
        left_pos = []
        right_pos = []
        
        # 获取左臂位置
        if self.robot_state['left_arm'] and 'positions' in self.robot_state['left_arm']:
            left_pos = self.robot_state['left_arm']['positions']
            # 确保只取前7个关节
            if len(left_pos) > 7:
                left_pos = left_pos[:7]
            elif len(left_pos) < 7:
                # 如果不足7个，用0填充
                left_pos.extend([0.0] * (7 - len(left_pos)))
        else:
            # 如果没有左臂数据，用0填充
            left_pos = [0.0] * 7
        
        # 获取右臂位置
        if self.robot_state['right_arm'] and 'positions' in self.robot_state['right_arm']:
            right_pos = self.robot_state['right_arm']['positions']
            # 确保只取前7个关节
            if len(right_pos) > 7:
                right_pos = right_pos[:7]
            elif len(right_pos) < 7:
                # 如果不足7个，用0填充
                right_pos.extend([0.0] * (7 - len(right_pos)))
        else:
            # 如果没有右臂数据，用0填充
            right_pos = [0.0] * 7
        
        # 合并左右臂位置
        joint_positions = left_pos + right_pos
        
        return joint_positions

    def print_robot_state(self):
        """打印机器人状态"""
        joint_positions = self.get_robot_joint_positions()
        
        # print(f"\n=== 机器人状态更新 ===")
        # print(f"左臂位置: {[f'{p:.3f}' for p in joint_positions[:7]]}")
        # print(f"右臂位置: {[f'{p:.3f}' for p in joint_positions[7:]]}")
        # print(f"更新时间: {time.strftime('%H:%M:%S', time.localtime(self.robot_state.get('timestamp', 0)))}")
        
        return joint_positions

    def set_control_command(self, arm_positions=None, left_hand_positions=None, right_hand_positions=None):
        """设置控制命令"""
        if arm_positions is not None:
            if len(arm_positions) == 14:
                self.control_command['arm_positions'] = [float(p) for p in arm_positions]
            else:
                print(f'双臂位置应为14个值，收到{len(arm_positions)}个')
                
        if left_hand_positions is not None:
            if len(left_hand_positions) == 6:
                self.control_command['left_hand_positions'] = [float(p) for p in left_hand_positions]
            else:
                print(f'左手位置应为6个值，收到{len(left_hand_positions)}个')
                
        if right_hand_positions is not None:
            if len(right_hand_positions) == 6:
                self.control_command['right_hand_positions'] = [float(p) for p in right_hand_positions]
            else:
                print(f'右手位置应为6个值，收到{len(right_hand_positions)}个')

    def send_control_data(self):
        """发送控制数据到服务端"""
        while True:
            if not self.is_connected:
                time.sleep(0.1)
                continue
                
            # 检查是否有控制命令需要发送
            if (self.control_command['arm_positions'] is not None or
                self.control_command['left_hand_positions'] is not None or
                self.control_command['right_hand_positions'] is not None):
                
                # 准备控制数据
                control_data = {}
                
                if self.control_command['arm_positions'] is not None:
                    control_data['arm_positions'] = self.control_command['arm_positions']
                    
                if self.control_command['left_hand_positions'] is not None:
                    control_data['left_hand_positions'] = self.control_command['left_hand_positions']
                    
                if self.control_command['right_hand_positions'] is not None:
                    control_data['right_hand_positions'] = self.control_command['right_hand_positions']
                
                # 发送控制数据
                try:
                    json_data = json.dumps(control_data)
                    self.socket_client.send((json_data + '\n').encode('utf-8'))
                    # print(f'已发送控制命令: {control_data}')
                    
                    # 清空控制命令
                    self.control_command = {
                        'arm_positions': None,
                        'left_hand_positions': None,
                        'right_hand_positions': None
                    }
                    
                except Exception as e:
                    print(f'发送控制数据失败: {e}')
                    self.is_connected = False
            
            time.sleep(0.1)  # 100ms发送间隔

    def get_robot_state(self):
        """获取机器人状态"""
        return self.robot_state

    def is_ready(self):
        """检查是否已连接并收到状态"""
        return (self.is_connected and 
                self.robot_state['left_arm'] is not None and 
                self.robot_state['right_arm'] is not None and
                time.time() - self.robot_state.get('timestamp', 0) < 5.0)  # 5秒内收到过状态

    def move_to_position(self, arm_positions=None, left_hand_positions=None, right_hand_positions=None):
        """移动到指定位置（简化接口）"""
        self.set_control_command(arm_positions, left_hand_positions, right_hand_positions)

    def shutdown(self):
        """关闭客户端"""
        try:
            if self.socket_client:
                self.socket_client.close()
        except Exception:
            pass

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
import time
import threading

class X100_ROS2(Node):
    def __init__(self, left_side_only=False, publish_debug=True):
        super().__init__("cr100_controller")
        
        # 存储左右臂的状态
        self.left_side_only = left_side_only
        self.publish_debug = publish_debug
        self.left_arm_state = None
        self.right_arm_state = None
        self.last_msg_time = None
        self.hand_control = True
        self._publish_debug_counts = {}
        self._publish_debug_last_log_t = {}
        self._right_arm_publish_disabled_logged = False
        self._right_hand_publish_disabled_logged = False
        self.left_arm_names = None
        self.left_arm_positions = None
        self.left_arm_velocities = None
        self.right_arm_names = None
        self.right_arm_positions = None
        self.right_arm_velocities = None
        self.left_state_seq = 0
        self.left_state_recv_wall_t = None
        self.left_state_header_stamp = None
        self.last_left_arm_cmd_wall_t = None
        self.last_left_arm_cmd_qpos = None
        self.left_arm_cmd_seq = 0
        self._left_arm_cmd_pub_times = []
        
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
        state_qos = QoSProfile(depth=10)
        state_qos.durability = DurabilityPolicy.VOLATILE
        state_qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # 创建两个状态订阅（左右臂）
        self.left_arm_sub = self.create_subscription(
            JointState, "/cr100/left_arm_state", self._left_arm_callback, state_qos)
        self.right_arm_sub = self.create_subscription(
            JointState, "/cr100/right_arm_state", self._right_arm_callback, state_qos)
        
        # 创建控制发布器（使用JointState格式）
        self.left_arm_pub = self.create_publisher(
            JointState, "/cr100/left_arm/online_joint_command", 1)
        self.left_hand_pub = self.create_publisher(
            JointState, "/cr100/left_dexterous_hand_command", 1)
        self.right_arm_pub = None
        self.dual_arm_pub = None
        self.right_hand_pub = None
        if not self.left_side_only:
            self.right_arm_pub = self.create_publisher(
                JointState, "/cr100/right_arm/online_joint_command", 1)
            self.dual_arm_pub = self.create_publisher(
                JointState, "/cr100/dual_arm/online_joint_command", 1)
            self.right_hand_pub = self.create_publisher(
                JointState, "/cr100/right_dexterous_hand_command", 1)
        
        # 启动spin线程
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        self._spin_thread.start()

    def _left_arm_callback(self, msg: JointState):
        now_t = time.time()
        self.last_msg_time = now_t
        self.left_state_recv_wall_t = now_t
        self.left_arm_state = msg
        self.left_state_header_stamp = msg.header.stamp
        self.left_state_seq += 1
        # 提取关节名称和位置
        self.left_arm_names = list(msg.name)
        self.left_arm_positions = list(msg.position)
        self.left_arm_velocities = list(msg.velocity)

    def _right_arm_callback(self, msg: JointState):
        self.last_msg_time = time.time()
        self.right_arm_state = msg
        # 提取关节名称和位置
        self.right_arm_names = list(msg.name)
        self.right_arm_positions = list(msg.position)
        self.right_arm_velocities = list(msg.velocity)

    def get_left_arm_state_status(self, max_left_state_age=None):
        now_t = time.time()
        positions = self.left_arm_positions
        velocities = self.left_arm_velocities
        names = self.left_arm_names
        state_age = None
        if self.left_state_recv_wall_t is not None:
            state_age = now_t - self.left_state_recv_wall_t

        position_len = len(positions) if positions is not None else 0
        velocity_len = len(velocities) if velocities is not None else 0
        name_len = len(names) if names is not None else 0
        state_received = self.left_arm_state is not None
        invalid_reason = None

        if not state_received:
            invalid_reason = "state_not_received"
        elif position_len < 7:
            invalid_reason = "position_len_lt_7"
        elif (
            max_left_state_age is not None
            and state_age is not None
            and state_age > max_left_state_age
        ):
            invalid_reason = "state_stale"

        state_head = None
        if positions is not None:
            state_head = [float(p) for p in positions[: min(3, position_len)]]

        return {
            "state_received": state_received,
            "state_valid": invalid_reason is None,
            "invalid_reason": invalid_reason,
            "seq": self.left_state_seq,
            "position_len": position_len,
            "velocity_len": velocity_len,
            "name_len": name_len,
            "state_age": state_age,
            "state_head": state_head,
        }

    def get_arm_state(self, require_fresh_left=False, max_left_state_age=None):
        """
        返回左右臂的状态，格式为 (positions, velocities)
        每个臂取前7个关节
        """
        n_expected = 14  # 左右臂各7个关节

        left_status = self.get_left_arm_state_status(max_left_state_age)
        if require_fresh_left and not left_status["state_valid"]:
            return None, None
        
        # 初始化返回数组
        pos = np.zeros(n_expected, dtype=float)
        vel = np.zeros(n_expected, dtype=float)
        
        # 处理左臂状态
        try:
            left_positions = getattr(self, "left_arm_positions", None) or []
            left_velocities = getattr(self, "left_arm_velocities", None) or []
            
            for i in range(min(7, len(left_positions))):
                pos[i] = float(left_positions[i])
            for i in range(min(7, len(left_velocities))):
                vel[i] = float(left_velocities[i])
        except Exception as e:
            self.get_logger().warn(f"获取左臂状态失败: {e}")
        
        # 处理右臂状态
        try:
            right_positions = getattr(self, "right_arm_positions", None) or []
            right_velocities = getattr(self, "right_arm_velocities", None) or []
            
            for i in range(min(7, len(right_positions))):
                pos[i+7] = float(right_positions[i])
            for i in range(min(7, len(right_velocities))):
                vel[i+7] = float(right_velocities[i])
        except Exception as e:
            self.get_logger().warn(f"获取右臂状态失败: {e}")
        
        return pos, vel

    def _create_joint_state_msg(self, positions, velocity=None, joint_names=None):
        """
        创建JointState消息
        注意：velocity可以是一个值，所有关节共享这个速度值
        """
        msg = JointState()
        
        # 设置时间戳
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # 设置关节名称
        if joint_names is not None:
            msg.name = joint_names
        else:
            # 生成默认关节名称
            msg.name = [f"joint_{i}" for i in range(len(positions))]
        
        # 设置位置
        msg.position = [float(p) for p in positions]
        
        # 设置速度 - 所有关节使用同一个速度值
        if velocity is not None:
            # 如果velocity是单个值，创建一个包含这个值的列表
            if isinstance(velocity, (int, float)):
                msg.velocity = [float(velocity)]
            else:
                msg.velocity = [float(v) for v in velocity]
        else:
            # 默认速度
            msg.velocity = [0.5]  # 所有关节使用默认速度0.5
        
        return msg

    def _log_publish_debug(self, key, topic_name, msg):
        if not self.publish_debug:
            return
        count = self._publish_debug_counts.get(key, 0) + 1
        self._publish_debug_counts[key] = count
        now_t = time.time()
        last_t = self._publish_debug_last_log_t.get(key, 0.0)
        if count <= 5 or now_t - last_t >= 1.0:
            self.get_logger().info(
                "[PublishDebug] "
                f"{key} publish_count={count} "
                f"topic={topic_name} "
                f"name_len={len(msg.name)} "
                f"position_len={len(msg.position)} "
                f"velocity_len={len(msg.velocity)} "
                f"position_head={list(msg.position[:3])}"
            )
            self._publish_debug_last_log_t[key] = now_t

    def _record_left_arm_command(self, qpos):
        now_t = time.time()
        self.left_arm_cmd_seq += 1
        self.last_left_arm_cmd_wall_t = now_t
        self.last_left_arm_cmd_qpos = np.asarray(qpos, dtype=np.float64).reshape(-1).copy()
        self._left_arm_cmd_pub_times.append(now_t)
        if len(self._left_arm_cmd_pub_times) > 200:
            self._left_arm_cmd_pub_times = self._left_arm_cmd_pub_times[-200:]

    def get_left_arm_debug_snapshot(self):
        now_t = time.time()
        cmd_q = None
        state_q = None
        state_vel = None

        if self.last_left_arm_cmd_qpos is not None:
            cmd_q = self.last_left_arm_cmd_qpos.copy()
        state_status = self.get_left_arm_state_status()
        left_positions = getattr(self, "left_arm_positions", None)
        if state_status["state_valid"] and left_positions is not None:
            state_q = np.zeros(7, dtype=np.float64)
            for i in range(min(7, len(left_positions))):
                state_q[i] = float(left_positions[i])
        left_velocities = getattr(self, "left_arm_velocities", None)
        if state_status["state_valid"] and left_velocities is not None:
            state_vel = np.zeros(7, dtype=np.float64)
            for i in range(min(7, len(left_velocities))):
                state_vel[i] = float(left_velocities[i])

        cmd_age = None
        if self.last_left_arm_cmd_wall_t is not None:
            cmd_age = now_t - self.last_left_arm_cmd_wall_t
        state_age = None
        if self.left_state_recv_wall_t is not None:
            state_age = now_t - self.left_state_recv_wall_t

        recent_cmd_times = [
            t for t in self._left_arm_cmd_pub_times if now_t - t <= 1.0
        ]
        cmd_hz = float(len(recent_cmd_times))

        err = None
        err_max = None
        err_mean = None
        if cmd_q is not None and state_q is not None:
            n = min(cmd_q.shape[0], state_q.shape[0], 7)
            err = np.abs(cmd_q[:n] - state_q[:n])
            err_max = float(np.max(err)) if n else None
            err_mean = float(np.mean(err)) if n else None

        state_vel_max = None
        moving_joints = 0
        if state_vel is not None:
            state_vel_max = float(np.max(np.abs(state_vel)))
            moving_joints = int(np.count_nonzero(np.abs(state_vel) > 0.01))

        return {
            "seq": self.left_arm_cmd_seq,
            "cmd_hz": cmd_hz,
            "cmd_age": cmd_age,
            "state_age": state_age,
            "cmd_q": cmd_q,
            "state_q": state_q,
            "state_vel": state_vel,
            "cmd_to_state_err": err,
            "cmd_to_state_err_max": err_max,
            "cmd_to_state_err_mean": err_mean,
            "state_vel_max": state_vel_max,
            "moving_joints": moving_joints,
            "state_status": state_status,
        }

    def publish_joint_command(self, left_arm_qpos=None, right_arm_qpos=None, 
                             left_hand_qpos=None, right_hand_qpos=None,
                             left_arm_vel=1.0, right_arm_vel=0.5,
                             left_hand_vel=1.0, right_hand_vel=0.5):
        """
        发布关节控制命令到各个话题（使用JointState格式）
        注意：速度参数现在可以是单个值，所有关节共享这个速度
        """
        # 发布左臂命令
        if left_arm_qpos is not None:
            msg = self._create_joint_state_msg(
                left_arm_qpos, 
                left_arm_vel,
                getattr(self, 'left_arm_names', None)
            )
            self.left_arm_pub.publish(msg)
            self._record_left_arm_command(left_arm_qpos)
            self._log_publish_debug(
                "left_arm",
                "/cr100/left_arm/online_joint_command",
                msg,
            )

        # # 发布右臂命令
        # if right_arm_qpos is not None:
        #     msg = self._create_joint_state_msg(
        #         right_arm_qpos, 
        #         right_arm_vel,
        #         getattr(self, 'right_arm_names', None)
        #     )
        #     self.right_arm_pub.publish(msg)
        if (
            self.publish_debug
            and right_arm_qpos is not None
            and not self._right_arm_publish_disabled_logged
        ):
            self.get_logger().info(
                "[PublishDebug] right_arm_qpos received, but "
                "/cr100/right_arm/online_joint_command publishing is disabled in code"
            )
            self._right_arm_publish_disabled_logged = True

        # 发布双臂命令
        # if left_arm_qpos is not None and right_arm_qpos is not None:
        #     # 合并左右臂命令
        #     dual_arm_positions = list(left_arm_qpos) + list(right_arm_qpos)
            
        #     # 使用左臂速度作为双臂速度（或取平均值）
        #     dual_arm_velocity = left_arm_vel
            
        #     # 合并关节名称
        #     dual_arm_names = []
        #     if hasattr(self, 'left_arm_names') and hasattr(self, 'right_arm_names'):
        #         dual_arm_names = self.left_arm_names + self.right_arm_names
            
        #     msg = self._create_joint_state_msg(
        #         dual_arm_positions, 
        #         dual_arm_velocity,
        #         dual_arm_names
        #     )
        #     self.dual_arm_pub.publish(msg)
        if self.hand_control is True:
            # 发布左手命令
            if left_hand_qpos is not None:
                # 为手部生成默认名称
                hand_names = [f"left_hand_joint_{i}" for i in range(len(left_hand_qpos))]
                msg = self._create_joint_state_msg(
                    left_hand_qpos, 
                    left_hand_vel,
                    hand_names
                )
                self.left_hand_pub.publish(msg)
                self._log_publish_debug(
                    "left_hand",
                    "/cr100/left_dexterous_hand_command",
                    msg,
                )

            # 发布右手命令
            if right_hand_qpos is not None and self.right_hand_pub is not None:
                # 为手部生成默认名称
                hand_names = [f"right_hand_joint_{i}" for i in range(len(right_hand_qpos))]
                msg = self._create_joint_state_msg(
                    right_hand_qpos, 
                    right_hand_vel,
                    hand_names
                )
                self.right_hand_pub.publish(msg)
                self._log_publish_debug(
                    "right_hand",
                    "/cr100/right_dexterous_hand_command",
                    msg,
                )
            elif (
                right_hand_qpos is not None
                and self.left_side_only
                and not self._right_hand_publish_disabled_logged
                and self.publish_debug
            ):
                self.get_logger().info(
                    "[PublishDebug] right_hand_qpos received, but "
                    "right hand publishing is disabled in left_side_only mode"
                )
                self._right_hand_publish_disabled_logged = True
            self.hand_control = False
        else:
            if self.publish_debug:
                skip_count = self._publish_debug_counts.get("hand_skip", 0) + 1
                self._publish_debug_counts["hand_skip"] = skip_count
                now_t = time.time()
                last_t = self._publish_debug_last_log_t.get("hand_skip", 0.0)
                if skip_count <= 5 or now_t - last_t >= 1.0:
                    self.get_logger().info(
                        "[PublishDebug] hand publish skipped this call "
                        f"skip_count={skip_count}; will publish hands on next call"
                    )
                    self._publish_debug_last_log_t["hand_skip"] = now_t
            self.hand_control = True

    def publish_left_arm_hand_command(
        self,
        left_arm_qpos=None,
        left_hand_qpos=None,
        left_arm_vel=1.0,
        left_hand_vel=1.0,
    ):
        """Publish only the left arm and left dexterous hand command topics."""
        if left_arm_qpos is not None:
            msg = self._create_joint_state_msg(
                left_arm_qpos,
                left_arm_vel,
                getattr(self, 'left_arm_names', None)
            )
            self.left_arm_pub.publish(msg)
            self._record_left_arm_command(left_arm_qpos)

        if left_hand_qpos is None:
            return

        if self.hand_control is True:
            hand_names = [f"left_hand_joint_{i}" for i in range(len(left_hand_qpos))]
            msg = self._create_joint_state_msg(
                left_hand_qpos,
                left_hand_vel,
                hand_names,
            )
            self.left_hand_pub.publish(msg)
            self.hand_control = False
        else:
            self.hand_control = True

    def publish_all_joints(self, all_qpos, velocity=0.5):
        """
        简化接口：一次性发布所有关节命令
        假设all_qpos包含所有关节：左臂7个 + 右臂7个 + 左手n个 + 右手n个
        所有关节使用同一个速度值
        """
        # 假设关节顺序：左臂(7) -> 右臂(7) -> 左手(6) -> 右手(6)
        left_arm_end = 7
        right_arm_end = 14
        left_hand_end = 20  # 假设左右手各6个关节
        
        left_arm_qpos = all_qpos[:left_arm_end]
        right_arm_qpos = all_qpos[left_arm_end:right_arm_end]
        left_hand_qpos = all_qpos[right_arm_end:left_hand_end]
        right_hand_qpos = all_qpos[left_hand_end:]
            
        self.publish_joint_command(
            left_arm_qpos, right_arm_qpos, left_hand_qpos, right_hand_qpos,
            velocity, velocity, velocity, velocity
        )

    def is_ready(self):
        """检查是否已收到状态消息"""
        return (self.left_arm_state is not None and 
                self.right_arm_state is not None and
                time.time() - self.last_msg_time < 2.0)  # 2秒内收到过消息

    def shutdown(self):
        """关闭节点"""
        try:
            self.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
        if hasattr(self, '_spin_thread') and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1)


if __name__ == "__main__":
    simulator = X100_Sim()
    with open("x100_say_hello.json", "r", encoding="utf-8") as f:
        yoga_dict = json.load(f)
    opt_mimic_pos = np.zeros((400, 39), dtype=np.double)
    for i in range(400):
        opt_mimic_pos[i, :] = np.array(yoga_dict[str(i)][:], dtype=np.double)
    qpos = np.zeros(38)  # 14+2+11*2
    index = 0
    try:
        while True:
            qpos[2:20] = opt_mimic_pos[index, 21:39]
            qpos[20:38] = opt_mimic_pos[index, 3:21]
            left_img, right_img = simulator.step(qpos)
            simulator.get_arm_state()
            time.sleep(0.02)
            index += 1
            if index >= 400:
                index -= 400
    except KeyboardInterrupt:
        simulator.end()
        exit(0)
