#!/usr/bin/env python3
import os
import json
import time
import math
import argparse
from typing import List, Tuple

import numpy as np

try:
    # SciPy 可选，如果没有则回退为简单均值滤波
    from scipy.signal import butter, filtfilt  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from PallasSDK import Humanoid
from PallasSDK import IPCCommandPacket
from PallasSDK import CtrlWordBuilder
from PallasSDK import ArmDataType
from PallasSDK import HandDataType

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory


LEFT_JOINT_NAMES: List[str] = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
]

RIGHT_JOINT_NAMES: List[str] = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]


def generate_simple_trajectories(seconds: float, num_points: int, angle_deg: float = 45.0, joint_index: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """生成参数化简单双臂轨迹：第 joint_index(0-based) 关节从 0→angle→0。

    seconds: 总时长(秒)
    num_points: 采样点数
    angle_deg: 峰值角度（度）
    joint_index: 目标关节索引（0..6），默认第6个关节→索引5
    返回: (left[N,7], right[N,7])，单位为弧度
    """
    num_points = max(2, int(num_points))
    seconds = max(0.02, float(seconds))
    mid = (num_points - 1) // 2
    peak = math.radians(angle_deg)

    prof = np.zeros((num_points,), dtype=float)
    # 上半段（含峰值点）
    for i in range(mid + 1):
        prof[i] = peak * (i / max(1, mid))
    # 下半段
    for i in range(mid + 1, num_points):
        ratio = (num_points - 1 - i) / max(1, (num_points - 1 - mid))
        prof[i] = peak * max(0.0, ratio)

    left = np.zeros((num_points, 7), dtype=float)
    right = np.zeros((num_points, 7), dtype=float)
    left[:, joint_index] = prof
    right[:, joint_index] = prof
    return left, right


def _append_zero_velocity_tail(points: np.ndarray, dt: float = 0.02, steps: int = 10) -> np.ndarray:
    """在轨迹末端追加一段匀减速至零速度的点，模仿 SDKRobot.py 的处理思路。

    points: 形状 (N, 7)
    返回: 形状 (N+steps+1, 7)
    """
    if points.shape[0] < 2:
        return points.copy()

    # 末端速度（使用最后两个点的差分近似）
    v_last = (points[-1] - points[-2]) / dt
    p_start = points[-1]

    out = [p for p in points]
    for i in range(1, steps + 1):
        t = i * dt
        p_t = p_start + v_last * t + 0.5 * (-v_last / (steps * dt)) * (t ** 2)
        out.append(p_t)
    # 再补一个最后点
    out.append(out[-1])
    return np.array(out, dtype=float)


def _lowpass_filter(series: np.ndarray, fs: float = 50.0, cutoff: float = 10.0, order: int = 2) -> np.ndarray:
    """对单通道序列做低通滤波。无 SciPy 时使用简单均值滤波退化实现。"""
    if series.size <= 3:
        return series.copy()

    if _HAS_SCIPY:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, series)
    # 简单移动均值滤波作为回退
    window = 5
    kernel = np.ones(window) / window
    padded = np.pad(series, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def load_and_filter_trajectories(json_path: str, fs: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """读取 x100_say_hello.json，抽取左右臂 7 关节轨迹，末端减速并低通滤波。

    返回: (left[N,7], right[N,7])
    """
    with open(json_path, "r", encoding="utf-8") as f:
        yoga_dict = json.load(f)

    # 依据 SDKRobot.py：共有 400 帧，每帧 39 维
    num_frames = len(yoga_dict)
    dim = len(next(iter(yoga_dict.values())))
    opt_mimic_pos = np.zeros((num_frames, dim), dtype=float)
    # 键为字符串索引 "0".."N"
    for i in range(num_frames):
        opt_mimic_pos[i, :] = np.array(yoga_dict[str(i)], dtype=float)

    # 左臂列 3..9，共7维；右臂列 21..27，共7维
    left = opt_mimic_pos[:, 3:10]
    right = opt_mimic_pos[:, 21:28]

    # 末端速度处理
    dt = 1.0 / fs
    left = _append_zero_velocity_tail(left, dt=dt, steps=10)
    right = _append_zero_velocity_tail(right, dt=dt, steps=10)

    # 逐关节低通滤波
    left_f = np.zeros_like(left)
    right_f = np.zeros_like(right)
    for j in range(7):
        left_f[:, j] = _lowpass_filter(left[:, j], fs=fs)
        right_f[:, j] = _lowpass_filter(right[:, j], fs=fs)
        # # tg DEBUG
        # left_f[:,:4] = 0
        # right_f[:,:4] = 0

    return left_f, right_f


def RightArmLD2QKM(joint_position):
	if len(joint_position) != 7:
		print("joint position dof is not 7!")
	
	joint_direction = [1, 1, 1, -1, 1, -1, -1]
	return  (np.array(joint_position) * np.array(joint_direction)).tolist()

def LeftArmLD2QKM(joint_position):
	if len(joint_position) != 7:
		print("joint position dof is not 7!")
	
	joint_direction = [1, -1, -1, -1, -1, -1, -1]
	return  (np.array(joint_position) * np.array(joint_direction)).tolist()


left_arm_joint_position_list = []
right_arm_joint_position_list = []
a = None
def ipc_callback(data):
	global left_arm_joint_position_list
	global right_arm_joint_position_list
	global a
	a = data

	# 得到头和腰部的关节位置
	head_joint_position = [data.head_waist_pos[0], data.head_waist_pos[1]]

	# 得到左手臂的关节位置
	left_arm_joint_position = [data.arm1_joint_pos[0] ,
			  data.arm1_joint_pos[1] ,
			  data.arm1_joint_pos[2] ,
			  data.arm1_joint_pos[3] ,
			  data.arm1_joint_pos[4] ,
			  data.arm1_joint_pos[5] ,
			  data.arm1_joint_pos[6]]
	# 得到右手臂的关节位置
	right_arm_joint_position =  [data.arm2_joint_pos[0] ,
			  data.arm2_joint_pos[1] ,
			  data.arm2_joint_pos[2] ,
			  data.arm2_joint_pos[3] ,
			  data.arm2_joint_pos[4] ,
			  data.arm2_joint_pos[5] ,
			  data.arm2_joint_pos[6]]

	# 得到左灵巧手的关节位置
	left_hand_joint_position = [data.hand1_joint_pos[0] ,
			  data.hand1_joint_pos[1] ,
			  data.hand1_joint_pos[2] ,
			  data.hand1_joint_pos[3] ,
			  data.hand1_joint_pos[4] ,
			  data.hand1_joint_pos[5] ]
	# 得到右灵巧手的关节位置
	right_hand_joint_position =  [data.hand2_joint_pos[0] ,
			  data.hand2_joint_pos[1] ,
			  data.hand2_joint_pos[2] ,
			  data.hand2_joint_pos[3] ,
			  data.hand2_joint_pos[4] ,
			  data.hand2_joint_pos[5]]

	# 得到左手臂的错误代码
	arm1_error_code = data.arm1_errorcode

	# 得到右手臂的错误代码
	arm2_error_code = data.arm2_errorcode

	# 得到左手臂的运行状态[当前没有更新这个参数]
	arm1_status = data.arm1_status

	# 得到右手臂的运行状态[当前没有更新这个参数]
	arm2_status = data.arm2_status

	# 得到左手臂的设置的工具坐标系
	arm1_tool = data.arm1_tool

	# 得到右手臂设置的工具坐标系
	arm2_tool = data.arm2_tool

	# 得到左手臂的笛卡尔位姿[X, Y, Z, R, P, Y] ,姿态的表示形式为  ZYZ 欧拉角的形式
	arm1_cart_pos = data.arm1_cart_pos

	# 得到右手臂的笛卡尔位姿[X, Y, Z, R, P, Y] ,姿态的表示形式为  ZYZ 欧拉角的形式
	arm2_cart_pos = data.arm2_cart_pos

	# 得到左手臂的笛卡尔速度[X, Y, Z, Rx, Ry, Rz]
	arm1_cart_speed = data.arm1_cart_speed

	# 得到右手臂的笛卡尔速度[X, Y, Z, Rx, Ry, Rz]
	arm2_cart_speed = data.arm2_cart_speed

	# 得到左手臂的关节速度
	arm1_joint_speed = data.arm1_joint_speed

	# 得到右手臂的关节速度
	arm2_joint_speed = data.arm2_joint_speed

	# 得到左灵巧手的关节力矩[当前没有更新这个参数]
	hand1_joint_force = data.hand1_joint_force

	# 得到左灵巧手的关节电流[当前没有更新这个参数]
	hand1_joint_current = data.hand1_joint_current

	# 得到右灵巧手的关节力矩[当前没有更新这个参数]
	hand2_joint_force = data.hand2_joint_force

	# 得到右灵巧手的关节电流[当前没有更新这个参数]
	hand2_joint_current = data.hand2_joint_current

	
	left_arm_joint_position = LeftArmLD2QKM(left_arm_joint_position)
	right_arm_joint_position = RightArmLD2QKM(right_arm_joint_position)


	left_arm_joint_position_list.append(left_arm_joint_position)
	right_arm_joint_position_list.append(left_arm_joint_position)




class X100ArmsTester(Node):
    def __init__(self, json_path: str, rate_hz: float = 50.0, max_points: int = 0):
        super().__init__("x100_test_node_python")

        self.rate_hz = rate_hz
        self.dt = 1.0 / max(1.0, rate_hz)
        self.max_points = max_points

        # ROS2 interfaces
        self.power_cli = self.create_client(SetBool, "/x100/arms_power_on_cmd")
        self.left_action = ActionClient(self, FollowJointTrajectory, "/x100/left_arm/follow_joint_trajectory")
        self.right_action = ActionClient(self, FollowJointTrajectory, "/x100/right_arm/follow_joint_trajectory")

        self.left_pub = self.create_publisher(JointTrajectoryPoint, "/x100/left_arm/joint_cmd", 10)
        self.right_pub = self.create_publisher(JointTrajectoryPoint, "/x100/right_arm/joint_cmd", 10)
        self.dual_pub = self.create_publisher(JointTrajectoryPoint, "/x100/dual_arms/joint_cmd", 10)
        self.set_online_cli = self.create_client(SetBool, "/x100/set_online_mode")

        # 载入与预处理轨迹
        self.get_logger().info(f"Loading trajectories from: {json_path}")
        left, right = load_and_filter_trajectories(json_path, fs=self.rate_hz)
        # 裁剪长度
        if self.max_points and self.max_points > 0:
            left = left[: self.max_points]
            right = right[: self.max_points]
        self.left_traj = left
        self.right_traj = right
        with open('traj.json', 'w', encoding='utf-8') as f:
            if hasattr(self.left_traj, 'tolist'):
                left_data = self.left_traj.tolist()
            else:
                left_data = self.left_traj

            if hasattr(self.right_traj, 'tolist'):
                right_data = self.right_traj.tolist()
            else:
                right_data = self.right_traj

            data_to_save = {
                "left_trajectory": left_data,
                "right_trajectory": right_data,
                "metadata": {
                    "frames": len(left_data),
                    "points_per_frame": len(left_data[0]) if left_data else 0
                }
            }

            json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    def call_power(self, on: bool, timeout_sec: float = 5.0) -> bool:
        if not self.power_cli.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error("Service /x100/arms_power_on_cmd not available")
            return False
        req = SetBool.Request()
        req.data = bool(on)
        future = self.power_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error("Service call failed")
            return False
        res = future.result()
        self.get_logger().info(f"Power {'ON' if on else 'OFF'} result: success={res.success}, message={res.message}")
        return bool(res.success)

    def _build_joint_trajectory(self, joint_names: List[str], points: np.ndarray) -> JointTrajectory:
        traj = JointTrajectory()
        traj.joint_names = list(joint_names)
        t = 0.0
        for i in range(points.shape[0]):
            t += self.dt
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in points[i].tolist()]
            # 可选：给零速度，部分控制器更稳
            pt.velocities = [0.0] * len(pt.positions)
            d = Duration(seconds=t)
            pt.time_from_start = d.to_msg()
            traj.points.append(pt)
        return traj

    def run_test2_left_action(self) -> bool:
        self.get_logger().info("[test2] left arm FollowJointTrajectory")
        if not self.left_action.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Left action server not available")
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = self._build_joint_trajectory(LEFT_JOINT_NAMES, self.left_traj)
        print("self.left_traj: ",self.left_traj)
        send_future = self.left_action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Left action goal rejected")
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        self.get_logger().info(f"Left action finished with error_code={result.error_code}\n{result.error_string}")
        return True

    def run_test3_right_action(self) -> bool:
        self.get_logger().info("[test3] right arm FollowJointTrajectory")
        if not self.right_action.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Right action server not available")
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = self._build_joint_trajectory(RIGHT_JOINT_NAMES, self.right_traj)
        send_future = self.right_action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Right action goal rejected")
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        self.get_logger().info(f"Right action finished with error_code={result.error_code}\n{result.error_string}")
        return True

    def run_test4_left_online(self):
        self.get_logger().info("[test4] left arm online topic streaming at %.1f Hz" % self.rate_hz)
        # 切换 Online 模式
        if not self.set_online_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Service /x100/set_online_mode not available")
            return
        req = SetBool.Request(); req.data = True
        future = self.set_online_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if res is None or not res.success:
            self.get_logger().error("Failed to enable online mode")
            return
        for i in range(self.left_traj.shape[0]):
            msg = JointTrajectoryPoint()
            msg.positions = [float(v) for v in self.left_traj[i].tolist()]
            self.left_pub.publish(msg)
            time.sleep(self.dt)

    def run_test5_right_online(self):
        self.get_logger().info("[test5] right arm online topic streaming at %.1f Hz" % self.rate_hz)
        if not self.set_online_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Service /x100/set_online_mode not available")
            return
        req = SetBool.Request(); req.data = True
        future = self.set_online_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if res is None or not res.success:
            self.get_logger().error("Failed to enable online mode")
            return
        for i in range(self.right_traj.shape[0]):
            msg = JointTrajectoryPoint()
            msg.positions = [float(v) for v in self.right_traj[i].tolist()]
            self.right_pub.publish(msg)
            time.sleep(self.dt)

    def run_test6_dual_online(self):
        self.get_logger().info("[test6] dual arms online topic streaming at %.1f Hz" % self.rate_hz)
        if not self.set_online_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Service /x100/set_online_mode not available")
            return
        req = SetBool.Request(); req.data = True
        future = self.set_online_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if res is None or not res.success:
            self.get_logger().error("Failed to enable online mode")
            return
        n = min(self.left_traj.shape[0], self.right_traj.shape[0])
        for i in range(n):
            msg = JointTrajectoryPoint()
            msg.positions = [float(v) for v in self.left_traj[i].tolist()] + [
                float(v) for v in self.right_traj[i].tolist()
            ]
            # 需严格 14 维
            if len(msg.positions) != 14:
                self.get_logger().warn("dual arms point size != 14, skip")
                continue
            self.dual_pub.publish(msg)
            time.sleep(self.dt)

    def run_test7_simple_action(self, seconds: float, num_points: int, joint_index: int = 5, angle_deg: float = 45.0):
        self.get_logger().info(f"[test7] simple action both arms: J{joint_index+1} 0→{angle_deg}°→0, {seconds}s/{num_points}pts")
        left, right = generate_simple_trajectories(seconds, num_points, angle_deg=angle_deg, joint_index=joint_index)
        # 设置内部 dt 以匹配给定持续时间
        self.dt = float(seconds) / max(1, int(num_points))

        # Power ON
        self.call_power(True)

        # Left action
        if not self.left_action.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Left action server not available")
            return
        goal_l = FollowJointTrajectory.Goal()
        goal_l.trajectory = self._build_joint_trajectory(LEFT_JOINT_NAMES, left)
        fh_l = self.left_action.send_goal_async(goal_l)
        rclpy.spin_until_future_complete(self, fh_l)
        gh_l = fh_l.result()
        if gh_l.accepted:
            rf_l = gh_l.get_result_async()
            rclpy.spin_until_future_complete(self, rf_l)
            self.get_logger().info(f"Left result code={rf_l.result().result.error_code}")
        else:
            self.get_logger().error("Left simple action rejected")

        # Right action
        if not self.right_action.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Right action server not available")
            return
        goal_r = FollowJointTrajectory.Goal()
        goal_r.trajectory = self._build_joint_trajectory(RIGHT_JOINT_NAMES, right)
        fh_r = self.right_action.send_goal_async(goal_r)
        rclpy.spin_until_future_complete(self, fh_r)
        gh_r = fh_r.result()
        if gh_r.accepted:
            rf_r = gh_r.get_result_async()
            rclpy.spin_until_future_complete(self, rf_r)
            self.get_logger().info(f"Right result code={rf_r.result().result.error_code}")
        else:
            self.get_logger().error("Right simple action rejected")

        # Power OFF
        self.call_power(False)

    def run_test8_simple_dual_online(self, seconds: float, num_points: int, joint_index: int = 5, angle_deg: float = 45.0):
        self.get_logger().info(f"[test8] simple dual online: J{joint_index+1} 0→{angle_deg}°→0, {seconds}s/{num_points}pts")
        left, right = generate_simple_trajectories(seconds, num_points, angle_deg=angle_deg, joint_index=joint_index)
        # 设置内部 dt 以匹配给定持续时间
        self.dt = float(seconds) / max(1, int(num_points))

        self.call_power(True)
        req = SetBool.Request(); req.data = True
        future = self.set_online_cli.call_async(req)

        n = min(left.shape[0], right.shape[0])
        for i in range(n):
            msg = JointTrajectoryPoint()
            msg.positions = [float(v) for v in left[i].tolist()] + [float(v) for v in right[i].tolist()]
            if len(msg.positions) == 14:
                self.dual_pub.publish(msg)
            else:
                self.get_logger().warn("dual simple point size != 14, skip")
            time.sleep(self.dt)
            
    def run_test9_simple_right_online(self):
        self.get_logger().info("run_test9_simple_right_online start...")
        deg_mm = False
        human = Humanoid(isDegressMillimeter = deg_mm)
        print("SDK Version: " + human.GetSDKVersion())
        human.Connect("", "192.168.10.86", "192.168.10.88") 
        # human.Disconnect()
        # human.SetIPCFeedbackCallback(ipc_callback)
        human.SetPower(True, 1)
        human.SetSpeed(30, 1)
        command = IPCCommandPacket()
        ctrl_word = CtrlWordBuilder()
        # 表示是否要对头部进行控制
        ctrl_word.head_move(False)
        # 表示是否要对左手臂进行控制
        ctrl_word.left_arm_move(True)
        # 表示是否要对右手臂进行控制
        ctrl_word.right_arm_move(True)
        # 表示手臂需要发送的数据类型，当前表示为关节角度
        ctrl_word.arm_data_type(ArmDataType.ARM_JOINT_POSITION)
        # 表示是否要对左灵巧手进行控制
        ctrl_word.left_hand_move(True)
        # 表示是否要对右灵巧手进行控制
        ctrl_word.right_hand_move(True)
        # 表示灵巧手需要发送的数据类型，当前表示为关节角度
        ctrl_word.hand_data_type(HandDataType.HAND_JOINT_POSITION)
        command.ctrl_word = ctrl_word.build()

        # 设置头部和腰部控制指令
        command.head_waist_cmd = [0.0, 0.0, 0.0]
        # 设置左手臂的控制指令
        command.arm1_cmd  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # 设置右手臂的控制指令
        command.arm2_cmd  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # 设置左手臂的参数[控制周期（ms）、前瞻时间、增益、低通滤波器的信号频率(hz)、低通滤波器的截止频率(hz)]
        command.arm1_param = [0.02, 0.03, 900.0, 50, 10]
        # 设置右手臂的参数[控制周期(ms)、前瞻时间、增益、低通滤波器的信号频率(hz)、低通滤波器的截止频率(hz)]
        command.arm2_param = [0.02, 0.03, 900.0, 50, 10]

        # 设置左灵巧手的控制指令[大拇指的弯曲、食指、中指、无名指、小拇指、大拇指的侧摆]
        command.hand1_cmd = [0.3, 0.5, 0.5, 0.5, 0.5, 0.5]
        # 设置左灵巧上的速度控制指令[大拇指的弯曲、食指、中指、无名指、小拇指、大拇指的侧摆]
        command.hand1_speed_limit = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        # 设置右灵巧手的控制指令[大拇指的弯曲、食指、中指、无名指、小拇指、大拇指的侧摆]
        command.hand2_cmd = [0.3, 0.5, 0.5, 0.5, 0.5, 0.5]
        # 设置右灵巧上的速度控制指令[大拇指的弯曲、食指、中指、无名指、小拇指、大拇指的侧摆]
        command.hand2_speed_limit = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        # 将该指令下发给到三个控制器中。
        human.MoveArms(command)
        self.get_logger().info("run_test9_simple_right_online end...")


def main():
    parser = argparse.ArgumentParser(description="X100 Arms Hardware Node Tester (Python)")
    parser.add_argument(
        "--json",
        type=str,
        default="/home/nro/K100/robot_services/src/arm_controller/test/test_node_python/x100_say_hello.json",
        help="路径：x100_say_hello.json",
    )
    parser.add_argument("--rate", type=float, default=50.0, help="在线控制频率 Hz")
    parser.add_argument("--max-points", type=int, default=0, help="最大点数（0 表示全部）")
    parser.add_argument(
        "--run",
        type=str,
        default="all",
        choices=[
            "all",
            "svc",
            "left_action",
            "right_action",
            "left_online",
            "right_online",
            "dual_online",
            "test7",
            "test8",
        ],
        help="选择要运行的测试（含 test7/test8 简单双臂轨迹）",
    )
    parser.add_argument("--simple-seconds", type=float, default=4.0, help="test7/8: 简单轨迹总时长 s")
    parser.add_argument("--simple-points", type=int, default=10, help="test7/8: 简单轨迹点数")
    parser.add_argument("--simple-angle-deg", type=float, default=45.0, help="test7/8: 峰值角度（度）")
    parser.add_argument("--simple-joint-idx", type=int, default=5, help="test7/8: 目标关节索引(0..6)，默认第6关节")

    args = parser.parse_args()

    rclpy.init()
    node = X100ArmsTester(json_path=args.json, rate_hz=args.rate, max_points=args.max_points)

    try:
        # test1: service 上电
        if args.run in ("all", "svc"):
            node.get_logger().info("[test1] service power ON")
            node.call_power(False)
            node.call_power(True)
            # node.run_test9_simple_right_online()

        # test2: 左臂 action 轨迹
        if args.run in ("all", "left_action"):
            node.run_test2_left_action()

        # test3: 右臂 action 轨迹
        if args.run in ("all", "right_action"):
            node.run_test3_right_action()

        # test4: 左臂在线控制 topic
        if args.run in ("all", "left_online"):
            node.call_power(True)
            node.run_test4_left_online()
            # node.call_power(False)

        # test5: 右臂在线控制 topic
        if args.run in ("all", "right_online"):
            node.call_power(True)
            node.run_test5_right_online()
            

        # test6: 双臂在线控制 topic
        if args.run in ("all", "dual_online"):
            node.run_test6_dual_online()
            node.call_power(False)

        # test7: 上电 + 简单轨迹 action（左右臂）+ 下电
        if args.run in ("all", "test7"):
            node.run_test7_simple_action(
                seconds=args.simple_seconds,
                num_points=10,
                # num_points=args.simple_points,
                joint_index=args.simple_joint_idx,
                angle_deg=args.simple_angle_deg,
            )

        # test8: 上电 + 简单轨迹 双臂在线控制（topic）
        if args.run in ("all", "test8"):
            node.run_test8_simple_dual_online(
                seconds=args.simple_seconds,
                num_points=100,
                joint_index=args.simple_joint_idx,
                angle_deg=args.simple_angle_deg,
            )

        # test1: service 下电（在全部流程最后）
        if args.run == "all":
            node.get_logger().info("[test1] service power OFF")
            node.call_power(False)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

