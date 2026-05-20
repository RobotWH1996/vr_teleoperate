import numpy as np
#from isaacgym import gymapi, gymutil, gymtorch

# vision processing
from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessorLegacy
import cv2
import zmq
import pickle
import zlib
import time

# hand and arm control
from constants_vuer import tip_indices

import dex_retargeting

# 获取主模块位置
print(f"dex_retargeting 主模块位置: {dex_retargeting.__file__}")
from dex_retargeting.retargeting_config import RetargetingConfig
# print(f"retargeting_config 模块位置: {RetargetingConfig.__file__}")
# utils
from pathlib import Path
import yaml
from multiprocessing import Process, shared_memory, Queue, Manager, Event, Lock

import os
import sys
from pytransform3d import rotations

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.robot_control.robot_arm_ik import X100_29_ArmIK
from teleop.replay import X100_Sim, X100_SocketClient,X100_ROS2

from sensor_msgs.msg import Image

try:
    from cv_bridge import CvBridge

    _cv_bridge = CvBridge()
except ImportError:
    _cv_bridge = None
# ...existing code...


# # 原 ZMQ image_receiver 删除 / 替换为 ROS2 订阅版本
# def image_receiver(
#     sm, resolution, crop_size_w, crop_size_h, topic="/camera/color/image_raw"
# ):
#     import rclpy
#     from rclpy.node import Node
#     import numpy as np
#     import time

#     class _ImageSub(Node):
#         def __init__(self):
#             super().__init__("teleop_image_receiver")
#             self.sub = self.create_subscription(Image, topic, self.cb, 10)
#             self.last_log_t = time.time()

#         def cb(self, msg: Image):
#             try:
#                 if _cv_bridge:
#                     frame = _cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
#                 else:
#                     # 基础手动解码
#                     import numpy as np

#                     dt = np.uint8
#                     if msg.encoding in ("mono16", "16UC1"):
#                         dt = np.uint16
#                     buf = np.frombuffer(msg.data, dtype=dt)
#                     channels = (
#                         3
#                         if "rgb8" in msg.encoding.lower()
#                         or "bgr8" in msg.encoding.lower()
#                         else 1
#                     )
#                     frame = buf.reshape(msg.height, msg.step // channels, channels)
#                     if msg.encoding.lower() == "rgb8":
#                         frame = frame[:, :, ::-1]  # 转 BGR

#                 # 统一到共享内存尺寸 (h, 2*w)
#                 target_h = sm.img_height
#                 target_w = sm.img_width * 2
#                 if frame.shape[0] != target_h or frame.shape[1] != target_w:
#                     import cv2

#                     frame = cv2.resize(frame, (target_w, target_h))

#                 sm.write_image(frame)

#             except Exception:
#                 # 静默忽略单帧错误
#                 pass

#     rclpy.init(args=None)
#     node = _ImageSub()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()


def image_receiver(image_queue, resolution, crop_size_w, crop_size_h):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    # socket.connect("tcp://127.0.0.1:5555")
    # socket.connect("tcp://10.168.2.187:5555")
    # socket.connect("tcp://172.27.60.12:5559")
    socket.connect("tcp://192.168.1.105:5559")
    while True:
        compressed_data = b""
        while True:
            chunk = socket.recv()
            compressed_data += chunk
            if len(chunk) < 60000:
                break
        data = zlib.decompress(compressed_data)
        frame_data = pickle.loads(data)

        # Decode and display the image
        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print("frame", frame)
        sm.write_image(frame_rgb)
        # Control receiving frequency
        time.sleep(0.01)


# 共享内存管理图像数据
class SharedMemoryImage:
    def __init__(self, img_shape):
        self.resolution = img_shape  # (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (
            self.resolution[0] - self.crop_size_h,
            self.resolution[1] - 2 * self.crop_size_w,
        )

        # 创建共享内存
        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(
            create=True, size=np.prod(self.img_shape) * np.uint8().itemsize
        )
        self.img_array = np.ndarray(
            (self.img_shape[0], self.img_shape[1], 3),
            dtype=np.uint8,
            buffer=self.shm.buf,
        )
        self.lock = Lock()  # 创建锁对象

    # 写入图像数据
    def write_image(self, image):
        with self.lock:
            np.copyto(self.img_array, image)

    # 读取图像数据
    def read_image(self):
        with self.lock:
            image_copy = self.img_array.copy()
            return image_copy

    # 清理共享内存
    def cleanup(self):
        self.shm.close()
        self.shm.unlink()


# 遥操作类
class VuerTeleop:
    def __init__(self, config_file_path, left_side_only=False):
        self.left_side_only = left_side_only
        self.resolution = (480, 640)  # (720, 1280) 图像分辨率
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (
            self.resolution[0] - self.crop_size_h,
            self.resolution[1] - 2 * self.crop_size_w,
        )

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(
            create=True, size=np.prod(self.img_shape) * np.uint8().itemsize
        )
        self.img_array = np.ndarray(
            (self.img_shape[0], self.img_shape[1], 3),
            dtype=np.uint8,
            buffer=self.shm.buf,
        )
        image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(
            self.resolution_cropped, self.shm.name, image_queue, toggle_streaming
        )
        self.processor = VuerPreprocessorLegacy()

        # 重定向配置
        RetargetingConfig.set_default_urdf_dir("./assets")
        with Path(config_file_path).open("r") as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg["left"])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = None
        if not self.left_side_only:
            right_retargeting_config = RetargetingConfig.from_dict(cfg["right"])
            self.right_retargeting = right_retargeting_config.build()
        self.target_joint_indices = self.left_retargeting.optimizer.target_joint_indices

    def step_left(self, reset_hand_flag=False):
        head_mat, left_wrist_mat, _, left_hand_mat, _ = self.processor.process(
            self.tv, reset_hand_flag
        )

        head_rmat = head_mat[:3, :3]
        left_wrist_mat[2, 3] += 1.28
        left_wrist_mat[0, 3] -= 0.1
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[
            self.target_joint_indices
        ]

        return head_rmat, left_wrist_mat, left_qpos

    def step(self,reset_hand_flag=False):
        if self.left_side_only:
            head_rmat, left_wrist_mat, left_qpos = self.step_left(reset_hand_flag)
            return head_rmat, left_wrist_mat, None, left_qpos, None

        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = (
            self.processor.process(self.tv,reset_hand_flag)
        )
        # print("left_hand_mat",left_hand_mat)
        # print("right_hand_mat",right_hand_mat)
        # print("head_mat", head_mat)
        # print("left_wrist_mat", left_wrist_mat)
        # print("right_wrist_mat", right_wrist_mat)

        head_rmat = head_mat[:3, :3]
        # 转换头部坐标原点到机器人坐标原点 g1: head -> pelvis
        left_wrist_mat[2, 3] += 1.28
        right_wrist_mat[2, 3] += 1.28
        left_wrist_mat[0, 3] -= 0.1
        # left_wrist_mat[1, 3] += 0.05
        right_wrist_mat[0, 3] -= 0.1
        # print("left_wrist_mat", left_wrist_mat)
        # print("right_wrist_mat", right_wrist_mat)
        # print("tip_indices",tip_indices)
        # print("self.target_joint_indices",self.target_joint_indices)
        # print("left_hand_mat[tip_indices]",left_hand_mat[tip_indices])
        # left_hand_mat[tip_indices] = [[ 0,0,0.15],
        #                             [ 0,0,0.15],
        #                             [ 0,0,0.15],
        #                             [ 0,0,0.15],
        #                             [ 0,0,0.15]]
    # print("left_hand_mat[tip_indices] ",left_hand_mat[tip_indices] )
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[
            self.target_joint_indices
        ]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[
            self.target_joint_indices
        ]

        return head_rmat, left_wrist_mat, right_wrist_mat, left_qpos, right_qpos


import threading
import rclpy
from rclpy.executors import MultiThreadedExecutor


def safe_rclpy_init():
    try:
        rclpy.init()
    except RuntimeError as e:
        if "already initialized" not in str(e):
            raise


def start_ros_node():
    safe_rclpy_init()
    ros2_node = X100_ROS2()  # 只创建一次
    executor = MultiThreadedExecutor()
    executor.add_node(ros2_node)

    t = threading.Thread(target=executor.spin, daemon=True)
    t.start()
    return ros2_node, executor, t

def extract_head_yaw_pitch(R_base, R_head):

    R_rel = R_base.T @ R_head   # 旋转相对
    # 分解为 Rz(yaw) * Ry(pitch)
    yaw = np.arctan2(R_rel[1,0], R_rel[0,0])
    pitch = np.arctan2(-R_rel[2,0], np.sqrt(R_rel[2,1]**2 + R_rel[2,2]**2))
    return yaw, -pitch

if __name__ == "__main__":

    teleoperator = VuerTeleop("teleop/x100_hand_braincoV2.yml")  # 因时灵巧手替换为x100hand
    sm = SharedMemoryImage((480, 640))
    frame = sm.read_image()
    np.copyto(teleoperator.img_array, np.array(frame))
    # print(frame)
    arm_ik = X100_29_ArmIK(Visualization=True)  # 手臂逆解
    image_process = Process(
        target=image_receiver,
        args=(
            sm,
            teleoperator.resolution,
            teleoperator.crop_size_w,
            teleoperator.crop_size_h,
            # "/camera/color/image_raw",
        ),
    )
    image_process.start()
    theta = np.deg2rad(0)
    base_head_rmat  = np.array([
        [ np.cos(theta), 0,  np.sin(theta)],
        [ 0,             1,  0            ],
        [-np.sin(theta), 0,  np.cos(theta)]
    ], dtype=np.float64)

    # 仅当使用 ros2 模式时在后台启动 spin（放在 ros2 初始化之后、主循环之前）
        # 创建机械臂控制客户端 172.27.60.12
    # robot_client = X100_SocketClient(server_host='192.168.1.105', server_port=5556)
    
    # 等待客户端连接就绪
    # print("等待客户端连接服务端...")
    # while not robot_client.is_ready():
    #     time.sleep(0.1)
    ros2 = None
    ros_executor = None
    ros2, ros_executor, ros_spin_thread = start_ros_node()    
    # print("客户端已就绪，开始控制循环")
    qpos = np.zeros(28)  # 14+2+11*2
    # 给发现少量时间
    time.sleep(0.2)

    POSITION_ERR_THRESH = 1.0  # 位置误差阈值，可按需要调整（弧度）

    # 启动独立线程等待用户按 s 开始遥操
    from threading import Event, Thread

    start_teleop_event = Event()
    reset_hand = Event()

    last_warn_log_t = 0.0  # 超阈值警告节流
    last_prompt_log_t = 0.0  # 提示节流
    last_err_log_t = time.time()  # 一般误差打印节流(用当前时间避免启动即打印)
    nocollision_sol_q = np.zeros(14)
    def _wait_start():
        while True:
            try:
                user_input = input("按下 s 回车开始遥操: ").strip().lower()
                if user_input == "s":
                    reset_hand.clear()
                    start_teleop_event.set()
                    print("[Teleop] 已开始发送指令 (再输入 q 停止, 可选)")
                elif user_input == "q":
                    start_teleop_event.clear()
                    print("[Teleop] 已暂停发送指令, 输入 s 重新开始")
                elif user_input == "r":
                    # start_teleop_event.clear()
                    reset_hand.set()
                    start_teleop_event.set()
                    print("rest hand!!!")
            except EOFError:
                break
            except Exception:
                pass

    Thread(target=_wait_start, daemon=True).start()
    reset_hand_flag = True
    is_left_mode = True
    try:
        while True:

            armstate = None
            armv = None

            # ros2 订阅器会在 X100_ROS2 内部更新数据
            # armstate  = robot_client.get_robot_joint_positions()
            armstate, armv = ros2.get_arm_state()
            # print("Simulated arm state:", armstate)
            # armstate = [0.50108 , 0.7308  ,-0.49917 , 1.30575 , 0.21461 ,-1.48926 , 0.72797 ,-0.15158  ,1.50507 , 1.60497 , 0.39974 ,-1.62626 ,-1.4733  ,-0.34471]
            # print("Simulated arm velocity:", armv)
            frame = sm.read_image()
            # print("frame", frame.shape)
            
            np.copyto(teleoperator.img_array, np.array(frame))

            # print("111",base_head_rmat,head_rmat)

            head_rmat, left_pose, right_pose, left_qpos, right_qpos = (
                teleoperator.step()
            )                
            yaw,pitch = extract_head_yaw_pitch(base_head_rmat,head_rmat)
            # print("left_pose",left_pose)
            # print("right_pose",right_pose)
            # 根据当前的关节角度和关节速度计算
            
            armstate_ik = armstate.copy()
            # armstate_ik[:7] = [-0.54,1.5,-1.73,1.633,1.41,-1.56,0.34]
            armstate_ik[1:3] = [1.5,-1.73]
            # print("armstate_ik:", armstate_ik)
            ## add ik pose  2026/04/07
            sol_q = arm_ik.solve_ik(left_pose, right_pose, armstate_ik)
            # print(sol_q)
            # is_collision = arm_ik.compute_collision(sol_q)
            # if not is_collision:
            # sol_q=[0.50108 , 1.5,-1.73 , 1.30575 , 0.21461 ,-1.48926 , 0.72797,-0.15158  ,1.50507 , 1.60497 , 0.39974 ,-1.62626 ,-1.4733  ,-0.34471]
            nocollision_sol_q = sol_q
            if reset_hand.is_set():
                if is_left_mode:
                    # nocollision_sol_q [:7] = [-0.45,1.48,-1.49,2.09,1.40,-1.60,0.041]
                    nocollision_sol_q [:7] = [-0.026,1.48,-1.63,0.1,1.60,-1.52,0.12]
                    # nocollision_sol_q [:7] = [-0.23,1.5,-1.78,0.468,1.83,-1.53,0.02] #left_arm reset
                else:
                    nocollision_sol_q [7:] = [-0.23,1.5,1.5,0.43,-1.55,-1.48,0.02] #right_arm reset

            # print("nocollision_sol_q",nocollision_sol_q) 
            q = np.concatenate((
                np.array([yaw, pitch]),   # head
                nocollision_sol_q[:7],                # 手臂 IK 解
                np.array([left_qpos[0],left_qpos[0]*1.155,left_qpos[2],left_qpos[2]*1.155,left_qpos[4],left_qpos[4]*1.155,left_qpos[6],left_qpos[6]*1.155,left_qpos[8],left_qpos[9],left_qpos[9]]),                # left hand
                nocollision_sol_q[7:],
                np.array([right_qpos[0],right_qpos[0]*1.155,right_qpos[2],right_qpos[2]*1.155,right_qpos[4],right_qpos[4]*1.155,right_qpos[6],right_qpos[6]*1.155,right_qpos[8],right_qpos[9],right_qpos[9]]),                # left hand
            ), dtype=np.float64)

            arm_ik.vis.display(q)

            # qpos[:7] = sol_q[:7]
            # # qpos[:7] = [0, 0, 0, 0, 0, 0, 0]
            # qpos[13:20] = sol_q[7:]
            # # qpos[13:20] = [0, 0, 0, 0, 0, 0, 0]
            # # qpos[13:20] = armstate[7:]
            # qpos[14 - 7] = (left_qpos[9] + left_qpos[10]) * 0.7
            # qpos[15 - 7] = (left_qpos[0] + left_qpos[1]) * 0.7
            # qpos[16 - 7] = (left_qpos[2] + left_qpos[3]) * 0.7
            # qpos[17 - 7] = (left_qpos[6] + left_qpos[7]) * 0.7
            # qpos[18 - 7] = (left_qpos[4] + left_qpos[5]) * 0.7
            # qpos[19 - 7] = left_qpos[8]
            # qpos[20] = (right_qpos[9] + right_qpos[10]) * 0.7
            # qpos[21] = (right_qpos[0] + right_qpos[1]) * 0.7
            # qpos[22] = (right_qpos[2] + right_qpos[3]) * 0.7
            # qpos[23] = (right_qpos[6] + right_qpos[7]) * 0.7
            # qpos[24] = (right_qpos[4] + right_qpos[5]) * 0.7
            # qpos[25] = right_qpos[8]
            # time.sleep(0.01)
            if is_left_mode:
                nocollision_sol_q[7:] = armstate[7:]   # right_arm fix
            else:
                nocollision_sol_q[:7] = armstate[:7]   # left_arm fix

            if armstate is not None and len(armstate) == 14:
                try:
                    left_err = np.abs(armstate[:7] - nocollision_sol_q[:7])
                    right_err = np.abs(armstate[7:] - nocollision_sol_q[7:])
                    # 周期性打印当前误差（1s）
                    now_t = time.time()
                    # print("now_t", now_t)
                    # print("last_err_log_t", last_err_log_t)
                    if now_t - last_err_log_t > 1.0 :
                        print(f"位置误差(当前): 左臂 {left_err}, 右臂 {right_err}")
                        last_err_log_t = now_t
                    # 判断是否在阈值内
                    if (
                        np.all(left_err < POSITION_ERR_THRESH)
                        and np.all(right_err < POSITION_ERR_THRESH)
                        and start_teleop_event.is_set()
                    ):
                        ros2.publish_joint_command(
                            left_arm_qpos = nocollision_sol_q[:7],  # 左臂7个关节
                            right_arm_qpos = nocollision_sol_q[7:],  # 右臂7个关节
                            left_hand_qpos=[left_qpos[9]*1000,left_qpos[8]*1000,left_qpos[0]*1000,left_qpos[2]*1000,left_qpos[6]*1000,left_qpos[4]*1000],  # 左手6个关节
                            right_hand_qpos=[right_qpos[9]*1000,right_qpos[8]*1000,right_qpos[0]*1000,right_qpos[2]*1000,right_qpos[6]*1000,right_qpos[4]*1000]  # 右手6个关节
                        )
 
                        # robot_client.move_to_position(
                        #     arm_positions=nocollision_sol_q,
                        #     left_hand_positions=[left_qpos[9]*1000,left_qpos[8]*1000,left_qpos[0]*1000,left_qpos[2]*1000,left_qpos[6]*1000,left_qpos[4]*1000],
                        #     right_hand_positions=[right_qpos[9]*1000,right_qpos[8]*1000,right_qpos[0]*1000,right_qpos[2]*1000,right_qpos[6]*1000,right_qpos[4]*1000]
                        # )
                    elif  now_t - last_err_log_t > 1.0 and start_teleop_event.is_set():
                        print(f"位置误差(当前): 左臂 {left_err}, 右臂 {right_err}")
                        last_err_log_t = now_t                
                except Exception as e:
                    # 打印一次异常类型帮助调试（不高频）
                    pass
            # left_img, right_img = simulator.step(qpos)
            # np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))

    except KeyboardInterrupt:
        pass
    finally:
        try:
            # robot_client.shutdown()
            if ros_executor:
                ros_executor.shutdown()
            if ros2:
                ros2.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        try:
            teleoperator.shm.close()
            teleoperator.shm.unlink()
        except Exception:
            pass
        try:
            sm.cleanup()
        except Exception:
            pass
