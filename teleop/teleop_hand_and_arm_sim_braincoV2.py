import numpy as np
from isaacgym import gymapi, gymutil, gymtorch

# vision processing
from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor

# hand and arm control
from constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
import cv2
import zmq
import pickle
import zlib
import time

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

from teleop.robot_control.robot_arm_ik import X100_braincov2_ArmIK
from teleop.replay import X100_Sim_BraincoV2


def image_receiver(image_queue, resolution, crop_size_w, crop_size_h):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://192.168.1.100:5555")

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
        sm.write_image(frame)
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
            self.resolution[1] - self.crop_size_w,
        )

        # 创建共享内存
        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]
        print("np.prod(self.img_shape)", np.prod(self.img_shape))
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
    def __init__(self, config_file_path):
        self.resolution = (720, 1280)  # (720, 1280) 图像分辨率
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (
            self.resolution[0] - self.crop_size_h,
            self.resolution[1] - self.crop_size_w,
        )

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]
        print("np.prod(self.img_shape)", np.prod(self.img_shape))
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
        self.processor = VuerPreprocessor()

        # 重定向配置
        RetargetingConfig.set_default_urdf_dir("./assets")
        with Path(config_file_path).open("r") as f:
            cfg = yaml.safe_load(f)

        left_retargeting_config = RetargetingConfig.from_dict(cfg["left"])
        right_retargeting_config = RetargetingConfig.from_dict(cfg["right"])

        # print("\\n构建retargeting对象...")
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()
        self.target_joint_indices = self.left_retargeting.optimizer.target_joint_indices

    def step(self):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = (
            self.processor.process(self.tv)
        )
        head_rmat = head_mat[:3, :3]
        # 转换头部坐标原点到机器人坐标原点 g1: head -> pelvis
        left_wrist_mat[2, 3] += 1.38
        right_wrist_mat[2, 3] += 1.38
        left_wrist_mat[0, 3] += 0.15
        right_wrist_mat[0, 3] += 0.15

        # 直接使用重定向器输出，不需要重新排序
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[
            self.target_joint_indices
        ]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[
            self.target_joint_indices
        ]

        return head_rmat, left_wrist_mat, right_wrist_mat, left_qpos, right_qpos


if __name__ == "__main__":

    teleoperator = VuerTeleop("teleop/braincov2_hand.yml")  # braincoV2 hand
    arm_ik = X100_braincov2_ArmIK(Visualization=True)  # 手臂逆解
    simulator = X100_Sim_BraincoV2()
    sm = SharedMemoryImage((720, 1280))
    qpos = np.zeros(38)  # 38个机器人关节
    image_process = Process(
        target=image_receiver,
        args=(
            sm,
            teleoperator.resolution,
            teleoperator.crop_size_w,
            teleoperator.crop_size_h,
        ),
    )
    image_process.start()
    try:
        while True:
            armstate = None
            armv = None
            # 获取手部位置
            # handstate = h1hand.get_hand_state()
            # 获取手臂位置  左右 位置 速度
            armstate, armv = simulator.get_arm_state()
            frame = sm.read_image()
            np.copyto(teleoperator.img_array, np.array(frame))
            print(frame)
            head_rmat, left_pose, right_pose, left_qpos, right_qpos = (
                teleoperator.step()
            )
            # 根据当前的关节角度和关节速度计算
            sol_q = arm_ik.solve_ik(left_pose, right_pose, armstate, armv)
            qpos[2:9] = sol_q[:7]  # 左臂关节：索引 2-8
            # logging.info("Left arm pose: %s", left_pose)
            zero_hand_pos = np.zeros(11)
            qpos[9:20] = left_qpos  # 左手关节：索引 9-19
            qpos[20:27] = sol_q[7:]  # 右臂关节：索引 20-26
            qpos[27:38] = right_qpos  # 右手关节：索引 27-37

            left_img, right_img = simulator.step(qpos)
            # np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))

    except KeyboardInterrupt:
        exit(0)
