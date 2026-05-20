import threading
import numpy as np
import time
from time import sleep
import sys
import os

# ==========================================
# 1. 动态加载 SDK 路径 (适配你的文件夹结构)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# 严格按照你截图中的路径和拼写：DataRead_Python_Linux_SDK -> motion_caputure
sdk_path = os.path.join(current_dir, "DataRead_Python_Linux_SDK", "motion_caputure")

if sdk_path not in sys.path:
    sys.path.append(sdk_path)

SDK_IMPORT_ERROR = None
try:
    from vdmocapsdk_dataread import *
    from vdmocapsdk_nodelist import *
except ImportError as e:
    SDK_IMPORT_ERROR = e
    print(f"[致命错误] 无法导入动捕手套 SDK！请检查 {sdk_path} 路径下是否有对应文件。错误详情: {e}")

# ==========================================
# 2. 全局变量与默认骨架 (T-Pose)
# ==========================================
global_index = 0
global_world_space = 0

global_initial_position_body = [
    [0, 0, 1.022], [0.074, 0, 1.002], [0.097, 0, 0.593], [0.104, 0, 0.111],
    [0.114, 0.159, 0.005], [-0.074, 0, 1.002], [-0.097, 0.001, 0.593],
    [-0.104, 0, 0.111], [-0.114, 0.158, 0.004], [0, 0.033, 1.123],
    [0, 0.03, 1.246], [0, 0.014, 1.362], [0, -0.048, 1.475],
    [0, -0.048, 1.549], [0, -0.016, 1.682], [0.071, -0.061, 1.526],
    [0.178, -0.061, 1.526], [0.421, -0.061, 1.526], [0.682, -0.061, 1.526],
    [-0.071, -0.061, 1.526], [-0.178, -0.061, 1.526], [-0.421, -0.061, 1.526],
    [-0.682, -0.061, 1.526],
]

global_initial_position_hand_right = [
    [0.682, -0.061, 1.526], [0.71, -0.024, 1.526], [0.728, -0.008, 1.526],
    [0.755, 0.013, 1.526], [0.707, -0.05, 1.526], [0.761, -0.024, 1.525],
    [0.812, -0.023, 1.525], [0.837, -0.022, 1.525], [0.709, -0.058, 1.526],
    [0.764, -0.046, 1.528], [0.816, -0.046, 1.528], [0.845, -0.046, 1.528],
    [0.709, -0.064, 1.526], [0.761, -0.069, 1.527], [0.812, -0.069, 1.527],
    [0.835, -0.069, 1.527], [0.708, -0.072, 1.526], [0.755, -0.089, 1.522],
    [0.791, -0.089, 1.522], [0.81, -0.089, 1.522],
]

global_initial_position_hand_left = [
    [-0.682, -0.061, 1.526], [-0.71, -0.024, 1.526], [-0.728, -0.008, 1.526],
    [-0.755, 0.013, 1.526], [-0.707, -0.05, 1.526], [-0.761, -0.024, 1.525],
    [-0.812, -0.023, 1.525], [-0.837, -0.022, 1.525], [-0.709, -0.058, 1.526],
    [-0.764, -0.046, 1.528], [-0.816, -0.046, 1.528], [-0.845, -0.046, 1.528],
    [-0.709, -0.064, 1.526], [-0.761, -0.069, 1.527], [-0.812, -0.069, 1.527],
    [-0.835, -0.069, 1.527], [-0.708, -0.072, 1.526], [-0.755, -0.089, 1.522],
    [-0.791, -0.089, 1.522], [-0.81, -0.089, 1.522],
]

# ==========================================
# 3. 核心封装类：MocapGloveClient
# ==========================================
class MocapGloveClient:
    def __init__(self, dst_ip="172.27.62.145", dst_port=7000, local_port=0, index=0):
        if SDK_IMPORT_ERROR is not None:
            raise ImportError(f"无法导入动捕手套 SDK: {SDK_IMPORT_ERROR}") from SDK_IMPORT_ERROR

        self.ip = dst_ip
        self.port = dst_port
        self.local_port = local_port
        self.index = index
        self.mocap_data = MocapData()
        self.running = False
        self.connected = False
        self.update_count = 0
        self.last_update_t = None
        self.last_error = None
        
        self.left_hand_mat = np.zeros((21, 3))
        self.right_hand_mat = np.zeros((21, 3))
        self.lock = threading.Lock()
        
        self._connect()
        self._start_thread()
        
    def _connect(self):
        if not udp_is_open(self.index):
            if not udp_open(self.index, self.local_port):
                self.last_error = f"udp_open_failed local_port={self.local_port}"
                print(f"[Glove] 手套 UDP 端口打开失败！(Local Port: {self.local_port})")
                return
        
        udp_set_position_in_initial_tpose(
            self.index, self.ip, self.port, 
            global_world_space, global_initial_position_body,
            global_initial_position_hand_right,
            global_initial_position_hand_left
        )
        
        if not udp_send_request_connect(self.index, self.ip, self.port):
            self.last_error = f"connect_request_failed dst={self.ip}:{self.port}"
            print(f"[Glove] 向 {self.ip}:{self.port} 发送 UDP 连接请求失败！")
            return

        self.connected = True
        self.last_error = None
        print(f"[Glove] 手套动捕系统连接成功！正在监听 {self.ip}:{self.port} ...")

    def _start_thread(self):
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        
    def _update_loop(self):
        while self.running:
            try:
                if udp_recv_mocap_data(self.index, self.ip, self.port, self.mocap_data):
                    if self.mocap_data.isUpdate:
                        self._parse_data()
            except Exception as exc:
                with self.lock:
                    self.last_error = str(exc)
                sleep(0.02)
            sleep(0.002)
            
    def _parse_data(self):
        with self.lock:
            # --- 处理左手 ---
            self.left_hand_mat[0] = [self.mocap_data.position_lHand[0][i] for i in range(3)]
            self.left_hand_mat[4]  = [self.mocap_data.position_lHand[3][i] for i in range(3)]
            self.left_hand_mat[8]  = [self.mocap_data.position_lHand[7][i] for i in range(3)]
            self.left_hand_mat[12] = [self.mocap_data.position_lHand[11][i] for i in range(3)]
            self.left_hand_mat[16] = [self.mocap_data.position_lHand[15][i] for i in range(3)]
            self.left_hand_mat[20] = [self.mocap_data.position_lHand[19][i] for i in range(3)]

            # --- 处理右手 ---
            self.right_hand_mat[0] = [self.mocap_data.position_rHand[0][i] for i in range(3)]
            self.right_hand_mat[4]  = [self.mocap_data.position_rHand[3][i] for i in range(3)]
            self.right_hand_mat[8]  = [self.mocap_data.position_rHand[7][i] for i in range(3)]
            self.right_hand_mat[12] = [self.mocap_data.position_rHand[11][i] for i in range(3)]
            self.right_hand_mat[16] = [self.mocap_data.position_rHand[15][i] for i in range(3)]
            self.right_hand_mat[20] = [self.mocap_data.position_rHand[19][i] for i in range(3)]
            self.update_count += 1
            self.last_update_t = time.monotonic()
            self.last_error = None
            
    def get_left_hand(self):
        with self.lock:
            return self.left_hand_mat.copy()
            
    def get_right_hand(self):
        with self.lock:
            return self.right_hand_mat.copy()

    def get_status(self):
        with self.lock:
            now_t = time.monotonic()
            data_age = None
            if self.last_update_t is not None:
                data_age = now_t - self.last_update_t
            return {
                "connected": self.connected,
                "running": self.running,
                "update_count": self.update_count,
                "last_update_t": self.last_update_t,
                "data_age": data_age,
                "last_error": self.last_error,
                "left_valid": self.last_update_t is not None,
            }

    def close(self):
        print("[Glove] 正在关闭手套连接...")
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.connected and udp_remove(self.index, self.ip, self.port):
            udp_close(self.index)
        print("[Glove] 连接已断开。")
