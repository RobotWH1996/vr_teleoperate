# AVP Teleop — Apple Vision Pro 人形机器人遥操作系统

基于 Apple Vision Pro 的双臂灵巧手遥操作系统，支持 Unitree H1_2 / X100 / K100 人形机器人平台。

---

## 系统架构

```
Apple Vision Pro ──(WebXR/Vuer)──→ OpenTeleVision
                                        │
                                   Preprocessor (坐标变换 Y-up → Z-up)
                                        │
                            ┌───────────┴───────────┐
                            │                       │
                     dex_retargeting            IK Solver
                     (指尖→手指关节)         (Pinocchio+CasADi)
                            │                       │
                            ▼                       ▼
                      手指 6-DoF×2            手臂 7-DoF×2
                            │                       │
                            └───────┬───────────────┘
                                    │
                         ┌──────────┼──────────┐
                         │          │          │
                        DDS       ROS2      Socket
                      (H1_2)   (X100/K100)  (通用)
                         │          │          │
                         ▼          ▼          ▼
                              机器人本体
```

## 特性

- **多平台支持**: Unitree H1_2 (DDS) / X100 & K100 (ROS2)
- **多种灵巧手**: Inspire Hand / BrainCo V2 / X100 Hand
- **双输入模式**: Apple Vision Pro 手势识别 / 动捕数据手套
- **实时立体视觉**: 机器人双目→AVP 沉浸式显示
- **仿真验证**: Isaac Gym + Meshcat 3D 可视化
- **延迟测量**: 内置臂部指令-反馈延迟基准测试工具

---

## 目录结构

```
avp_teleop_release/
├── README.md                          ← 本文件
├── requirements.txt                   ← Python 依赖
├── environment.yml                    ← Conda 完整环境导出
│
├── config/                            ← 灵巧手重定向配置
│   ├── inspire_hand.yml               ← Inspire Hand
│   ├── x100_hand.yml                  ← X100 Hand
│   └── x100_hand_braincoV2.yml        ← BrainCo V2
│
├── teleop/                            ← 核心源码
│   ├── TeleVision.py                  ← WebXR 服务器
│   ├── Preprocessor.py                ← 坐标预处理
│   ├── constants_vuer.py              ← 变换矩阵常量
│   ├── motion_utils.py                ← 矩阵工具
│   ├── replay.py                      ← 仿真/ROS2/Socket 控制
│   ├── teleop_hand_and_arm.py         ← 硬件入口 (H1_2+DDS)
│   ├── teleop_hand_and_arm_sim.py     ← 仿真入口 (ROS2)
│   ├── teleop_hand_and_arm_sim_glove.py ← 手套入口
│   ├── teleop_hand_and_arm_sim_braincoV2.py ← K100 BrainCo V2 仿真入口
│   ├── robot_control/
│   │   ├── robot_arm.py               ← H1 臂部控制 (DDS)
│   │   ├── robot_hand.py              ← Inspire 手控制 (DDS)
│   │   └── robot_arm_ik.py            ← IK 求解器
│   ├── image_server/
│   │   ├── image_server.py            ← ZMQ 图像推流
│   │   └── image_client.py            ← 图像测试客户端
│   ├── webrtc/
│   │   └── zed_server.py              ← WebRTC 视频流
│   ├── mocap_sdk/
│   │   └── mocap_glove_client.py      ← 动捕手套客户端
│   └── latency/
│       └── measure_arm_latency.py     ← 延迟测量工具
│
├── assets/                            ← 机器人 URDF 与 mesh
│   ├── k100_description/              ← K100 + BrainCo 手 (IK/仿真)
│   └── k100_description_braincoV2/    ← K100 BrainCo V2 型号
│
├── examples/k100_arm_demo/            ← K100 手臂 ROS2 控制示例 (PallasSDK)
│
└── docs/
    └── TECHNICAL_DOC.md               ← 详细技术文档
```

---

## 环境安装

### 1. 创建 Conda 环境

```bash
conda create -n avp_teleop python=3.10
conda activate avp_teleop
```

### 2. 安装 Pinocchio (运动学库)

```bash
conda install -c conda-forge pinocchio
```

### 3. 安装 CasADi (数值优化)

```bash
conda install -c conda-forge casadi
```

### 4. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 5. 安装 Unitree DDS (H1_2 硬件模式需要)

```bash
pip install unitree_dds_wrapper
```

### 6. 生成 TLS 证书 (Vuer/WebXR 要求 HTTPS)

```bash
# 安装 mkcert
sudo apt install mkcert
mkcert -install

# 在 teleop/ 目录下生成证书
cd teleop
mkcert -cert-file cert.pem -key-file key.pem localhost <你的IP>
```

### 7. (可选) 安装 Isaac Gym (仿真)

按照 NVIDIA Isaac Gym 官方文档安装。

---

## 快速开始

### 模式 1: 硬件控制 (H1_2 机器人)

**前置条件**: H1_2 机器人开机, DDS 网络可达, Inspire 手服务已启动

```bash
# 1. 在机器人端启动图像流
cd teleop/image_server
python image_server.py

# 2. 在主控 PC 启动遥操
cd teleop
python teleop_hand_and_arm.py

# 3. 在 AVP 浏览器中打开 https://<主控PC-IP>:8012
# 4. 终端中输入 's' 回车开始遥操
```

### 模式 2: 仿真 + ROS2 (X100/K100)

**前置条件**: ROS2 环境已 source, 机器人 ROS2 节点运行

```bash
# 启动遥操 (含 Meshcat 可视化)
cd teleop
python teleop_hand_and_arm_sim.py

# 在 AVP 浏览器中打开 https://<IP>:8012
# 输入 's' 开始 / 'q' 暂停 / 'r' 复位手臂
```

### 模式 3: 仿真 + 数据手套

```bash
# 确保动捕手套软件在 192.168.1.100:7000 广播
cd teleop
python teleop_hand_and_arm_sim_glove.py
```

---

## 运行时控制

| 按键 | 功能 |
|------|------|
| `s` + Enter | 开始发送遥操指令 |
| `q` + Enter | 暂停发送 |
| `r` + Enter | 复位手臂到安全位置 |
| `Ctrl+C` | 退出程序 |

---

## 关键技术参数

| 参数 | 值 |
|------|-----|
| 手部追踪频率 | 60 Hz (AVP 原生) |
| 控制回路频率 | ~30-60 Hz |
| IK 最大迭代数 | 50 次/帧 |
| 臂部自由度 | 7×2 = 14 DoF |
| 手部自由度 | 6×2 = 12 DoF |
| 图像分辨率 | 480×640 (单目) |
| 端到端延迟 | ~30-100 ms |

---

## 技术栈

| 组件 | 库/框架 |
|------|---------|
| WebXR 服务器 | [Vuer](https://github.com/vuer-ai/vuer) |
| 运动学/动力学 | [Pinocchio](https://github.com/stack-of-tasks/pinocchio) |
| 数值优化 | [CasADi](https://web.casadi.org/) + IPOPT |
| 手势重定向 | [dex_retargeting](https://github.com/dexsuite/dex-retargeting) |
| DDS 通信 | unitree_dds_wrapper |
| ROS2 通信 | rclpy (sensor_msgs, trajectory_msgs) |
| 图像传输 | ZeroMQ (PUSH/PULL) |
| 仿真 | NVIDIA Isaac Gym |
| 3D 可视化 | Meshcat |
| 深度学习 | PyTorch (ACT 模仿学习) |

---

## 详细文档

完整的系统架构、数据流、坐标系变换、IK 优化细节、通信协议等详见:

**[技术文档 (TECHNICAL_DOC.md)](docs/TECHNICAL_DOC.md)**

---

## 致谢

本项目基于以下开源工作:

- [TeleVision / OpenTeleVision](https://github.com/OpenTeleVision/TeleVision) — WebXR 遥操作框架
- [dex-retargeting](https://github.com/dexsuite/dex-retargeting) — 灵巧手重定向
- [Vuer](https://github.com/vuer-ai/vuer) — WebXR 渲染引擎
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio) — 刚体运动学
- [unitree_dds_wrapper](https://github.com/unitreerobotics/unitree_dds_wrapper) — Unitree DDS 通信
- [ACT (Action Chunking with Transformers)](https://github.com/tonyzhaozh/act) — 模仿学习
