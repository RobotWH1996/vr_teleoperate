# AVP Teleoperation 技术文档

> **版本**: 1.0  
> **最后更新**: 2026-04-13  
> **项目名称**: Apple Vision Pro 遥操作系统 (AVP Teleop)

---

## 目录

- [1. 系统概述](#1-系统概述)
- [2. 系统架构](#2-系统架构)
  - [2.1 总体架构图](#21-总体架构图)
  - [2.2 软件分层架构](#22-软件分层架构)
  - [2.3 进程与线程模型](#23-进程与线程模型)
- [3. 核心模块详解](#3-核心模块详解)
  - [3.1 WebXR 感知层 — OpenTeleVision](#31-webxr-感知层--opentelevsion)
  - [3.2 坐标预处理 — Preprocessor](#32-坐标预处理--preprocessor)
  - [3.3 手势重定向 — dex_retargeting](#33-手势重定向--dex_retargeting)
  - [3.4 逆运动学 — IK Solver](#34-逆运动学--ik-solver)
  - [3.5 机器人控制层](#35-机器人控制层)
  - [3.6 图像流服务](#36-图像流服务)
  - [3.7 动捕手套模块](#37-动捕手套模块)
  - [3.8 仿真环境](#38-仿真环境)
  - [3.9 延迟测量工具](#39-延迟测量工具)
- [4. 数据流详细分析](#4-数据流详细分析)
  - [4.1 主控制回路数据流](#41-主控制回路数据流)
  - [4.2 图像流数据流](#42-图像流数据流)
  - [4.3 关节数据格式](#43-关节数据格式)
- [5. 坐标系与变换详解](#5-坐标系与变换详解)
  - [5.1 坐标系定义](#51-坐标系定义)
  - [5.2 坐标变换矩阵](#52-坐标变换矩阵)
  - [5.3 变换链路](#53-变换链路)
- [6. IK 优化问题详解](#6-ik-优化问题详解)
  - [6.1 优化变量与参数](#61-优化变量与参数)
  - [6.2 代价函数](#62-代价函数)
  - [6.3 约束条件](#63-约束条件)
  - [6.4 求解策略](#64-求解策略)
- [7. 通信协议与接口](#7-通信协议与接口)
  - [7.1 Unitree DDS 接口](#71-unitree-dds-接口)
  - [7.2 ROS2 话题接口](#72-ros2-话题接口)
  - [7.3 ZMQ 图像流协议](#73-zmq-图像流协议)
  - [7.4 Socket 通信协议](#74-socket-通信协议)
- [8. 运行模式](#8-运行模式)
  - [8.1 硬件模式 (H1_2)](#81-硬件模式-h1_2)
  - [8.2 仿真模式 (Isaac Gym + ROS2)](#82-仿真模式-isaac-gym--ros2)
  - [8.3 数据手套模式](#83-数据手套模式)
- [9. 机器人模型与 URDF](#9-机器人模型与-urdf)
- [10. 手势重定向配置](#10-手势重定向配置)
- [11. 性能与延迟分析](#11-性能与延迟分析)
- [12. 安全机制](#12-安全机制)
- [13. 文件索引](#13-文件索引)

---

## 1. 系统概述

本项目是一套基于 **Apple Vision Pro (AVP)** 的人形机器人双臂灵巧手遥操作系统。操作者佩戴 AVP 头显，系统通过 WebXR 协议实时捕获操作者的**头部姿态**、**双手腕部 4x4 变换矩阵**和 **25 个手部关键点**，经过坐标变换、手势重定向和逆运动学求解后，驱动机器人执行相应的双臂和灵巧手动作。

### 核心能力

| 能力 | 说明 |
|------|------|
| **头部追踪** | 6-DoF 头部姿态 → 机器人头部 yaw/pitch |
| **双臂控制** | 双手腕部 6-DoF 位姿 → 双臂各 7 自由度关节角 |
| **灵巧手控制** | 25 点手部关键点 → 各 6 自由度手指关节 |
| **立体视觉** | 机器人端双目图像 → AVP 端立体渲染 |
| **多模态输入** | VR 手势识别 / 动捕数据手套 可切换 |

### 支持的机器人平台

| 平台 | 本体 | 灵巧手 | 通信方式 |
|------|------|--------|---------|
| Unitree H1_2 | 人形双臂 (14-DoF) | Inspire Hand (12-DoF) | DDS |
| X100 / K100 | 双臂 (14-DoF) | BrainCo V2 (22-DoF) | ROS2 / Socket |

---

## 2. 系统架构

### 2.1 总体架构图

```
┌──────────────────────────────────────────────────────────────────────┐
│                      操作者 (Operator)                               │
│                                                                      │
│   ┌─────────────────────┐    ┌──────────────────────────────────┐   │
│   │  Apple Vision Pro   │    │  动捕手套 (可选)                  │   │
│   │  ┌───────────────┐  │    │  MocapGloveClient               │   │
│   │  │ WebXR Hands   │──┼────┤  UDP → 21点x3D 手部骨架          │   │
│   │  │ Camera Pose   │  │    └──────────────────────────────────┘   │
│   │  │ Stereo View   │  │                                           │
│   │  └───────────────┘  │                                           │
│   └────────┬────────────┘                                           │
└────────────┼────────────────────────────────────────────────────────┘
             │ HTTPS/WebSocket (Vuer)
             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     主控 PC (Host Computer)                          │
│                                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │ OpenTele-   │  │ Vuer-        │  │ dex_retargeting           │  │
│  │ Vision      │──│ Preprocessor │──│ (手势重定向)               │  │
│  │ (WebXR      │  │ (坐标变换)    │  │ 人手指尖→机器人手指关节    │  │
│  │  Server)    │  └──────┬───────┘  └─────────┬─────────────────┘  │
│  └─────────────┘         │                    │                     │
│                          ▼                    ▼                     │
│                 ┌────────────────┐   ┌─────────────────┐           │
│                 │ Arm IK Solver  │   │ 手指关节角        │           │
│                 │ (Pinocchio +   │   │ (6-DoF × 2)     │           │
│                 │  CasADi)       │   └────────┬────────┘           │
│                 └───────┬────────┘            │                     │
│                         │                     │                     │
│                         ▼                     ▼                     │
│                 ┌─────────────────────────────────────┐             │
│                 │          通信层                       │             │
│                 │  DDS / ROS2 / Socket / ZMQ          │             │
│                 └───────────────┬─────────────────────┘             │
└─────────────────────────────────┼───────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       机器人 (Robot)                                 │
│                                                                      │
│   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────────┐   │
│   │ 左臂 7-DoF │  │ 右臂 7-DoF │  │ 左手 6-DoF │  │ 右手 6-DoF  │   │
│   └───────────┘  └───────────┘  └───────────┘  └──────────────┘   │
│                                                                      │
│   ┌────────────────────────────────┐                                │
│   │ 双目摄像头 → ZMQ JPEG 流       │                                │
│   └────────────────────────────────┘                                │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 软件分层架构

```
┌─────────────────────────────────────────────────────────┐
│                    应用层 (Entry Scripts)                 │
│  teleop_hand_and_arm.py      ← 硬件真机 (H1_2 + DDS)   │
│  teleop_hand_and_arm_sim.py  ← 仿真 (Isaac Gym + ROS2) │
│  teleop_hand_and_arm_sim_glove.py ← 仿真 + 数据手套     │
├─────────────────────────────────────────────────────────┤
│                   遥操核心层 (VuerTeleop)                 │
│  ┌──────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │ OpenTele │  │ Preprocessor│  │ dex_retargeting  │   │
│  │ Vision   │  │             │  │ RetargetingConfig │   │
│  └──────────┘  └─────────────┘  └──────────────────┘   │
├─────────────────────────────────────────────────────────┤
│                   运动规划层 (IK + Filter)                │
│  ┌───────────────────┐  ┌──────────────────────────┐   │
│  │ X100_29_ArmIK     │  │ WeightedMovingFilter     │   │
│  │ X100_braincov2_IK │  │ (加权移动平均滤波)         │   │
│  │ Arm_IK (H1_2)     │  └──────────────────────────┘   │
│  └───────────────────┘                                   │
├─────────────────────────────────────────────────────────┤
│                   机器人接口层 (Robot Interface)          │
│  ┌───────────────┐ ┌─────────────┐ ┌────────────────┐  │
│  │ H1ArmController│ │H1HandContrl │ │ X100_ROS2      │  │
│  │ (DDS LowCmd)  │ │(DDS Inspire)│ │ (ROS2 JointSt) │  │
│  └───────────────┘ └─────────────┘ └────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                   感知层 (Perception)                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ image_server│  │ image_client │  │ WebRTC (ZED)   │ │
│  │ (ZMQ PUSH)  │  │ (ZMQ PULL)   │  │ (aiohttp)      │ │
│  └─────────────┘  └──────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 2.3 进程与线程模型

```
┌─────────────── 主进程 (Main Process) ───────────────────┐
│                                                          │
│  主线程 (Main Thread)                                     │
│  ├── 控制循环: AVP读取 → 预处理 → IK → 发送指令            │
│  │                                                       │
│  ├── 用户输入线程 (_wait_start)                            │
│  │   └── 等待 's' 启动 / 'q' 停止 / 'r' 复位              │
│  │                                                       │
│  ├── ROS2 Executor 线程                                   │
│  │   └── MultiThreadedExecutor.spin()                    │
│  │       ├── left_arm_state 回调                          │
│  │       └── right_arm_state 回调                         │
│  │                                                       │
│  └── (可选) MocapGloveClient 线程                         │
│      └── UDP 接收 → _parse_data()                        │
│                                                          │
├─────────── 子进程 1: Vuer WebXR Server ─────────────────┤
│  └── asyncio event loop                                  │
│      ├── on_hand_move handler → shared memory             │
│      ├── on_cam_move handler → shared memory              │
│      └── main_image loop → 渲染立体图像                    │
│                                                          │
├─────────── 子进程 2: Image Receiver ────────────────────┤
│  └── ZMQ PULL → decompress → shared memory               │
│                                                          │
└──── (可选) 子进程 3: WebRTC Server ─────────────────────┘
      └── aiohttp web server (port 8080)                   
```

---

## 3. 核心模块详解

### 3.1 WebXR 感知层 — OpenTeleVision

**文件**: `teleop/TeleVision.py`

OpenTeleVision 是整个系统的入口感知模块，基于 [Vuer](https://github.com/vuer-ai/vuer) 框架搭建 WebXR 服务器，接收 Apple Vision Pro 的手部和头部追踪数据。

#### 类结构

```python
class OpenTeleVision:
    """
    核心属性（通过 multiprocessing.Array 跨进程共享）:
    - left_hand_shared  (16,)  → 4×4 矩阵（列优先展开）
    - right_hand_shared (16,)  → 4×4 矩阵（列优先展开）
    - left_landmarks_shared  (75,) → 25×3 关键点
    - right_landmarks_shared (75,) → 25×3 关键点
    - head_matrix_shared (16,) → 4×4 头部矩阵
    - aspect_shared (1,) → 摄像头宽高比
    """
```

#### WebXR 事件处理

| 事件 | 处理器 | 数据 |
|------|--------|------|
| `HAND_MOVE` | `on_hand_move()` | leftHand(16), rightHand(16), leftLandmarks(25×3), rightLandmarks(25×3) |
| `CAMERA_MOVE` | `on_cam_move()` | camera.matrix(16), camera.aspect(1) |

#### 图像流模式

| 模式 | 实现 | 特点 |
|------|------|------|
| `image` | SharedMemory + JPEG | 通过共享内存接收图像，JPEG 编码后推送到 AVP |
| `webrtc` | WebRTC + aiohttp | 通过 WebRTC 协议直接推送视频流，延迟更低 |

#### 手部关键点索引 (25点)

```
 0: 手腕 (Wrist)
 1-4: 拇指 (Thumb) — metacarpal, proximal, distal, tip
 5-8: 食指 (Index)
 9-12: 中指 (Middle)
 13-16: 无名指 (Ring)
 17-20: 小指 (Pinky)
 21-24: 各指间关节
 
 指尖索引 (tip_indices) = [4, 9, 14, 19, 24]
```

### 3.2 坐标预处理 — Preprocessor

**文件**: `teleop/Preprocessor.py`

将 AVP 原始坐标系（Y-up）转换为机器人坐标系（Z-up），并计算相对位姿。

#### VuerPreprocessor / VuerPreprocessorLegacy

```
输入:
  tv.head_matrix      → 4×4 头部齐次变换矩阵 (Y-up)
  tv.left_hand         → 4×4 左手腕部矩阵 (Y-up)
  tv.right_hand        → 4×4 右手腕部矩阵 (Y-up)
  tv.left_landmarks    → 25×3 左手关键点 (Y-up)
  tv.right_landmarks   → 25×3 右手关键点 (Y-up)

处理步骤:
  1. 矩阵有效性检查 (mat_update: det ≠ 0)
  2. 坐标系变换: grd_yup2grd_zup
  3. 腕部矩阵 × 手臂对齐矩阵 (X100_l_arm / X100_r_arm)
  4. 平移: 减去头部位置 → 相对腕部位置
  5. 手指: 齐次坐标 → 变换 → 腕部坐标系下的相对位置
  6. 手指: × 手臂到手的对齐矩阵 (X100_arm_2_hand)

输出:
  head_mat            → 4×4 (Z-up)
  rel_left_wrist_mat  → 4×4 相对头部的左腕位姿
  rel_right_wrist_mat → 4×4 相对头部的右腕位姿
  rel_left_fingers    → 25×3 左手腕部坐标系下的手指位置
  rel_right_fingers   → 25×3 右手腕部坐标系下的手指位置
```

#### 坐标偏移 (在 VuerTeleop.step 中)

```python
# H1_2 模式: 头→骨盆偏移
left_wrist_mat[2, 3] += 0.45   # Z 方向上移
right_wrist_mat[0, 3] += 0.20  # X 方向前移

# X100 模式: 头→底座偏移
left_wrist_mat[2, 3] += 1.28
left_wrist_mat[0, 3] -= 0.1
```

### 3.3 手势重定向 — dex_retargeting

**配置文件**: `config/inspire_hand.yml`, `config/x100_hand.yml`, `config/x100_hand_braincoV2.yml`

使用 `dex_retargeting` 库实现**向量法手势重定向**——将人手指尖位置映射为机器人手指关节角度。

#### 配置结构

```yaml
left:
  type: vector                # 向量重定向方法
  urdf_path: xxx.urdf         # 机器人手 URDF
  wrist_link_name: "L_hand_base_link"
  target_joint_names: [...]   # 目标关节名 (11-12个)
  target_origin_link_names: ["L_hand_base_link" × 5]  # 基准链接
  target_task_link_names: ["L_thumb_tip", ...]         # 指尖链接
  scaling_factor: 0.9~1.5     # 手大小缩放比
  target_link_human_indices: [[0,0,0,0,0], [4,9,14,19,24]]  # 人手关键点→指尖映射
  low_pass_alpha: 0.8         # 低通滤波系数
```

#### 重定向流程

```
人手 25 关键点
    │
    ├── 取指尖 5 点 (tip_indices = [4, 9, 14, 19, 24])
    │
    ▼
retarget(tip_positions)
    │
    ├── 计算源向量: 手腕→指尖方向
    ├── 计算目标向量: URDF 基准链接→指尖链接
    ├── 优化: 最小化向量方向差异
    │
    ▼
target_joint_indices 选择
    │  (从完整关节角中选取目标关节)
    ▼
left_qpos / right_qpos (6~12 维关节角)
```

### 3.4 逆运动学 — IK Solver

**文件**: `teleop/robot_control/robot_arm_ik.py`

三种 IK 求解器均基于 **Pinocchio + CasADi** 实现数值优化 IK。

#### 类层次

| 类名 | 适用机型 | URDF | 关节数 |
|------|---------|------|--------|
| `Arm_IK` | Unitree H1_2 | `h1_2/h1_2.urdf` | 14 (双臂各7) |
| `X100_29_ArmIK` | X100/K100 | `k100_description/k100_brainco.urdf` | 14 |
| `X100_braincov2_ArmIK` | X100 BrainCo V2 | `x100_urdf_ros1/x100_38_brainco.urdf` | 14 |

#### IK 求解原理

```
输入:
  left_wrist   → 4×4 目标左手末端位姿
  right_wrist  → 4×4 目标右手末端位姿
  current_q    → 当前关节角度 (初始值)
  current_dq   → 当前关节角速度

优化问题 (CasADi/IPOPT):
  minimize:
    50 × translational_cost        ← 位移误差（L2距离）
    + 2 × rotation_cost            ← 旋转误差（log3 旋转向量）
    + 0.01 × regularization_cost   ← 正则化（关节角L2范数）
    + 0.05~0.5 × smooth_cost       ← 平滑项（与上一时刻解的差）

  subject to:
    q_lower ≤ q ≤ q_upper          ← 关节限位

  solver: IPOPT (max_iter=50, tol=1e-4~1e-6)

输出:
  sol_q → 14 维关节角度解
```

#### 加权移动滤波器

```python
class WeightedMovingFilter:
    """
    权重: [0.4, 0.3, 0.2, 0.1]  ← 最近帧权重最大
    维度: 14 (双臂关节数)
    用途: IK 求解失败时, 用历史数据做加权平均
    """
```

### 3.5 机器人控制层

#### H1ArmController (DDS)

**文件**: `teleop/robot_control/robot_arm.py`

```
通信协议: Unitree DDS
话题:
  发布: rt/lowcmd  (LowCmd_)
  订阅: rt/lowstate (LowState_)

初始化流程:
  1. 等待 lowstate 连接
  2. 读取当前关节角度
  3. 锁定腿部关节 (1000步平滑过渡)
  4. 启动三个线程:
     - SubscribeState:  2ms 周期读取状态
     - Control:         2ms 周期计算指令
     - LowCommandWriter: 2ms 周期发送指令

PD 控制参数:
  弱电机 (肩/肘/踝): kp=140, kd=7.5
  强电机 (髋/膝):    kp=200, kd=5.0
  腕部电机:          kp=35,  kd=6.0

关节编号 (JointIndex):
  0-5:   左腿 (hip_yaw, hip_roll, hip_pitch, knee, ankle, ankle_roll)
  6-11:  右腿
  12:    腰部
  13-19: 左臂 (shoulder_pitch/roll/yaw, elbow_pitch/roll, wrist_pitch/yaw)
  20-26: 右臂
  27-34: 未使用
```

#### H1HandController (DDS)

**文件**: `teleop/robot_control/robot_hand.py`

```
通信协议: Unitree DDS (unitree_go)
话题:
  发布: rt/inspire/cmd   (MotorCmds_)
  订阅: rt/inspire/state (MotorStates_)

控制接口:
  cmd.cmds[0:5]  → 右手 6 个电机
  cmd.cmds[6:11] → 左手 6 个电机
  每个电机: .q 设定角度

预设标签:
  "open":  全部 1.0 (张开)
  "close": 全部 0.0 (握拳)
  "half":  全部 0.5 (半握)
```

#### X100_ROS2 (ROS2)

**文件**: `teleop/replay.py` 中的 `X100_ROS2` 类

```
通信协议: ROS2 (sensor_msgs/JointState, trajectory_msgs)

订阅:
  /cr100/left_arm_state   → JointState (位置, 速度, 名称)
  /cr100/right_arm_state  → JointState

发布:
  /cr100/left_arm/online_joint_command   → JointState
  /cr100/right_arm/online_joint_command  → JointState
  /cr100/dual_arm/online_joint_command   → JointState
  /cr100/left_dexterous_hand_command     → JointState
  /cr100/right_dexterous_hand_command    → JointState
```

### 3.6 图像流服务

**文件**: `teleop/image_server/image_server.py`, `teleop/image_server/image_client.py`

```
┌──────────────┐    ZMQ PUSH/PULL    ┌──────────────┐
│ Robot Camera  │ ──────────────────→ │  Host PC     │
│ (V4L2)       │    tcp://*:5555     │ (Image Queue)│
│              │                     │              │
│ 2560×720     │  JPEG → pickle →   │ Shared Mem   │
│ 30fps SBS    │  zlib → 60KB chunks │ → AVP Render │
└──────────────┘                     └──────────────┘

图像格式:
  原始: 2560×720 (Side-by-Side 双目)
  编码: JPEG → pickle → zlib 压缩
  传输: 60KB 分块发送 (最后一块 < 60KB 作为结束标志)
  解码: zlib 解压 → unpickle → cv2.imdecode → BGR→RGB
```

### 3.7 动捕手套模块

**文件**: `teleop/mocap_sdk/mocap_glove_client.py`

```
┌────────────────┐   UDP   ┌──────────────────┐
│ VD MoCap SDK   │ ──────→ │ MocapGloveClient │
│ (动捕软件)     │ 7000    │ (Python ctypes)  │
│               │         │                  │
│ 全身23点+     │         │ left_hand_mat    │
│ 左右手各20点  │         │ right_hand_mat   │
│              │         │ (21×3 每手)      │
└────────────────┘         └──────────────────┘

T-Pose 初始骨架:
  全身: 23 关键点 (含髋、膝、踝、脊椎、头、肩、肘、腕)
  单手: 20 关键点 (腕部 + 5指 × 4关节)

手套→VR 映射 (在 VuerTeleop.step 中):
  glove_tip_indices = [4, 8, 12, 16, 20]  # 5个指尖
  DIST_OPEN  = [0.103, 0.160, 0.164, 0.153, 0.131]  # 张开时指尖距手腕距离
  DIST_CLOSE = [0.061, 0.07,  0.09,  0.09,  0.1]    # 握拳时指尖距手腕距离

  closeness = clip((DIST_OPEN - dist) / (DIST_OPEN - DIST_CLOSE), 0, 1)
  → 线性映射为 0~1 的开合度 → 乘以 1000 发送给电机
```

### 3.8 仿真环境

**文件**: `teleop/replay.py` 中的 `X100_Sim` / `X100_Sim_BraincoV2`

```
仿真器: NVIDIA Isaac Gym
物理引擎: PhysX
  dt: 1/60s
  substeps: 2
  gravity: (0, 0, -9.81)
  contact_offset: 0.002

URDF:
  X100_Sim:         k100_description/k100.urdf
  X100_Sim_BraincoV2: x100_urdf_ros1/x100_38_brainco_fixed.urdf

驱动模式: DOF_MODE_POS (位置控制)
  stiffness (kp): 40.0
  damping (kd):   4.0
  effort:         200.0

双目相机:
  分辨率: 640×480
  左眼偏移: (0, +0.033, 0)   ← 瞳距 66mm
  右眼偏移: (0, -0.033, 0)
  摄像头位置: (0.05, 0, 1.38)
  视线方向: (1, 0, 0)
```

### 3.9 延迟测量工具

**文件**: `teleop/latency/measure_arm_latency.py`

自动化的机械臂控制延迟测量工具。

```
测量原理:
  1. 发送方波关节指令 (高/低交替)
  2. 监听状态回调, 记录关节角到达目标的时间
  3. 按 10%/50%/90%/100% 进度记录延迟
  4. 统计 mean/p50/p90/p99/max

参数:
  --arm:        left/right
  --joint:      关节索引
  --amplitude:  方波幅值 (rad)
  --period:     方波周期 (s)
  --cycles:     测量周期数
  --publish-rate: 指令重发频率 (Hz)
  --velocity:   速度值

输出:
  CSV 事件日志 + JSON 统计摘要 → latency_logs/
```

---

## 4. 数据流详细分析

### 4.1 主控制回路数据流

```
循环开始 (约 30-60Hz)
    │
    ├─1─ 获取机器人当前状态
    │    ros2.get_arm_state() → (positions[14], velocities[14])
    │
    ├─2─ 获取立体图像 (如有)
    │    sm.read_image() → (480, 1280, 3) RGB
    │    → np.copyto(teleoperator.img_array, frame)
    │
    ├─3─ (手套模式) 获取手套数据
    │    glove_client.get_left_hand()  → (21, 3) float
    │    glove_client.get_right_hand() → (21, 3) float
    │
    ├─4─ 遥操步进
    │    teleoperator.step()
    │    ├── Preprocessor.process(tv) → 坐标变换
    │    ├── dex_retargeting.retarget() → 手指关节角
    │    └── 返回: head_rmat, left_pose, right_pose, left_qpos, right_qpos
    │
    ├─5─ 提取头部yaw/pitch
    │    extract_head_yaw_pitch(base_R, head_R) → (yaw, pitch)
    │
    ├─6─ IK 求解
    │    arm_ik.solve_ik(left_pose, right_pose, armstate) → sol_q[14]
    │
    ├─7─ 安全检查
    │    │ 计算 |armstate - sol_q| < POSITION_ERR_THRESH
    │    │ 且 start_teleop_event.is_set()
    │    ▼
    ├─8─ 发送指令
    │    ros2.publish_joint_command(
    │        left_arm_qpos  = sol_q[:7],
    │        right_arm_qpos = sol_q[7:],
    │        left_hand_qpos = [qpos[9]*1000, qpos[8]*1000, ...],  ← 6个电机
    │        right_hand_qpos = [qpos[9]*1000, qpos[8]*1000, ...]
    │    )
    │
    └─9─ (可选) Meshcat 可视化
         arm_ik.vis.display(q)  ← 含 head + arm + hand 关节角
```

### 4.2 图像流数据流

```
机器人摄像头 (V4L2)
    │  2560×720 SBS (Side-by-Side)
    ▼
image_server.py
    │  cv2.imencode(".jpg") → pickle → zlib
    │  分块发送 (60KB/chunk)
    ▼
ZMQ tcp://*:5555 (PUSH)
    │
    ▼
image_receiver (子进程)
    │  recv → 拼接 → zlib.decompress → unpickle
    │  cv2.imdecode → BGR→RGB
    │  sm.write_image(frame_rgb)
    ▼
SharedMemoryImage
    │  lock → np.copyto → unlock
    ▼
OpenTeleVision.main_image()
    │  读取共享内存
    │  左半 → ImageBackground (layer=1)
    │  右半 → ImageBackground (layer=2)
    ▼
Apple Vision Pro (立体显示)
```

### 4.3 关节数据格式

#### H1_2 关节 (35电机, 取 13-26 为臂部)

```
Index  Name                   臂位序
13     left_shoulder_pitch     L0
14     left_shoulder_roll      L1
15     left_shoulder_yaw       L2
16     left_elbow_pitch        L3
17     left_elbow_roll         L4
18     left_wrist_pitch        L5
19     left_wrist_yaw          L6
20     right_shoulder_pitch    R0
21     right_shoulder_roll     R1
22     right_shoulder_yaw      R2
23     right_elbow_pitch       R3
24     right_elbow_roll        R4
25     right_wrist_pitch       R5
26     right_wrist_yaw         R6
```

#### Inspire 灵巧手 (12电机)

```
Index  分配
0-5    右手 [食指, 中指, 无名指, 小指, 拇指弯, 拇指摇]
6-11   左手 [同上]
```

#### X100/BrainCo 手指电机映射

```
ROS2 发送顺序 (6值):
  [拇指摇摆×1000, 拇指弯曲×1000, 食指×1000, 中指×1000, 无名指×1000, 小指×1000]

来源映射:
  [qpos[9]×1000, qpos[8]×1000, qpos[0]×1000, qpos[2]×1000, qpos[6]×1000, qpos[4]×1000]
```

---

## 5. 坐标系与变换详解

### 5.1 坐标系定义

```
AVP (WebXR) 坐标系:          机器人 (Z-up) 坐标系:
        Y (up)                       Z (up)
        │                            │
        │                            │
        │                            │
        └───── X (right)             └───── X (forward)
       /                            /
      /                            /
     Z (forward)                  Y (left)
```

### 5.2 坐标变换矩阵

#### Y-up → Z-up 变换

```python
grd_yup2grd_zup = [
    [ 0,  0, -1,  0],   # X_robot = -Z_avp  (AVP前方 → 机器人前方)
    [-1,  0,  0,  0],   # Y_robot = -X_avp  (AVP右方 → 机器人左方)
    [ 0,  1,  0,  0],   # Z_robot = +Y_avp  (AVP上方 → 机器人上方)
    [ 0,  0,  0,  1]
]
```

#### 腕部到灵巧手对齐

```python
# X100 手臂末端 → 手指坐标系
X100_arm_2_hand = [
    [ 0,  0,  1,  0],
    [ 0,  1,  0,  0],
    [-1,  0,  0,  0],
    [ 0,  0,  0,  1]
]

# X100 手臂末端 → BrainCo 手
x100_arm_2_brainco_hand = [
    [ 0,  0,  1,  0],
    [ 0,  1,  0,  0],
    [-1,  0,  0,  0],
    [ 0,  0,  0,  1]
]

# 左臂对齐矩阵
X100_l_arm = [
    [ 0,  0, -1,  0],
    [-1,  0,  0,  0],
    [ 0,  1,  0,  0],
    [ 0,  0,  0,  1]
]

# 右臂对齐矩阵
X100_r_arm = [
    [ 0,  0, -1,  0],
    [ 1,  0,  0,  0],
    [ 0, -1,  0,  0],
    [ 0,  0,  0,  1]
]
```

### 5.3 变换链路

```
AVP 手腕矩阵 (Y-up)
    │
    ├── mat_update() ← 检查 det ≠ 0
    │
    ▼
grd_yup2grd_zup × M_wrist × inv(grd_yup2grd_zup)
    │                     ↑ 基变换
    ▼
wrist_mat (Z-up, 世界坐标系)
    │
    ├── × X100_l_arm / X100_r_arm   ← 腕部对齐
    ├── - head_mat[:3, 3]            ← 减去头部位置 → 相对位姿
    │
    ▼
rel_wrist_mat (Z-up, 相对头部)
    │
    ├── + [0, 0, 1.28]   ← 头到底座高度补偿
    ├── + [-0.1, 0, 0]   ← 前后补偿
    │
    ▼
target_pose → IK Solver
```

---

## 6. IK 优化问题详解

### 6.1 优化变量与参数

```
决策变量:
  q ∈ ℝ^14                   ← 双臂关节角 (左7 + 右7)

参数:
  Tf_l ∈ SE(3)               ← 左手末端目标位姿
  Tf_r ∈ SE(3)               ← 右手末端目标位姿
  q_last ∈ ℝ^14              ← 上一时刻关节角 (平滑基准)
```

### 6.2 代价函数

```
L(q) = w_t × ‖e_trans(q)‖² + w_r × ‖e_rot(q)‖² + w_reg × ‖q‖² + w_s × ‖q - q_last‖²

其中:
  e_trans(q) = [FK_L(q).translation - Tf_l[:3,3] ;    ← 左手位移误差
                FK_R(q).translation - Tf_r[:3,3]]     ← 右手位移误差

  e_rot(q) = [log3(FK_L(q).R × Tf_l.R^T) ;           ← 左手旋转误差 (旋转向量)
              log3(FK_R(q).R × Tf_r.R^T)]             ← 右手旋转误差

权重配置:
  X100_29_ArmIK:     w_t=50, w_r=2, w_reg=0.01, w_s=0.5
  X100_braincov2_IK: w_t=50, w_r=2, w_reg=0.01, w_s=0.05
  Arm_IK (H1_2):     使用 log6 SE(3) 整体误差, w_total=50, w_reg=0.01, w_s=10
```

### 6.3 约束条件

```
q_lower ≤ q ≤ q_upper    ← 从 URDF 读取的关节限位

(被注释掉的实验性约束):
  |q - q_last| ≤ max_delta_q   ← 单步最大关节角变化量
```

### 6.4 求解策略

```
求解器: CasADi + IPOPT (内点法)
  max_iter: 50
  tol: 1e-4 ~ 1e-6
  print_level: 0 (静默)

异常处理:
  1. 正常求解 → 返回 sol_q
  2. 求解失败 → opti.debug.value(var_q)
     → 加权移动滤波 (权重 [0.4, 0.3, 0.2, 0.1])
     → 返回滤波后的 sol_q 或直接返回 current_q

臂长缩放:
  scale_factor = robot_arm_length / human_arm_length
  默认: human=0.51m, robot=0.52m (X100)
        human=0.60m, robot=0.75m (H1_2)
```

---

## 7. 通信协议与接口

### 7.1 Unitree DDS 接口

```
协议: Unitree DDS Wrapper
消息类型:
  unitree_hg.msg.dds_.LowCmd_   → 35电机位置/速度/力矩/PD参数
  unitree_hg.msg.dds_.LowState_ → 35电机状态 + IMU

话题:
  rt/lowcmd   (Pub)   ← 控制指令
  rt/lowstate (Sub)   ← 状态反馈

CRC 校验: 自定义 CRC32
指令周期: 2ms
```

### 7.2 ROS2 话题接口

```
QoS 配置: VOLATILE + RELIABLE, depth=10

订阅话题:
  /cr100/left_arm_state         ← JointState (7关节位置+速度)
  /cr100/right_arm_state        ← JointState

发布话题:
  /cr100/left_arm/online_joint_command    ← JointState (7关节)
  /cr100/right_arm/online_joint_command   ← JointState
  /cr100/dual_arm/online_joint_command    ← JointState (14关节)
  /cr100/left_dexterous_hand_command      ← JointState (6关节)
  /cr100/right_dexterous_hand_command     ← JointState (6关节)

JointState 格式:
  header.stamp:  当前时间戳
  name:          关节名称列表
  position:      关节位置 (float64[])
  velocity:      速度 [单个值, 所有关节共享]
```

### 7.3 ZMQ 图像流协议

```
模式: PUSH/PULL (单向流)
端口: tcp://*:5555 (可配置为 5559)
HWM:  1 (高水位标记, 丢弃旧帧)

帧格式:
  原始帧 → cv2.imencode(".jpg")
         → pickle.dumps()
         → zlib.compress()
         → 60KB 分块发送

接收端判断:
  if len(chunk) < 60000: break  ← 最后一块表示帧结束
```

### 7.4 Socket 通信协议

```
类型: TCP Socket (JSON 行协议)
端口: 5556

客户端→服务端 (控制指令):
  {"arm_positions": [float × 14],
   "left_hand_positions": [float × 6],
   "right_hand_positions": [float × 6]}\n

服务端→客户端 (状态反馈):
  {"left_arm": {"positions": [...], ...},
   "right_arm": {"positions": [...], ...}}\n

发送间隔: 100ms
接收超时: 1s
重连策略: 5s 定时重连
```

---

## 8. 运行模式

### 8.1 硬件模式 (H1_2)

**入口**: `teleop/teleop_hand_and_arm.py`

```
前置条件:
  1. Unitree H1_2 机器人开机, DDS 网络可达
  2. Inspire 灵巧手服务已启动
  3. 机器人端 image_server.py 运行中
  4. cert.pem + key.pem 已生成

启动命令:
  cd teleop && python teleop_hand_and_arm.py

运行流程:
  1. VuerTeleop("inspire_hand.yml") → WebXR 服务器启动
  2. H1HandController() → DDS 灵巧手连接
  3. H1ArmController()  → DDS 臂部连接 + 锁腿
  4. Arm_IK()           → IK 求解器初始化
  5. image_receiver 子进程启动
  6. 用户输入 's' 开始
  7. 主循环: 读状态 → VR step → IK → 发送指令
```

### 8.2 仿真模式 (Isaac Gym + ROS2)

**入口**: `teleop/teleop_hand_and_arm_sim.py`

```
前置条件:
  1. ROS2 环境已 source
  2. 机器人端 ROS2 节点运行中 (或仅使用仿真)
  3. Meshcat 可视化 (可选)

启动命令:
  cd teleop && python teleop_hand_and_arm_sim.py

与硬件模式差异:
  - 使用 VuerPreprocessorLegacy (不同坐标偏移)
  - 使用 X100_29_ArmIK (K100 URDF)
  - 通过 ROS2 话题通信
  - 支持 Meshcat 3D 可视化
  - 位置误差阈值安全检查
  - 支持单臂模式切换 (is_left_mode)
```

### 8.3 数据手套模式

**入口**: `teleop/teleop_hand_and_arm_sim_glove.py`

```
与仿真模式差异:
  - USE_DATA_GLOVE = True → 启用动捕手套
  - MocapGloveClient 初始化 (UDP 192.168.1.100:7000)
  - VuerTeleop.step() 接收 glove_left_mat / glove_right_mat
  - 手指使用距离→开合度线性映射 (替代 dex_retargeting)

手套关节映射:
  指尖距离 → closeness (0~1)
  closeness → qpos 数组:
    qpos[0] = closeness[1]  # 食指
    qpos[2] = closeness[2]  # 中指
    qpos[4] = closeness[4]  # 小指
    qpos[6] = closeness[3]  # 无名指
    qpos[8] = closeness[0]  # 拇指弯曲
    qpos[9] = closeness[0]  # 拇指摇摆
```

---

## 9. 机器人模型与 URDF

### URDF 文件索引

| 机型 | URDF 路径 | 用途 |
|------|----------|------|
| H1_2 人形 | `assets/h1_2/h1_2.urdf` | Arm_IK (H1_2 硬件) |
| K100 双臂 | `assets/k100_description/k100_brainco.urdf` | X100_29_ArmIK |
| X100 双臂 | `assets/x100_urdf_ros1/x100_38_brainco.urdf` | X100_braincov2_ArmIK |
| X100 固定底座 | `assets/x100_urdf_ros1/x100_38_brainco_fixed.urdf` | Isaac Gym 仿真 |
| Inspire 左手 | `assets/inspire_hand/inspire_hand_left.urdf` | 手势重定向 |
| Inspire 右手 | `assets/inspire_hand/inspire_hand_right.urdf` | 手势重定向 |
| X100 左手 | `assets/x100_hand/x100_hand_left.urdf` | 手势重定向 |
| X100 右手 | `assets/x100_hand/x100_hand_right.urdf` | 手势重定向 |
| BrainCo 左手 | `assets/BrainCo-Hand-Revo2-URDF-V2/.../brainco-lefthand-URDF-V2.urdf` | 手势重定向 |
| BrainCo 右手 | `assets/BrainCo-Hand-Revo2-URDF-V2/.../brainco-righthand-URDF-V2.urdf` | 手势重定向 |

### IK 中锁定的关节

IK 求解时大量关节被锁定，仅保留臂部 14 个活动关节：

```
X100_29_ArmIK 锁定列表:
  - head_yaw_joint, head_pitch_joint
  - 左手: thumb_metacarpal/proximal/distal, index/middle/ring/pinky proximal/distal (11 个)
  - 右手: 同上 (11 个)
  总锁定: 24 个 → 保留 14 个臂部关节
```

---

## 10. 手势重定向配置

### 重定向类型对比

| 配置文件 | 手型 | URDF | scaling_factor | 关节数 |
|---------|------|------|----------------|--------|
| `inspire_hand.yml` | Inspire | inspire_hand | 1.5 | 12 |
| `x100_hand.yml` | X100 自有 | x100_hand | 0.95 | 11 |
| `x100_hand_braincoV2.yml` | BrainCo V2 | BrainCo-Hand-Revo2 | 0.9 | 11 |

### 目标关节映射

```
Inspire Hand (12关节):
  L_index_proximal, L_index_intermediate,
  L_middle_proximal, L_middle_intermediate,
  L_pinky_proximal, L_pinky_intermediate,
  L_ring_proximal, L_ring_intermediate,
  L_thumb_proximal_yaw, L_thumb_proximal_pitch,
  L_thumb_intermediate, L_thumb_distal

X100/BrainCo (11关节):
  (同上, 但无 L_thumb_distal)
```

---

## 11. 性能与延迟分析

### 各环节延迟估计

| 环节 | 典型延迟 | 说明 |
|------|---------|------|
| AVP 手部追踪 | ~8ms | WebXR 原生 |
| Vuer 网络传输 | ~5-15ms | WebSocket, 取决于网络 |
| 坐标预处理 | <1ms | 纯矩阵运算 |
| dex_retargeting | ~2-5ms | 向量优化 |
| IK 求解 (CasADi) | ~5-20ms | IPOPT, max_iter=50 |
| ROS2 话题发布 | ~1-2ms | 本地 |
| 网络→机器人 | ~2-10ms | 取决于网络拓扑 |
| 电机响应 | ~10-50ms | 取决于 PD 参数和负载 |
| **端到端** | **~30-100ms** | 手到机器人动作 |

### 图像流延迟

| 环节 | 典型延迟 |
|------|---------|
| 摄像头采集 | ~33ms (30fps) |
| JPEG 编码 + 压缩 | ~5-10ms |
| ZMQ 网络传输 | ~5-15ms |
| 解压 + 解码 | ~5-10ms |
| Vuer 渲染 | ~16ms (60fps) |
| **图像端到端** | **~70-100ms** |

---

## 12. 安全机制

### 位置误差阈值

```python
POSITION_ERR_THRESH = 1.0  # (弧度, 仿真模式中可配为 150)

# 仅当所有关节误差 < 阈值时才发送指令
if np.all(left_err < THRESH) and np.all(right_err < THRESH):
    ros2.publish_joint_command(...)
```

### 腿部锁定

```python
# H1ArmController 初始化时:
# 1000步平滑过渡锁定腿部关节
for i in range(1000):
    q_t = init_q + (target_q - init_q) * i / 1000
    for id in JointIndex:
        if id not in JointArmIndex:  # 非臂部 = 腿部
            msg.motor_cmd[id].kp = 200
            msg.motor_cmd[id].kd = 5
            msg.motor_cmd[id].q = q_t[i]
```

### 手臂复位

```python
# 输入 'r' 复位手臂到安全位置
if reset_hand.is_set():
    if is_left_mode:
        nocollision_sol_q[:7] = [-0.026, 1.48, -1.63, 0.1, 1.60, -1.52, 0.12]
    else:
        nocollision_sol_q[7:] = [-0.23, 1.5, 1.5, 0.43, -1.55, -1.48, 0.02]
```

### IK 失败回退

```python
# IK 求解失败时:
# 1. 使用 debug value + 加权滤波
# 2. 或直接返回当前关节角度 (不动)
except Exception:
    sol_q = self.smooth_filter.filtered_data
    return current_lr_arm_motor_q  # 保持不动
```

---

## 13. 文件索引

```
avp_teleop_release/
│
├── README.md                              ← 快速入门指南
├── requirements.txt                       ← Python 依赖
├── environment.yml                        ← Conda 完整环境
│
├── config/                                ← 手势重定向配置
│   ├── inspire_hand.yml                   ← Inspire 灵巧手
│   ├── x100_hand.yml                      ← X100 自有手
│   └── x100_hand_braincoV2.yml            ← BrainCo V2 手
│
├── teleop/                                ← 核心遥操代码
│   ├── TeleVision.py                      ← WebXR 服务器 (OpenTeleVision)
│   ├── Preprocessor.py                    ← 坐标预处理
│   ├── constants_vuer.py                  ← 坐标变换常量
│   ├── motion_utils.py                    ← 矩阵工具函数
│   ├── replay.py                          ← 仿真/ROS2/Socket 控制
│   │
│   ├── teleop_hand_and_arm.py             ← [入口] 硬件模式 (H1_2)
│   ├── teleop_hand_and_arm_sim.py         ← [入口] 仿真模式 (ROS2)
│   ├── teleop_hand_and_arm_sim_glove.py   ← [入口] 手套模式
│   │
│   ├── robot_control/                     ← 机器人控制接口
│   │   ├── robot_arm.py                   ← H1 臂部控制 (DDS)
│   │   ├── robot_hand.py                  ← Inspire 手控制 (DDS)
│   │   └── robot_arm_ik.py                ← IK 求解器 (Pinocchio + CasADi)
│   │
│   ├── image_server/                      ← 图像流
│   │   ├── image_server.py                ← 机器人端 ZMQ 推流
│   │   └── image_client.py                ← 测试客户端
│   │
│   ├── webrtc/                            ← WebRTC 路径
│   │   └── zed_server.py                  ← ZED 相机 WebRTC 服务
│   │
│   ├── mocap_sdk/                         ← 动捕手套
│   │   └── mocap_glove_client.py          ← MocapGloveClient
│   │
│   └── latency/                           ← 延迟测量
│       └── measure_arm_latency.py         ← 臂部延迟基准测试
│
└── docs/                                  ← 文档
    ├── TECHNICAL_DOC.md                   ← 本技术文档
    └── diagrams/                          ← 图表资源目录
```

---

*文档结束 — AVP Teleoperation 技术文档 v1.0*
