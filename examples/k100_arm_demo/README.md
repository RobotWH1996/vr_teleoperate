# K100 手臂控制示例

通过 PallasSDK + ROS2 对 K100 双臂发送关节轨迹的参考脚本。

## 依赖

- [PallasSDK](https://github.com/)（需单独安装，本仓库不包含）
- ROS2（`rclpy`、`control_msgs`、`trajectory_msgs` 等）

## 文件

| 文件 | 说明 |
|------|------|
| `test_node_python.py` | 双臂轨迹下发与关节状态读取示例 |
| `x100_say_hello.json` | 示例轨迹数据 |

## 运行

```bash
# 先 source ROS2 与机器人工作空间
python test_node_python.py
```
