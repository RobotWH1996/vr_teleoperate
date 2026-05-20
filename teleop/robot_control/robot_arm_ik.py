import casadi
import meshcat.geometry as mg
import numpy as np

import time

import os
import sys
import pinocchio as pin

# 手动设置缺失的变量
sys.modules["pinocchio"].WITH_HPP_FCL = True
sys.modules["pinocchio"].WITH_HPP_FCL_BINDINGS = True

from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper

from pinocchio.visualize import MeshcatVisualizer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class WeightedMovingFilter:
    def __init__(self, weights, dim):
        """
        初始化加权移动滤波器

        参数:
        - weights: 权重数组，例如 [0.4, 0.3, 0.2, 0.1]
        - dim: 数据维度，比如关节数量
        """
        self.weights = np.array(weights)
        self.weights /= np.sum(self.weights)  # 归一化权重
        self.dim = dim
        self.history = []

    def add_data(self, new_data):
        """
        添加新数据到历史记录

        参数:
        - new_data: 新的一组数据 (np.ndarray)
        """
        if len(self.history) >= len(self.weights):
            self.history.pop(0)
        self.history.append(new_data.copy())

    @property
    def filtered_data(self):
        """
        获取加权平均后的数据

        返回:
        - filtered_data: 加权移动平均后的结果
        """
        if not self.history:
            return np.zeros(self.dim)

        weighted_sum = np.zeros(self.dim)
        for i, data in enumerate(self.history):
            weighted_sum += self.weights[i] * data
        return weighted_sum


class X100_braincov2_ArmIK:
    def __init__(self, Unit_Test=False, Visualization=False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test  # 单元测试
        self.Visualization = Visualization  # 是否可视化

        if not self.Unit_Test:
            self.robot = pin.RobotWrapper.BuildFromURDF(
                "./assets/x100_urdf_ros1/x100_38_brainco.urdf",
                "./assets/x100_urdf_ros1/",
            )
        else:
            self.robot = pin.RobotWrapper.BuildFromURDF(
                "./assets/x100_urdf_ros1/x100_38_brainco.urdf",
                "./assets/x100_urdf_ros1/",
            )  # for test

        self.mixed_jointsToLockIDs = [
            "AGV_yaw_left_joint",
            "AGV_move_left_joint",
            "AGV_yaw_right_joint",
            "AGV_move_right_joint",
            "AGV_yaw_behind_joint",
            "AGV_move_behind_joint",
            "waist_pitch_lower_joint",
            "waist_pitch_middle_joint",
            "waist_pitch_upper_joint",
            "torso_joint",
            "head_yaw_joint",
            "head_pitch_joint",
            "L_thumb_metacarpal_joint",
            "L_thumb_proximal_joint",
            "L_thumb_distal_joint",
            "L_index_proximal_joint",
            "L_index_distal_joint",
            "L_middle_proximal_joint",
            "L_middle_distal_joint",
            "L_ring_proximal_joint",
            "L_ring_distal_joint",
            "L_pinky_proximal_joint",
            "L_pinky_distal_joint",
            "R_thumb_metacarpal_joint",
            "R_thumb_proximal_joint",
            "R_thumb_distal_joint",
            "R_index_proximal_joint",
            "R_index_distal_joint",
            "R_middle_proximal_joint",
            "R_middle_distal_joint",
            "R_ring_proximal_joint",
            "R_ring_distal_joint",
            "R_pinky_proximal_joint",
            "R_pinky_distal_joint",
        ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        # 左右手两个末端位置
        self.reduced_robot.model.addFrame(
            pin.Frame(
                "L_ee",
                self.reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0, 0, -0.08]).T),
                pin.FrameType.OP_FRAME,
            )
        )

        self.reduced_robot.model.addFrame(
            pin.Frame(
                "R_ee",
                self.reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0, 0, -0.08]).T),
                pin.FrameType.OP_FRAME,
            )
        )

        # for idx, name in enumerate(self.reduced_robot.model.names):
        #     print(f"{idx}: {name}")
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables 创建符号变量
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)  # 关节角度
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)  # 左手末端执行器的目标位姿变换矩阵
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)  # 右手末端执行器的目标位姿变换矩阵
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")  # 获取索引
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")  # 获取索引

        # 平移误差函数
        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
                )
            ],
        )
        # 旋转误差函数
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(
                        self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T
                    ),
                    cpin.log3(
                        self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T
                    ),
                )
            ],
        )

        # Defining the optimization problem
        # 定义优化问题
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)  # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.rotation_cost = casadi.sumsqr(
            self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        # 设置优化条件和目标
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.minimize(
            50 * self.translational_cost
            + 2 * self.rotation_cost
            + 0.01 * self.regularization_cost
            + 0.05 * self.smooth_cost
        )

        opts = {
            "ipopt": {"print_level": 0, "max_iter": 50, "tol": 1e-6},
            "print_time": False,  # print or not
            "calc_lam_p": False,  # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        # 移动平均滤波函数
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None

        if self.Visualization:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(
                self.reduced_robot.model,
                self.reduced_robot.collision_model,
                self.reduced_robot.visual_model,
            )
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel("pinocchio")
            self.vis.displayFrames(
                True, frame_ids=[107, 108], axis_length=0.15, axis_width=5
            )
            self.vis.display(pin.neutral(self.reduced_robot.model))

            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = ["L_ee_target", "R_ee_target"]
            FRAME_AXIS_POSITIONS = (
                np.array(
                    [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
                )
                .astype(np.float32)
                .T
            )
            FRAME_AXIS_COLORS = (
                np.array(
                    [
                        [1, 0, 0],
                        [1, 0.6, 0],
                        [0, 1, 0],
                        [0.6, 1, 0],
                        [0, 0, 1],
                        [0, 0.6, 1],
                    ]
                )
                .astype(np.float32)
                .T
            )
            axis_length = 0.1
            axis_width = 20
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )

    # If the robot arm is not the same size as your arm :)
    def scale_arms(
        self,
        human_left_pose,
        human_right_pose,
        human_arm_length=0.51,
        robot_arm_length=0.52,
    ):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def solve_ik(
        self,
        left_wrist,
        right_wrist,
        current_lr_arm_motor_q=None,
        current_lr_arm_motor_dq=None,
    ):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        left_wrist, right_wrist = self.scale_arms(left_wrist, right_wrist)
        if self.Visualization:
            self.vis.viewer["L_ee_target"].set_transform(
                left_wrist
            )  # for visualization
            self.vis.viewer["R_ee_target"].set_transform(
                right_wrist
            )  # for visualization

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)  # for smooth

        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            # self.smooth_filter.add_data(sol_q)
            # sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            # sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            # sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            print(
                f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_wrist} \nright_pose: \n{right_wrist}"
            )
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q

#策略1增加平滑的权重ok  2增加 迭代的初始的 关节角ok   3限制其目标点位置与姿态的距离？？限制多少比较合适？待定  4限制范围
class X100_29_ArmIK:
    def __init__(self, Unit_Test=False, Visualization=False, Debug=True):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test  # 单元测试
        self.Visualization = Visualization  # 是否可视化
        self.Debug = Debug

        if not self.Unit_Test:
            self.robot = pin.RobotWrapper.BuildFromURDF(
                "./assets/k100_description/k100_brainco.urdf", "./assets/k100_description/"
            )
        else:
            self.robot = pin.RobotWrapper.BuildFromURDF(
                "./assets/k100_description/k100_brainco.urdf", "./assets/k100_description/"
            )  # for test

        self.mixed_jointsToLockIDs = [
            # "world_to_base",
            "head_yaw_joint",
            "head_pitch_joint",
            # "w_body_wheel1_joint",
            # "w_body_wheel2_joint",
            # "w_body_wheel3_joint",
            # "w_body_wheel3_1_joint",
            # "w_body_wheel4_joint",
            # "w_body_wheel4_1_joint",
            # "w_body_wheel5_joint",
            # "w_body_wheel5_1_joint",
            # "L_base_link_joint",
            "left_thumb_metacarpal_joint",
            "left_thumb_proximal_joint",
            "left_thumb_distal_joint",
            # "left_thumb_tip_joint",
            "left_index_proximal_joint",
            "left_index_distal_joint",
            # "left_index_tip_joint",
            "left_middle_proximal_joint",
            "left_middle_distal_joint",
            # "left_middle_tip_joint",
            "left_ring_proximal_joint",
            "left_ring_distal_joint",
            # "left_ring_tip_joint",
            "left_pinky_proximal_joint",
            "left_pinky_distal_joint",
            # "left_pinky_tip_joint",

            # "R_base_link_joint",
            "right_thumb_metacarpal_joint",
            "right_thumb_proximal_joint",
            "right_thumb_distal_joint",
            # "right_thumb_tip_joint",
            "right_index_proximal_joint",
            "right_index_distal_joint",
            # "right_index_tip_joint",
            "right_middle_proximal_joint",
            "right_middle_distal_joint",
            # "right_middle_tip_joint",
            "right_ring_proximal_joint",
            "right_ring_distal_joint",
            # "right_ring_tip_joint",
            "right_pinky_proximal_joint",
            "right_pinky_distal_joint",
            # "right_pinky_tip_joint",
        ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        # 左右手两个末端位置
        self.reduced_robot.model.addFrame(
            pin.Frame(
                "L_ee",
                self.reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0, 0, -0.08]).T),
                pin.FrameType.OP_FRAME,
            )
        )

        self.reduced_robot.model.addFrame(
            pin.Frame(
                "R_ee",
                self.reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0, 0, -0.08]).T),
                pin.FrameType.OP_FRAME,
            )
        )

        # for idx, name in enumerate(self.reduced_robot.model.names):
        #     print(f"{idx}: {name}")
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables 创建符号变量
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)  # 关节角度
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)  # 左手末端执行器的目标位姿变换矩阵
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)  # 右手末端执行器的目标位姿变换矩阵
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")  # 获取索引
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")  # 获取索引

        # 平移误差函数
        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
                )
            ],
        )
        # 旋转误差函数
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(
                        self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T
                    ),
                    cpin.log3(
                        self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T
                    ),
                )
            ],
        )
        self.left_translational_error = casadi.Function(
            "x100_29_left_translational_error",
            [self.cq, self.cTf_l],
            [self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3]],
        )
        self.left_rotational_error = casadi.Function(
            "x100_29_left_rotational_error",
            [self.cq, self.cTf_l],
            [
                cpin.log3(
                    self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T
                )
            ],
        )

        # Defining the optimization problem
        # 定义优化问题
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)  # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.rotation_cost = casadi.sumsqr(
            self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        # 设置优化条件和目标
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        # q_min = self.reduced_robot.model.lowerPositionLimit
        # q_max = self.reduced_robot.model.upperPositionLimit
        # # 上下限 限制
        # safe_lower_bound = casadi.fmax(self.var_q_last - max_delta_q, q_min)
        # safe_upper_bound = casadi.fmin(self.var_q_last + max_delta_q, q_max)
        # self.opti.subject_to(
        #     self.opti.bounded(
        #         self.safe_lower_bound,
        #         self.var_q,
        #         self.safe_upper_bound,
        #     )
        # )
        # self.opti.minimize(
        #     50 * self.translational_cost
        #     + 2 * self.rotation_cost
        #     + 0.01 * self.regularization_cost
        #     + 0.05 * self.smooth_cost
        # )
        self.opti.minimize(
            50 * self.translational_cost
            + 2 * self.rotation_cost
            + 0.01 * self.regularization_cost
            + 0.1 * self.smooth_cost
        )
        opts = {
            # "ipopt": {"print_level": 0, "max_iter": 50, "tol": 1e-6},
            "ipopt": {"print_level": 0, "max_iter": 50, "tol": 1e-4},########zm
            "print_time": False,  # print or not
            "calc_lam_p": False,  # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.left_opti = casadi.Opti()
        self.left_var_q = self.left_opti.variable(7)
        self.left_var_q_last = self.left_opti.parameter(7)
        self.left_param_right_q = self.left_opti.parameter(7)
        self.left_param_tf = self.left_opti.parameter(4, 4)
        self.left_full_q = casadi.vertcat(self.left_var_q, self.left_param_right_q)
        self.left_translational_cost = casadi.sumsqr(
            self.left_translational_error(self.left_full_q, self.left_param_tf)
        )
        self.left_rotation_cost = casadi.sumsqr(
            self.left_rotational_error(self.left_full_q, self.left_param_tf)
        )
        self.left_regularization_cost = casadi.sumsqr(self.left_var_q)
        self.left_smooth_cost = casadi.sumsqr(self.left_var_q - self.left_var_q_last)
        self.left_opti.subject_to(
            self.left_opti.bounded(
                self.reduced_robot.model.lowerPositionLimit[:7],
                self.left_var_q,
                self.reduced_robot.model.upperPositionLimit[:7],
            )
        )
        self.left_opti.minimize(
            50 * self.left_translational_cost
            + 2 * self.left_rotation_cost
            + 0.01 * self.left_regularization_cost
            + 0.1 * self.left_smooth_cost
        )
        self.left_opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)#
        self.left_init_q = np.zeros(7, dtype=np.float64)
        # 移动平均滤波函数
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None
        
        ### [新增] 用于记录上一周期的平滑目标位姿 (4x4 矩阵)
        self.last_target_left = None
        self.last_target_right = None
        self.last_good_sol_q = None
        self.ik_failure_count = 0
        self.last_ik_failure_log_t = 0.0
        self.left_last_good_sol_q = None
        self.left_ik_failure_count = 0
        self.left_last_ik_failure_log_t = 0.0
        self.left_last_solve_success = None
        self.left_last_fallback_source = None

        if self.Visualization:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(
                self.robot.model,
                self.robot.collision_model,
                self.robot.visual_model,
            )
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel("pinocchio")
            self.vis.displayFrames(
                True, frame_ids=[107, 108], axis_length=0.15, axis_width=5
            )
            self.vis.display(pin.neutral(self.robot.model))
            # print("pin.neutral(self.robot.model)",pin.neutral(self.robot.model))
            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = ["L_ee_target", "R_ee_target"]
            FRAME_AXIS_POSITIONS = (
                np.array(
                    [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
                )
                .astype(np.float32)
                .T
            )
            FRAME_AXIS_COLORS = (
                np.array(
                    [
                        [1, 0, 0],
                        [1, 0.6, 0],
                        [0, 1, 0],
                        [0.6, 1, 0],
                        [0, 0, 1],
                        [0, 0.6, 1],
                    ]
                )
                .astype(np.float32)
                .T
            )
            axis_length = 0.1
            axis_width = 20
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )
        # Load model
        # self.collision_model = pin.buildModelFromUrdf(
        #     "./assets/k100_description_braincoV2/k100.urdf"
        # )

        # # Load collision geometries
        # self.collision_geom_model = pin.buildGeomFromUrdf(
        #     self.collision_model,
        #     "./assets/k100_description_braincoV2/k100.urdf",
        #     pin.GeometryType.COLLISION,
        #     "./assets/k100_description_braincoV2/",
        # )

        # # Add collisition pairs
        # self.collision_geom_model.addAllCollisionPairs()
        # # print("num collision pairs - initial:", len(geom_model.collisionPairs))

        # # Remove collision pairs listed in the SRDF file
        # srdf_model_path = "./assets/k100_description_braincoV2/k100.srdf"

        # pin.removeCollisionPairs(
        #     self.collision_model, self.collision_geom_model, srdf_model_path
        # )
        # print(
        #     "num collision pairs - after removing useless collision pairs:",
        #     len(self.collision_geom_model.collisionPairs),
        # )

        # # Load reference configuration
        # pin.loadReferenceConfigurations(self.collision_model, srdf_model_path)
        # self.left_arm_joints = [
        #     "left_shoulder_pitch_joint",
        #     "left_shoulder_roll_joint",
        #     "left_shoulder_yaw_joint",
        #     "left_elbow_joint",
        #     "left_wrist_roll_joint",
        #     "left_wrist_pitch_joint",
        #     "left_wrist_yaw_joint",
        # ]

        # self.right_arm_joints = [
        #     "right_shoulder_pitch_joint",
        #     "right_shoulder_roll_joint",
        #     "right_shoulder_yaw_joint",
        #     "right_elbow_joint",
        #     "right_wrist_roll_joint",
        #     "right_wrist_pitch_joint",
        #     "right_wrist_yaw_joint",
        # ]
    #限制每个频率  目标点矩阵变化的补偿  需要测试
    # 限制每个频率 目标点矩阵变化的补偿
    def limit_pose(self, current_pose_mat, target_pose_mat, max_dist=0.06, max_angle=0.08):
        """
        限制位姿增量
        :param current_pose_mat: 上一帧的 4x4 位姿矩阵 (None 代表第一帧)
        :param target_pose_mat: 原始目标的 4x4 位姿矩阵
        :param max_dist: 最大允许位移 (米)
        :param max_angle: 最大允许旋转 (弧度)
        """
        if current_pose_mat is None:
            return target_pose_mat.copy()

        # 将 4x4 Numpy矩阵转换为 Pinocchio SE3 对象
        curr_se3 = pin.SE3(current_pose_mat[:3, :3], current_pose_mat[:3, 3])
        targ_se3 = pin.SE3(target_pose_mat[:3, :3], target_pose_mat[:3, 3])

        # --- 1. 位置限制 ---
        curr_p = curr_se3.translation
        targ_p = targ_se3.translation
        delta_p = targ_p - curr_p
        dist = np.linalg.norm(delta_p)

        if dist > max_dist:
            # 限制位移，沿方向矢量截断
            targ_p = curr_p + (delta_p / dist) * max_dist

        # --- 2. 姿态限制 ---
        curr_R = curr_se3.rotation
        targ_R = targ_se3.rotation
        
        # 计算相对旋转矩阵 R_rel: curr_R * R_rel = targ_R => R_rel = curr_R^T * targ_R
        relative_R = curr_R.T @ targ_R
        # 转换为旋转向量 (角轴表示)
        exp_v = pin.log3(relative_R) 
        angle = np.linalg.norm(exp_v)

        if angle > max_angle:
            # 限制旋转角度
            exp_v = (exp_v / angle) * max_angle
            targ_R = curr_R @ pin.exp3(exp_v)

        # 返回 4x4 齐次变换矩阵
        return pin.SE3(targ_R, targ_p).homogeneous
    # 获取关节索引
    def get_joint_indices(self, model, joint_names):
        """获取关节名称对应的索引列表"""
        indices = []
        for name in joint_names:
            if model.existJointName(name):
                joint_id = model.getJointId(name)
                joint = model.joints[joint_id]
                # 关节在配置向量中的起始索引
                start_idx = joint.idx_q
                # 关节的自由度数 - 兼容不同版本的 pinocchio
                if hasattr(joint, "nq") and callable(joint.nq):
                    nq = joint.nq()
                elif hasattr(joint, "nq"):
                    nq = joint.nq
                else:
                    nq = 1  # 默认为1个自由度
                # 添加索引范围
                indices.extend(range(start_idx, start_idx + nq))
            else:
                print(f"警告: 关节 '{name}' 不存在于模型中")
        return indices

    def set_joint_values(self, q, indices, values):
        """为指定关节设置值"""
        if len(indices) != len(values):
            raise ValueError("索引和值的长度不匹配")
        for idx, val in zip(indices, values):
            q[idx] = val

    def compute_collision(self, q):
        left_arm_indices = self.get_joint_indices(
            self.collision_model, self.left_arm_joints
        )
        right_arm_indices = self.get_joint_indices(
            self.collision_model, self.right_arm_joints
        )
        qneu = pin.neutral(self.collision_model)
        self.set_joint_values(qneu, left_arm_indices, q[:7])
        self.set_joint_values(qneu, right_arm_indices, q[7:])
        data = self.collision_model.createData()
        geom_data = pin.GeometryData(self.collision_geom_model)

        # Compute all the collisions
        pin.computeCollisions(
            self.collision_model,
            data,
            self.collision_geom_model,
            geom_data,
            qneu,
            False,
        )

        # Print the status of collision for all collision pairs
        for k in range(len(self.collision_geom_model.collisionPairs)):
            cr = geom_data.collisionResults[k]
            cp = self.collision_geom_model.collisionPairs[k]
            if cr.isCollision():
                print("collison change joint!!!")
                return True  # 有碰撞，返回高代价
        return False  # 无碰撞，返回低代价

    # If the robot arm is not the same size as your arm :)
    def scale_arms(
        self,
        human_left_pose,
        human_right_pose,
        human_arm_length=0.51,
        robot_arm_length=0.52,
    ):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def get_ee_poses(self, q14):
        """Return current left/right end-effector poses as 4x4 matrices."""
        q14 = np.asarray(q14, dtype=np.float64).reshape(-1)
        if q14.shape[0] != self.reduced_robot.model.nq:
            raise ValueError(
                f"Expected {self.reduced_robot.model.nq} arm joints, got {q14.shape[0]}"
            )

        data = self.reduced_robot.model.createData()
        pin.forwardKinematics(self.reduced_robot.model, data, q14)
        pin.updateFramePlacements(self.reduced_robot.model, data)

        left_pose = data.oMf[self.L_hand_id].homogeneous.copy()
        right_pose = data.oMf[self.R_hand_id].homogeneous.copy()
        return left_pose, right_pose

    def solve_left_ik(
        self,
        left_wrist_robot_pose,
        current_lr_arm_motor_q=None,
        current_lr_arm_motor_dq=None,
        fallback_q=None,
    ):
        def _valid_q14(q):
            if q is None:
                return None
            q = np.asarray(q, dtype=np.float64).reshape(-1)
            if q.shape[0] != self.reduced_robot.model.nq:
                return None
            if not np.all(np.isfinite(q)):
                return None
            return q.copy()

        def _valid_q7(q):
            if q is None:
                return None
            q = np.asarray(q, dtype=np.float64).reshape(-1)
            if q.shape[0] != 7:
                return None
            if not np.all(np.isfinite(q)):
                return None
            return q.copy()

        target_pose = np.asarray(left_wrist_robot_pose, dtype=np.float64)
        if target_pose.shape != (4, 4) or not np.all(np.isfinite(target_pose)):
            raise ValueError("left_wrist_robot_pose must be a finite 4x4 matrix")

        current_q = _valid_q14(current_lr_arm_motor_q)
        fallback_full_q = _valid_q14(fallback_q)
        last_good_q = _valid_q14(self.left_last_good_sol_q)

        seed_left_q = current_q[:7] if current_q is not None else None
        if seed_left_q is None:
            seed_left_q = _valid_q7(self.left_init_q)
        if seed_left_q is None and last_good_q is not None:
            seed_left_q = last_good_q[:7].copy()
        if seed_left_q is None and fallback_full_q is not None:
            seed_left_q = fallback_full_q[:7].copy()
        if seed_left_q is None:
            seed_left_q = np.zeros(7, dtype=np.float64)

        right_hold_q = current_q[7:] if current_q is not None else None
        if right_hold_q is None and fallback_full_q is not None:
            right_hold_q = fallback_full_q[7:].copy()
        if right_hold_q is None and last_good_q is not None:
            right_hold_q = last_good_q[7:].copy()
        if right_hold_q is None:
            right_hold_q = np.zeros(7, dtype=np.float64)

        self.left_last_solve_success = False
        self.left_last_fallback_source = None

        self.left_opti.set_initial(self.left_var_q, seed_left_q)
        self.left_opti.set_value(self.left_var_q_last, seed_left_q)
        self.left_opti.set_value(self.left_param_right_q, right_hold_q)
        self.left_opti.set_value(self.left_param_tf, target_pose)

        if self.Visualization:
            self.vis.viewer["L_ee_target"].set_transform(target_pose)

        try:
            self.left_opti.solve()
            left_sol_q = np.asarray(
                self.left_opti.value(self.left_var_q), dtype=np.float64
            ).reshape(-1)
            if left_sol_q.shape[0] != 7:
                raise ValueError(
                    f"Left IK returned {left_sol_q.shape[0]} joints, expected 7"
                )
            if not np.all(np.isfinite(left_sol_q)):
                raise ValueError("Left IK returned non-finite joint values")

            if current_lr_arm_motor_dq is not None:
                _ = np.asarray(current_lr_arm_motor_dq, dtype=np.float64)[:7] * 0.0

            sol_q = np.concatenate((left_sol_q, right_hold_q)).astype(np.float64)
            self.left_init_q = left_sol_q.copy()
            self.left_last_good_sol_q = sol_q.copy()
            self.left_last_solve_success = True
            self.left_last_fallback_source = "success"
            return sol_q

        except Exception as e:
            self.left_ik_failure_count += 1
            fallback_source = "seed_q"
            fallback_sol_q = np.concatenate((seed_left_q, right_hold_q)).astype(np.float64)
            for source, candidate in (
                ("left_last_good_sol_q", self.left_last_good_sol_q),
                ("fallback_q", fallback_q),
                ("current_lr_arm_motor_q", current_lr_arm_motor_q),
            ):
                candidate_q = _valid_q14(candidate)
                if candidate_q is not None:
                    fallback_source = source
                    fallback_sol_q = np.concatenate(
                        (candidate_q[:7], right_hold_q)
                    ).astype(np.float64)
                    break

            self.left_last_solve_success = False
            self.left_last_fallback_source = fallback_source

            now_t = time.time()
            if (
                self.Debug
                and (
                    self.left_ik_failure_count <= 5
                    or now_t - self.left_last_ik_failure_log_t >= 1.0
                )
            ):
                try:
                    return_status = self.left_opti.stats().get(
                        "return_status", type(e).__name__
                    )
                except Exception:
                    return_status = type(e).__name__
                print(
                    "[LeftIKDebug] solve failed, returning "
                    f"{fallback_source}; left_init_q preserved "
                    f"failure_count={self.left_ik_failure_count} "
                    f"return_status={return_status}"
                )
                self.left_last_ik_failure_log_t = now_t

            return fallback_sol_q.copy()

    def solve_ik(
        self,
        left_wrist,
        right_wrist,
        current_lr_arm_motor_q=None,
        current_lr_arm_motor_dq=None,
        fallback_q=None,
    ):
        def _valid_q(q):
            if q is None:
                return None
            q = np.asarray(q, dtype=np.float64).reshape(-1)
            if q.shape[0] != self.reduced_robot.model.nq:
                return None
            if not np.all(np.isfinite(q)):
                return None
            return q.copy()

        seed_q = _valid_q(current_lr_arm_motor_q)
        if seed_q is None:
            seed_q = _valid_q(self.init_data)
        if seed_q is None:
            seed_q = np.zeros(self.reduced_robot.model.nq, dtype=np.float64)

        self.opti.set_initial(self.var_q, seed_q)

        left_wrist, right_wrist = self.scale_arms(left_wrist, right_wrist)

        if self.Visualization:
            self.vis.viewer["L_ee_target"].set_transform(
                left_wrist
            )  # for visualization
            self.vis.viewer["R_ee_target"].set_transform(
                right_wrist
            )  # for visualization

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, seed_q)  # for smooth
        # # 2. [新增] 限制位姿跳变步长
        # # max_dist 和 max_angle 可以根据你的控制频率调整 (例如 ??Hz 下 0.02m 和 0.05rad  )
        # limited_left = self.limit_pose(self.last_target_left, left_wrist, max_dist=0.02, max_angle=0.05)
        # limited_right = self.limit_pose(self.last_target_right, right_wrist, max_dist=0.02, max_angle=0.05)
        # if self.Visualization:
        #     self.vis.viewer["L_ee_target"].set_transform(
        #         limited_left
        #     )  # 可视化受限后的真实目标点
        #     self.vis.viewer["R_ee_target"].set_transform(
        #         limited_right
        #     )

        # # 3. 将【限幅后】的位姿传给优化器
        # self.opti.set_value(self.param_tf_l, limited_left)
        # self.opti.set_value(self.param_tf_r, limited_right)
        # self.opti.set_value(self.var_q_last, self.init_data)  # for smooth


        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = np.asarray(self.opti.value(self.var_q), dtype=np.float64).reshape(-1)
            if sol_q.shape[0] != self.reduced_robot.model.nq:
                raise ValueError(
                    f"IK returned {sol_q.shape[0]} joints, "
                    f"expected {self.reduced_robot.model.nq}"
                )
            if not np.all(np.isfinite(sol_q)):
                raise ValueError("IK returned non-finite joint values")
            # self.smooth_filter.add_data(sol_q)
            # sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - seed_q) * 0.0

            self.init_data = sol_q.copy()
            self.last_good_sol_q = sol_q.copy()

            # sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            # if self.Visualization:
            #     self.vis.display(sol_q)  # for visualization

            return sol_q

        except Exception as e:
            self.ik_failure_count += 1
            fallback_source = "seed_q"
            fallback_sol_q = seed_q
            for source, candidate in (
                ("last_good_sol_q", self.last_good_sol_q),
                ("fallback_q", fallback_q),
                ("init_data", self.init_data),
                ("current_lr_arm_motor_q", current_lr_arm_motor_q),
            ):
                candidate_q = _valid_q(candidate)
                if candidate_q is not None:
                    fallback_source = source
                    fallback_sol_q = candidate_q
                    break

            now_t = time.time()
            if (
                self.Debug
                and (
                    self.ik_failure_count <= 5
                    or now_t - self.last_ik_failure_log_t >= 1.0
                )
            ):
                try:
                    return_status = self.opti.stats().get("return_status", type(e).__name__)
                except Exception:
                    return_status = type(e).__name__
                print(
                    "[IKDebug] solve failed, returning "
                    f"{fallback_source}; init_data preserved "
                    f"failure_count={self.ik_failure_count} "
                    f"return_status={return_status}"
                )
                self.last_ik_failure_log_t = now_t

            return fallback_sol_q.copy()

#策略1增加平滑的权重ok  2增加 迭代的初始的 关节角ok   3限制其目标点位置与姿态的距离？？限制多少比较合适？待定
class Arm_IK:
    def __init__(self):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.robot = pin.RobotWrapper.BuildFromURDF(
            "../assets/h1_2/h1_2.urdf", "../assets/h1_2"
        )
        # self.robot = pin.RobotWrapper.BuildFromURDF('../../assets/h1_2/h1_2.urdf', '../../assets/h1_2/') # for test

        self.mixed_jointsToLockIDs = [
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "torso_joint",
            "L_index_proximal_joint",
            "L_index_intermediate_joint",
            "L_middle_proximal_joint",
            "L_middle_intermediate_joint",
            "L_pinky_proximal_joint",
            "L_pinky_intermediate_joint",
            "L_ring_proximal_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "L_thumb_distal_joint",
            "R_index_proximal_joint",
            "R_index_intermediate_joint",
            "R_middle_proximal_joint",
            "R_middle_intermediate_joint",
            "R_pinky_proximal_joint",
            "R_pinky_intermediate_joint",
            "R_ring_proximal_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_intermediate_joint",
            "R_thumb_distal_joint",
        ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        # for i, joint in enumerate(self.reduced_robot.model.joints):
        #     joint_name = self.reduced_robot.model.names[i]
        #     print(f"Joint {i}: {joint_name}, ID: {joint.id}")

        self.reduced_robot.model.addFrame(
            pin.Frame(
                "L_ee",
                self.reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0.1, 0, 0]).T),
                pin.FrameType.OP_FRAME,
            )
        )

        self.reduced_robot.model.addFrame(
            pin.Frame(
                "R_ee",
                self.reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0.1, 0, 0]).T),
                pin.FrameType.OP_FRAME,
            )
        )

        # self.init_data = np.zeros(self.reduced_robot.model.nq)
        # 增加一步：给 var_q 一个良好的初始值（上一次的解）
        self.opti.set_initial(self.var_q, self.init_data)
        # # Initialize the Meshcat visualizer  for visualization
        # self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
        # self.vis.initViewer(open=True)
        # self.vis.loadViewerModel("pinocchio")
        # self.vis.displayFrames(True, frame_ids=[113, 114], axis_length = 0.15, axis_width = 5)
        # self.vis.display(pin.neutral(self.reduced_robot.model))

        # # for i in range(self.reduced_robot.model.nframes):
        # #    frame = self.reduced_robot.model.frames[i]
        # #    frame_id = self.reduced_robot.model.getFrameId(frame.name)
        # #    print(f"Frame ID: {frame_id}, Name: {frame.name}")

        # # Enable the display of end effector target frames with short axis lengths and greater width.
        # frame_viz_names = ['L_ee_target', 'R_ee_target']
        # FRAME_AXIS_POSITIONS = (
        #     np.array([[0, 0, 0], [1, 0, 0],
        #               [0, 0, 0], [0, 1, 0],
        #               [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
        # )
        # FRAME_AXIS_COLORS = (
        #     np.array([[1, 0, 0], [1, 0.6, 0],
        #               [0, 1, 0], [0.6, 1, 0],
        #               [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
        # )
        # axis_length = 0.1
        # axis_width = 10
        # for frame_viz_name in frame_viz_names:
        #     self.vis.viewer[frame_viz_name].set_object(
        #         mg.LineSegments(
        #             mg.PointsGeometry(
        #                 position=axis_length * FRAME_AXIS_POSITIONS,
        #                 color=FRAME_AXIS_COLORS,
        #             ),
        #             mg.LineBasicMaterial(
        #                 linewidth=axis_width,
        #                 vertexColors=True,
        #             ),
        #         )
        #     )

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.L_hand_id].inverse() * cpin.SE3(self.cTf_l)
                    ).vector,
                    cpin.log6(
                        self.cdata.oMf[self.R_hand_id].inverse() * cpin.SE3(self.cTf_r)
                    ).vector,
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.totalcost = casadi.sumsqr(
            self.error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.regularization = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last) # for smooth

        # Setting optimization constraints and goals
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        # self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)
        self.opti.minimize(50 * self.totalcost + 0.01 * self.regularization + 10 * self.smooth_cost) # for smooth

        opts = {
            "ipopt": {"print_level": 3, "max_iter": 50, "tol": 1e-4},
            "print_time": True,
        }
        self.opti.solver("ipopt", opts)
    
    def adjust_pose(
        self,
        human_left_pose,
        human_right_pose,
        human_arm_length=0.60,
        robot_arm_length=0.75,
    ):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def ik_fun(self, left_pose, right_pose, motorstate=None, motorV=None):
        if motorstate is not None:
            self.init_data = motorstate
        self.opti.set_initial(self.var_q, self.init_data)

        left_pose, right_pose = self.adjust_pose(left_pose, right_pose)

        # self.vis.viewer['L_ee_target'].set_transform(left_pose)     # for visualization
        # self.vis.viewer['R_ee_target'].set_transform(right_pose)    # for visualization

        # ：将上一时刻的位姿传给优化器作为惩罚基准
        self.opti.set_value(self.var_q_last, self.init_data)

        self.opti.set_value(self.param_tf_l, left_pose)
        self.opti.set_value(self.param_tf_r, right_pose)
        # self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            # sol = self.opti.solve()
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)

            # self.vis.display(sol_q) # for visualization
            self.init_data = sol_q

            if motorV is not None:
                v = motorV * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0
            #须要求力矩？
            tau_ff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                sol_q,
                v,
                np.zeros(self.reduced_robot.model.nv),
            )

            return sol_q, tau_ff, True

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            # sol_q = self.opti.debug.value(self.var_q)   # return original value
            # return sol_q, "", False
            return self.init_data, np.zeros(self.reduced_robot.model.nv), False


if __name__ == "__main__":
    arm_ik = Arm_IK()

    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.3, +0.2, 0.2]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.3, -0.2, 0.2]),
    )

    rotation_speed = 0.005  # Rotation speed in radians per iteration

    user_input = input(
        "Please enter the start signal (enter 's' to start the subsequent program):"
    )
    if user_input.lower() == "s":

        for i in range(150):
            angle = rotation_speed * i
            L_quat = pin.Quaternion(
                np.cos(angle / 2), 0, np.sin(angle / 2), 0
            )  # y axis
            R_quat = pin.Quaternion(
                np.cos(angle / 2), 0, 0, np.sin(angle / 2)
            )  # z axis

            L_tf_target.translation += np.array([0.001, 0.001, 0.001])
            R_tf_target.translation += np.array([0.001, -0.001, 0.001])
            L_tf_target.rotation = L_quat.toRotationMatrix()
            R_tf_target.rotation = R_quat.toRotationMatrix()

            arm_ik.ik_fun(L_tf_target.homogeneous, R_tf_target.homogeneous)
            time.sleep(0.02)
