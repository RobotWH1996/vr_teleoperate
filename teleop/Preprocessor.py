import math
import numpy as np

from constants_vuer import (
    X100_l_arm,
    X100_r_arm,
    grd_yup2grd_zup,
    grd_xup2grd_zup,
    X100_arm_2_hand,
    hand2inspire_r_arm,
    hand2inspire_l_finger,
    hand2inspire_r_finger,
    hand2inspire,
    x100_arm_2_brainco_hand,
)
from motion_utils import mat_update, fast_mat_inv


class VuerPreprocessor:
    def __init__(self):
        self.vuer_head_mat = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 1.5], [0, 0, 1, -0.2], [0, 0, 0, 1]]
        )

        self.vuer_right_wrist_mat = np.array(
            [[1, 0, 0, 0.5], [0, 1, 0, 1], [0, 0, 1, -0.5], [0, 0, 0, 1]]
        )

        self.vuer_left_wrist_mat = np.array(
            [[1, 0, 0, -0.5], [0, 1, 0, 1], [0, 0, 1, -0.5], [0, 0, 0, 1]]
        )

    def process(self, tv):
        self.vuer_head_mat = mat_update(self.vuer_head_mat, tv.head_matrix.copy())
        print("vuer_head_mat", self.vuer_head_mat)
        self.vuer_right_wrist_mat = mat_update(
            self.vuer_right_wrist_mat, tv.right_hand.copy()
        )
        self.vuer_left_wrist_mat = mat_update(
            self.vuer_left_wrist_mat, tv.left_hand.copy()
        )

        # change of basis
        head_mat = grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)
        right_wrist_mat = (
            grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        )
        left_wrist_mat = (
            grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        )

        # 腕部坐标系转换 y up -> z up
        rel_left_wrist_mat = left_wrist_mat @ X100_l_arm
        rel_left_wrist_mat[0:3, 3] = rel_left_wrist_mat[0:3, 3] - head_mat[0:3, 3]

        rel_right_wrist_mat = right_wrist_mat @ X100_r_arm  # wTr = wTh @ hTr
        rel_right_wrist_mat[0:3, 3] = rel_right_wrist_mat[0:3, 3] - head_mat[0:3, 3]

        # homogeneous
        left_fingers = np.concatenate(
            [tv.left_landmarks.copy().T, np.ones((1, tv.left_landmarks.shape[0]))]
        )
        right_fingers = np.concatenate(
            [tv.right_landmarks.copy().T, np.ones((1, tv.right_landmarks.shape[0]))]
        )

        # change of basis
        left_fingers = grd_yup2grd_zup @ left_fingers
        right_fingers = grd_yup2grd_zup @ right_fingers

        rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers
        rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers

        # x100_arm_2_brainco_hand X100_arm_2_hand
        rel_left_fingers = (x100_arm_2_brainco_hand.T @ rel_left_fingers)[0:3, :].T
        rel_right_fingers = (x100_arm_2_brainco_hand.T @ rel_right_fingers)[0:3, :].T

        return (
            head_mat,
            rel_left_wrist_mat,
            rel_right_wrist_mat,
            rel_left_fingers,
            rel_right_fingers,
        )

    def get_hand_gesture(self, tv):
        self.vuer_right_wrist_mat = mat_update(
            self.vuer_right_wrist_mat, tv.right_hand.copy()
        )
        self.vuer_left_wrist_mat = mat_update(
            self.vuer_left_wrist_mat, tv.left_hand.copy()
        )

        # change of basis
        right_wrist_mat = (
            grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        )
        left_wrist_mat = (
            grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        )

        left_fingers = np.concatenate(
            [tv.left_landmarks.copy().T, np.ones((1, tv.left_landmarks.shape[0]))]
        )
        right_fingers = np.concatenate(
            [tv.right_landmarks.copy().T, np.ones((1, tv.right_landmarks.shape[0]))]
        )

        # change of basis
        left_fingers = grd_yup2grd_zup @ left_fingers
        right_fingers = grd_yup2grd_zup @ right_fingers

        rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers
        rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers
        rel_left_fingers = (hand2inspire_l_finger.T @ rel_left_fingers)[0:3, :].T
        rel_right_fingers = (hand2inspire_r_finger.T @ rel_right_fingers)[0:3, :].T
        all_fingers = np.concatenate([rel_left_fingers, rel_right_fingers], axis=0)

        return all_fingers


class VuerPreprocessorLegacy:
    def __init__(self):
        self.vuer_head_mat = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 1.5], [0, 0, 1, -0.2], [0, 0, 0, 1]]
        )
        self.vuer_right_wrist_mat = np.array(
            [[1, 0, 0, 0.7], [0, 1, 0, 1.0], [0, 0, 1, -0.5], [0, 0, 0, 1]]
        )

        self.vuer_left_wrist_mat = np.array(
            [[1, 0, 0, -0.7], [0, 1, 0, 1.0], [0, 0, 1, -0.5], [0, 0, 0, 1]]
        )
        self.head_mat = grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)

    def process(self, tv,reset_head_flag=True):
        self.vuer_head_mat = mat_update(self.vuer_head_mat, tv.head_matrix.copy())
        # print("vuer_head_mat", self.vuer_head_mat)
        self.vuer_right_wrist_mat = mat_update(
            self.vuer_right_wrist_mat, tv.right_hand.copy()
        )
        self.vuer_left_wrist_mat = mat_update(
            self.vuer_left_wrist_mat, tv.left_hand.copy()
        )
        # print("vuer_left_wrist_mat", self.vuer_left_wrist_mat)
        # print("vuer_right_wrist_mat", self.vuer_right_wrist_mat)
        # change of basis
        head_mat = grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)
        # if reset_head_flag:
        self.head_mat = head_mat
        # print("head_mat", head_mat)
        right_wrist_mat = (
            grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        )
        left_wrist_mat = (
            grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        )

        self.left_wrist_mat_zup = left_wrist_mat.copy()
        self.right_wrist_mat_zup = right_wrist_mat.copy()

        # 腕部坐标系转换 y up -> z up
        rel_left_wrist_mat = left_wrist_mat @ X100_l_arm
        rel_left_wrist_mat[0:3, 3] = rel_left_wrist_mat[0:3, 3] - self.head_mat[0:3, 3]
        # print("rel_left_wrist_mat", rel_left_wrist_mat)
        rel_right_wrist_mat = right_wrist_mat @ X100_r_arm  # wTr = wTh @ hTr
        rel_right_wrist_mat[0:3, 3] = rel_right_wrist_mat[0:3, 3] - self.head_mat[0:3, 3]
        # print("rel_right_wrist_mat", rel_right_wrist_mat)
        # homogeneous
        left_fingers = np.concatenate(
            [tv.left_landmarks.copy().T, np.ones((1, tv.left_landmarks.shape[0]))]
        )
        right_fingers = np.concatenate(
            [tv.right_landmarks.copy().T, np.ones((1, tv.right_landmarks.shape[0]))]
        )

        # change of basis
        left_fingers = grd_yup2grd_zup @ left_fingers
        right_fingers = grd_yup2grd_zup @ right_fingers
        # print("left_fingers",left_fingers)
        # print("right_fingers",right_fingers)
        rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers
        rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers
        # print("rel_left_fingers",rel_left_fingers)
        # print("right_fingers",right_fingers)
        # x100_arm_2_brainco_hand X100_arm_2_hand
        rel_left_fingers = (X100_arm_2_hand.T @ rel_left_fingers)[0:3, :].T
        rel_right_fingers = (X100_arm_2_hand.T @ rel_right_fingers)[0:3, :].T

        return (
            head_mat,
            rel_left_wrist_mat,
            rel_right_wrist_mat,
            rel_left_fingers,
            rel_right_fingers,
        )

    def get_hand_gesture(self, tv):
        self.vuer_right_wrist_mat = mat_update(
            self.vuer_right_wrist_mat, tv.right_hand.copy()
        )
        self.vuer_left_wrist_mat = mat_update(
            self.vuer_left_wrist_mat, tv.left_hand.copy()
        )

        # change of basis
        right_wrist_mat = (
            grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        )
        left_wrist_mat = (
            grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        )

        left_fingers = np.concatenate(
            [tv.left_landmarks.copy().T, np.ones((1, tv.left_landmarks.shape[0]))]
        )
        right_fingers = np.concatenate(
            [tv.right_landmarks.copy().T, np.ones((1, tv.right_landmarks.shape[0]))]
        )

        # change of basis
        left_fingers = grd_yup2grd_zup @ left_fingers
        right_fingers = grd_yup2grd_zup @ right_fingers

        rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers
        rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers
        rel_left_fingers = (hand2inspire_l_finger.T @ rel_left_fingers)[0:3, :].T
        rel_right_fingers = (hand2inspire_r_finger.T @ rel_right_fingers)[0:3, :].T
        all_fingers = np.concatenate([rel_left_fingers, rel_right_fingers], axis=0)

        return all_fingers
