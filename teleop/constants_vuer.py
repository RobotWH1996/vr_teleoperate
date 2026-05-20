import numpy as np

tip_indices = [4, 9, 14, 19, 24]
# tip_indices = [3, 7, 12, 17, 22]

# brainco_hand

hand2inspire_l_arm = np.array([[1, 0, 0, 0],
                               [0, 0, -1, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 1]])

hand2inspire_r_arm = np.array([[1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, -1, 0, 0],
                               [0, 0, 0, 1]])

hand2inspire_l_finger = np.array([[0, -1, 0, 0],
                                  [0, 0, -1, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 0, 1]])

hand2inspire_r_finger = np.array([[0, -1, 0, 0],
                                  [0, 0, -1, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 0, 1]])

x100_arm_2_brainco_hand = np.array([
        [ 0, 0, 1, 0],
        [ 0, 1, 0, 0],
        [ -1, 0, 0, 0],
        [ 0, 0, 0, 1]
    ])
    

grd_yup2grd_zup = np.array([[0, 0, -1, 0],
                            [-1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])

# legacy
hand2inspire = np.array([[0, -1, 0, 0],
                         [0, 0, -1, 0],
                         [1, 0, 0, 0],
                         [0, 0, 0, 1]])

hand2X100 = np.array([[0, -1, 0, 0],
                      [0, 0, 1, 0],
                      [-1, 0, 0, 0], 
                      [0, 0, 0, 1]])

grd_xup2grd_zup = np.array([
        [0, 0, -1, 0], # 新的X轴是原来的Y轴
        [0, 1, 0, 0], # 新的Y轴是原来的Z轴
        [1, 0, 0, 0], # 新的Z轴是原来的X轴
        [0, 0, 0, 1]
    ])

X100_l_arm = np.array([
        [0, 0, -1, 0],
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

X100_r_arm = np.array([
    [0, 0, -1, 0],  # 新的X轴 = -原Z轴
    [1, 0, 0, 0],    # 新的Y轴 = 原X轴
    [0, -1, 0, 0],   # 新的Z轴 = -原Y轴
    [0, 0, 0, 1]
])


X100_arm_2_hand = np.array([
        [-1, 0, 0, 0],
        [ 0,-1, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 0, 0, 1]
    ])

# X100_arm_2_hand = np.array([
#         [-1, 0, 0, 0],
#         [ 0,-1, 0, 0],
#         [ 0, 0, 1, 0],
#         [ 0, 0, 0, 1]
#     ])

X100_arm_2_hand = np.array([
        [ 0, 0, 1, 0],
        [ 0, 1, 0, 0],
        [ -1, 0, 0, 0],
        [ 0, 0, 0, 1]
    ])


left_joint_names  = [ 'L_pinky_proximal_joint', 'L_ring_proximal_joint', 'L_middle_proximal_joint',
                                                       'L_index_proximal_joint', 'L_thumb_proximal_pitch_joint', 'L_thumb_proximal_yaw_joint' ]
right_joint_names = [ 'R_pinky_proximal_joint', 'R_ring_proximal_joint', 'R_middle_proximal_joint',
                                                       'R_index_proximal_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_proximal_yaw_joint' ]