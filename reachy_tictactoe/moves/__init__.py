import os
import numpy as np

from glob import glob

dir_path = os.path.dirname(os.path.realpath(__file__))


names = [
    os.path.splitext(os.path.basename(f))[0]
    for f in glob(os.path.join(dir_path, '*.npz'))
]

moves = {
    name: np.load(os.path.join(dir_path, f'{name}.npz'))
    for name in names
}


rest_pos = {
    'right_arm.shoulder_pitch': 50,
    'right_arm.shoulder_roll': -15,
    'right_arm.arm_yaw': 0,
    'right_arm.elbow_pitch': -80,
    'right_arm.hand.forearm_yaw': -15,
    'right_arm.hand.wrist_pitch': -60,
    'right_arm.hand.wrist_roll': 0,
}

base_pos = {
    'right_arm.shoulder_pitch': 60,
    'right_arm.shoulder_roll': -15,
    'right_arm.arm_yaw': 0,
    'right_arm.elbow_pitch': -95,
    'right_arm.hand.forearm_yaw': -15,
    'right_arm.hand.wrist_pitch': -50,
    'right_arm.hand.wrist_roll': 0,
    'right_arm.hand.gripper': -45,
}