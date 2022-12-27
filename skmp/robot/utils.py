from typing import List

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rpy_angle, rpy_matrix
from skrobot.model import RobotModel


def set_robot_state(
    robot_model: RobotModel, joint_names: List[str], angles: np.ndarray, with_base=False
) -> None:
    if with_base:
        assert len(joint_names) + 3 == len(angles)
    else:
        assert len(joint_names) == len(angles)

    if with_base:
        av_joint, av_base = angles[:-3], angles[-3:]
        x, y, theta = av_base
        co = Coordinates(pos=[x, y, 0.0], rot=rpy_matrix(theta, 0.0, 0.0))
        robot_model.newcoords(co)
    else:
        av_joint = angles

    for joint_name, angle in zip(joint_names, av_joint):
        robot_model.__dict__[joint_name].joint_angle(angle)


def get_robot_state(robot_model: RobotModel, joint_names: List[str], with_base=False) -> np.ndarray:
    av_joint = np.array([robot_model.__dict__[jn].joint_angle() for jn in joint_names])
    if not with_base:
        return av_joint
    else:
        x, y, _ = robot_model.translation
        rpy = rpy_angle(robot_model.rotation)[0]
        theta = rpy[0]
        av_whole = np.hstack((av_joint, [x, y, theta]))
        return av_whole
