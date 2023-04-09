from typing import List

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rpy_matrix
from skrobot.model import RobotModel
from tinyfk import BaseType


def set_robot_state(
    robot_model: RobotModel,
    joint_names: List[str],
    angles: np.ndarray,
    base_type: BaseType = BaseType.FIXED,
) -> None:
    if base_type == BaseType.PLANER:
        assert len(joint_names) + 3 == len(angles)
        av_joint, av_base = angles[:-3], angles[-3:]
        x, y, theta = av_base
        co = Coordinates(pos=[x, y, 0.0], rot=rpy_matrix(theta, 0.0, 0.0))
        robot_model.newcoords(co)
    elif base_type == BaseType.FLOATING:
        assert len(joint_names) + 6 == len(angles)
        av_joint, av_base = angles[:-6], angles[-6:]
        xyz, rpy = av_base[:3], av_base[3:]
        co = Coordinates(pos=xyz, rot=rpy_matrix(*np.flip(rpy)))
        robot_model.newcoords(co)
    else:
        assert len(joint_names) == len(angles)
        av_joint = angles

    for joint_name, angle in zip(joint_names, av_joint):
        robot_model.__dict__[joint_name].joint_angle(angle)
