import copy
import uuid
from dataclasses import dataclass
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
from skrobot.models import PR2
from tinyfk import BaseType

from skmp.collision import SphereCollection
from skmp.constraint import BoxConst, PairWiseSelfCollFreeConst
from skmp.kinematics import (
    ArticulatedCollisionKinematicsMap,
    ArticulatedEndEffectorKinematicsMap,
)


class CollisionMode(Enum):
    DEFAULT = 0
    RARM = 1
    LARM = 2
    RGRIPPER = 3
    LGRIPPER = 4


@dataclass
class PR2Config:
    control_arm: Literal["rarm", "larm", "dual"] = "rarm"
    collision_mode: CollisionMode = CollisionMode.DEFAULT
    selcol_mode: Literal["easy", "normal"] = "easy"
    base_type: BaseType = BaseType.FIXED
    use_torso: bool = False

    @classmethod
    def urdf_path(cls) -> Path:
        return Path("~/.skrobot/pr2_description/pr2.urdf")  # temp

    @classmethod
    def get_default_config_table(cls) -> Dict[str, float]:
        table = {
            "torso_lift_joint": 0.3,
            "l_shoulder_pan_joint": np.deg2rad(75),
            "l_shoulder_lift_joint": np.deg2rad(50),
            "l_upper_arm_roll_joint": np.deg2rad(110),
            "l_elbow_flex_joint": np.deg2rad(-110),
            "l_forearm_roll_joint": np.deg2rad(-20),
            "l_wrist_flex_joint": np.deg2rad(-10),
            "l_wrist_roll_joint": np.deg2rad(-10),
            "r_shoulder_pan_joint": np.deg2rad(-75),
            "r_shoulder_lift_joint": np.deg2rad(50),
            "r_upper_arm_roll_joint": np.deg2rad(-110),
            "r_elbow_flex_joint": np.deg2rad(-110),
            "r_forearm_roll_joint": np.deg2rad(20),
            "r_wrist_flex_joint": np.deg2rad(-10),
            "r_wrist_roll_joint": np.deg2rad(-10),
            "head_pan_joint": 0.0,
            "head_tilt_joint": np.deg2rad(50),
        }
        return table

    @classmethod
    def rarm_joint_names(cls) -> List[str]:
        return [
            "r_shoulder_pan_joint",
            "r_shoulder_lift_joint",
            "r_upper_arm_roll_joint",
            "r_elbow_flex_joint",
            "r_forearm_roll_joint",
            "r_wrist_flex_joint",
            "r_wrist_roll_joint",
        ]

    @classmethod
    def larm_joint_names(cls) -> List[str]:
        return [
            "l_shoulder_pan_joint",
            "l_shoulder_lift_joint",
            "l_upper_arm_roll_joint",
            "l_elbow_flex_joint",
            "l_forearm_roll_joint",
            "l_wrist_flex_joint",
            "l_wrist_roll_joint",
        ]

    @classmethod
    def rarm_collision_link_names(cls) -> List[str]:
        return [
            "r_shoulder_pan_link",
            "r_shoulder_lift_link",
            "r_upper_arm_link",
            "r_forearm_link",
            "r_gripper_palm_link",
            "r_gripper_r_finger_link",
            "r_gripper_l_finger_link",
        ]

    @classmethod
    def larm_collision_link_names(cls) -> List[str]:
        return [
            "l_shoulder_pan_link",
            "l_shoulder_lift_link",
            "l_upper_arm_link",
            "l_forearm_link",
            "l_gripper_palm_link",
            "l_gripper_r_finger_link",
            "l_gripper_l_finger_link",
        ]

    @classmethod
    def rgripper_collision_link_names(cls) -> List[str]:
        return [
            "r_gripper_palm_link",
            "r_gripper_r_finger_link",
            "r_gripper_l_finger_link",
        ]

    @classmethod
    def lgripper_collision_link_names(cls) -> List[str]:
        return [
            "l_gripper_palm_link",
            "l_gripper_r_finger_link",
            "l_gripper_l_finger_link",
        ]

    @classmethod
    def base_collision_link_names(cls) -> List[str]:
        return ["base_link"]

    def _get_control_joint_names(self) -> List[str]:
        if self.control_arm == "rarm":
            joint_names = self.rarm_joint_names()
        elif self.control_arm == "larm":
            joint_names = self.larm_joint_names()
        elif self.control_arm == "dual":
            joint_names = self.rarm_joint_names() + self.larm_joint_names()
        else:
            assert False

        if self.use_torso:
            joint_names.append("torso_lift_joint")

        return joint_names

    def _get_endeffector_names(self) -> List[str]:
        if self.control_arm == "rarm":
            endeffector_names = ["r_gripper_tool_frame"]
        elif self.control_arm == "larm":
            endeffector_names = ["l_gripper_tool_frame"]
        elif self.control_arm == "dual":
            endeffector_names = ["r_gripper_tool_frame", "l_gripper_tool_frame"]
        else:
            assert False
        return endeffector_names

    def get_endeffector_kin(self):
        kinmap = ArticulatedEndEffectorKinematicsMap(
            self.urdf_path(),
            self._get_control_joint_names(),
            self._get_endeffector_names(),
            base_type=self.base_type,
        )
        return kinmap

    def get_box_const(self) -> BoxConst:
        if self.base_type == BaseType.PLANER:
            base_bounds = np.array([-1.0, -2.0, -1.0]), np.array([2.0, 2.0, 1.0])
        elif self.base_type == BaseType.FLOATING:
            base_bounds = -np.ones(6), np.ones(6)
        else:
            base_bounds = None

        bounds = BoxConst.from_urdf(
            self.urdf_path(), self._get_control_joint_names(), base_bounds=base_bounds
        )
        return bounds

    def _get_collision_link_names(self):
        mode = self.collision_mode
        if mode == CollisionMode.DEFAULT:
            return (
                self.rarm_collision_link_names()
                + self.larm_collision_link_names()
                + self.base_collision_link_names()
            )
        elif mode == CollisionMode.RARM:
            return self.rarm_collision_link_names()
        elif mode == CollisionMode.LARM:
            return self.larm_collision_link_names()
        elif mode == CollisionMode.RGRIPPER:
            return self.rgripper_collision_link_names()
        elif mode == CollisionMode.LGRIPPER:
            return self.lgripper_collision_link_names()
        else:
            assert False

    def get_collision_kin(self) -> ArticulatedCollisionKinematicsMap:
        link_wise_sphere_collection: Dict[str, SphereCollection] = {}

        def unique_name(link_name) -> str:
            return link_name + str(uuid.uuid4())[:8]

        link_name = "r_upper_arm_link"
        collection = []

        collection.append((np.array([0.0, 0.0, 0.0]), 0.17, unique_name(link_name)))
        collection.append((np.array([0.20, 0, -0.01]), 0.1, unique_name(link_name)))

        collection.append((np.array([0.255, +0.0245, -0.042]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.255, -0.0245, -0.042]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.29, +0.0245, -0.028]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.29, -0.0245, -0.028]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.29, +0.0245, -0.044]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.29, -0.0245, -0.044]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.325, +0.0245, -0.023]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.325, -0.0245, -0.023]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.325, +0.0245, -0.042]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.325, -0.0245, -0.042]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.35, +0.0245, -0.028]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.35, -0.0245, -0.028]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.35, +0.0245, -0.042]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.35, -0.0245, -0.042]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.375, +0.0245, -0.042]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.375, -0.0245, -0.042]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.4, +0.0245, -0.042]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.4, -0.0245, -0.042]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.42, +0.0245, -0.038]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.42, -0.0245, -0.038]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.435, +0.0245, -0.025]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.435, -0.0245, -0.025]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.442, +0.0245, -0.003]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.442, -0.0245, -0.003]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.438, +0.0245, 0.018]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.438, -0.0245, 0.018]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.425, +0.0245, 0.032]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.425, -0.0245, 0.032]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.410, +0.0245, 0.040]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.410, -0.0245, 0.040]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.390, +0.0245, 0.040]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.390, -0.0245, 0.040]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.375, +0.0245, 0.040]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.375, -0.0245, 0.040]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.365, +0.0245, 0.020]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.365, -0.0245, 0.020]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.360, +0.0245, 0.0195]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.360, -0.0245, 0.0195]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.395, +0.0245, 0.005]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.395, -0.0245, 0.005]), 0.05, unique_name(link_name)))

        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection[link_name] = tmp

        link_name = "l_upper_arm_link"
        collection = []
        collection.append((np.array([0.0, 0.0, 0.0]), 0.17, unique_name(link_name)))
        collection.append((np.array([0.20, 0, -0.01]), 0.1, unique_name(link_name)))

        collection.append((np.array([0.255, +0.0245, -0.042]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.255, -0.0245, -0.042]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.29, +0.0245, -0.028]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.29, -0.0245, -0.028]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.29, +0.0245, -0.044]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.29, -0.0245, -0.044]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.325, +0.0245, -0.023]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.325, -0.0245, -0.023]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.325, +0.0245, -0.042]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.325, -0.0245, -0.042]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.35, +0.0245, -0.028]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.35, -0.0245, -0.028]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.35, +0.0245, -0.042]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.35, -0.0245, -0.042]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.375, +0.0245, -0.042]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.375, -0.0245, -0.042]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.4, +0.0245, -0.042]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.4, -0.0245, -0.042]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.42, +0.0245, -0.038]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.42, -0.0245, -0.038]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.435, +0.0245, -0.025]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.435, -0.0245, -0.025]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.442, +0.0245, -0.003]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.442, -0.0245, -0.003]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.438, +0.0245, 0.018]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.438, -0.0245, 0.018]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.425, +0.0245, 0.032]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.425, -0.0245, 0.032]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.410, +0.0245, 0.040]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.410, -0.0245, 0.040]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.390, +0.0245, 0.040]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.390, -0.0245, 0.040]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.375, +0.0245, 0.040]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.375, -0.0245, 0.040]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.365, +0.0245, 0.020]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.365, -0.0245, 0.020]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.360, +0.0245, 0.0195]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.360, -0.0245, 0.0195]), 0.05, unique_name(link_name)))

        collection.append((np.array([0.395, +0.0245, 0.005]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.395, -0.0245, 0.005]), 0.05, unique_name(link_name)))

        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection[link_name] = tmp

        link_name = "r_elbow_flex_link"
        collection = []
        collection.append((np.array([0.07, 0.0, -0.04]), 0.045, unique_name(link_name)))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection[link_name] = tmp

        link_name = "l_elbow_flex_link"
        collection = []
        collection.append((np.array([0.07, 0.0, -0.04]), 0.045, unique_name(link_name)))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection[link_name] = tmp

        link_name = "r_forearm_link"
        collection = []
        # collection.append((np.array([0.0, 0.0, 0.0]), 0.1, unique_name(link_name)))
        collection.append((np.array([0.12, 0.0, 0.005]), 0.07, unique_name(link_name)))
        collection.append((np.array([0.16, 0.02, -0.005]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.16, -0.02, -0.005]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.19, 0.02, -0.005]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.19, -0.02, -0.005]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.22, 0.02, -0.015]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.22, -0.02, -0.014]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.25, 0.025, -0.015]), 0.045, unique_name(link_name)))
        collection.append((np.array([0.25, -0.025, -0.015]), 0.045, unique_name(link_name)))
        collection.append((np.array([0.28, 0.025, -0.016]), 0.044, unique_name(link_name)))
        collection.append((np.array([0.28, -0.025, -0.016]), 0.044, unique_name(link_name)))
        collection.append((np.array([0.31, 0.025, -0.005]), 0.045, unique_name(link_name)))
        collection.append((np.array([0.31, -0.025, -0.005]), 0.045, unique_name(link_name)))
        collection.append((np.array([0.32, 0.023, -0.002]), 0.045, unique_name(link_name)))
        collection.append((np.array([0.32, -0.023, -0.002]), 0.045, unique_name(link_name)))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection[link_name] = tmp

        link_name = "l_forearm_link"
        collection = []
        collection.append((np.array([0.0, 0.0, 0.0]), 0.1, unique_name(link_name)))
        collection.append((np.array([0.12, 0.0, 0.005]), 0.07, unique_name(link_name)))
        collection.append((np.array([0.16, 0.02, -0.005]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.16, -0.02, -0.005]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.19, 0.02, -0.005]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.19, -0.02, -0.005]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.22, 0.02, -0.015]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.22, -0.02, -0.014]), 0.05, unique_name(link_name)))
        collection.append((np.array([0.25, 0.025, -0.015]), 0.045, unique_name(link_name)))
        collection.append((np.array([0.25, -0.025, -0.015]), 0.045, unique_name(link_name)))
        collection.append((np.array([0.28, 0.025, -0.016]), 0.044, unique_name(link_name)))
        collection.append((np.array([0.28, -0.025, -0.016]), 0.044, unique_name(link_name)))
        collection.append((np.array([0.31, 0.025, -0.005]), 0.045, unique_name(link_name)))
        collection.append((np.array([0.31, -0.025, -0.005]), 0.045, unique_name(link_name)))
        collection.append((np.array([0.32, 0.023, -0.002]), 0.045, unique_name(link_name)))
        collection.append((np.array([0.32, -0.023, -0.002]), 0.045, unique_name(link_name)))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection[link_name] = tmp

        link_name = "r_gripper_palm_link"
        collection = []
        collection.append((np.array([0.07, 0.02, 0.0]), 0.04, unique_name(link_name)))
        collection.append((np.array([0.06, 0.00, 0.0]), 0.04, unique_name(link_name)))
        collection.append((np.array([0.07, -0.02, 0.0]), 0.04, unique_name(link_name)))
        collection.append((np.array([0.1, 0.025, 0.0]), 0.04, unique_name(link_name)))
        collection.append((np.array([0.1, -0.025, 0.0]), 0.04, unique_name(link_name)))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection[link_name] = tmp

        link_name = "r_gripper_r_finger_link"
        collection = []
        collection.append((np.array([0.105, 0.0, 0.0]), 0.02, unique_name(link_name)))
        collection.append((np.array([0.0805, -0.01, 0.0]), 0.02, unique_name(link_name)))
        collection.append((np.array([0.06, -0.02, 0.0]), 0.03, unique_name(link_name)))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection[link_name] = tmp

        link_name = "r_gripper_l_finger_link"
        collection = []
        collection.append((np.array([0.105, 0.0, 0.0]), 0.02, unique_name(link_name)))
        collection.append((np.array([0.0805, 0.01, 0.0]), 0.02, unique_name(link_name)))
        collection.append((np.array([0.06, 0.02, 0.0]), 0.03, unique_name(link_name)))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection[link_name] = tmp

        link_name = "l_gripper_palm_link"
        collection = []
        collection.append((np.array([0.07, 0.02, 0.0]), 0.04, unique_name(link_name)))
        collection.append((np.array([0.06, 0.00, 0.0]), 0.04, unique_name(link_name)))
        collection.append((np.array([0.07, -0.02, 0.0]), 0.04, unique_name(link_name)))
        collection.append((np.array([0.1, 0.025, 0.0]), 0.04, unique_name(link_name)))
        collection.append((np.array([0.1, -0.025, 0.0]), 0.04, unique_name(link_name)))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection[link_name] = tmp

        link_name = "l_gripper_r_finger_link"
        collection = []
        collection.append((np.array([0.105, 0.0, 0.0]), 0.02, unique_name(link_name)))
        collection.append((np.array([0.0805, -0.01, 0.0]), 0.02, unique_name(link_name)))
        collection.append((np.array([0.06, -0.02, 0.0]), 0.03, unique_name(link_name)))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection[link_name] = tmp

        link_name = "l_gripper_l_finger_link"
        collection = []
        collection.append((np.array([0.105, 0.0, 0.0]), 0.02, unique_name(link_name)))
        collection.append((np.array([0.0805, 0.01, 0.0]), 0.02, unique_name(link_name)))
        collection.append((np.array([0.06, 0.02, 0.0]), 0.03, unique_name(link_name)))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection[link_name] = tmp

        link_name = "r_shoulder_pan_link"
        collection = []
        collection.append((np.array([-0.03, 0.0, -0.43]), 0.15, unique_name(link_name)))
        collection.append((np.array([-0.03, 0.0, -0.33]), 0.15, unique_name(link_name)))
        collection.append((np.array([-0.03, 0.0, -0.23]), 0.15, unique_name(link_name)))
        collection.append((np.array([-0.03, 0.0, -0.13]), 0.15, unique_name(link_name)))
        collection.append((np.array([-0.03, 0.0, -0.03]), 0.15, unique_name(link_name)))
        collection.append((np.array([-0.03, 0.0, 0.07]), 0.16, unique_name(link_name)))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection[link_name] = tmp

        link_name = "l_shoulder_pan_link"
        collection = []
        collection.append((np.array([-0.03, 0.0, -0.43]), 0.15, unique_name(link_name)))
        collection.append((np.array([-0.03, 0.0, -0.33]), 0.15, unique_name(link_name)))
        collection.append((np.array([-0.03, 0.0, -0.23]), 0.15, unique_name(link_name)))
        collection.append((np.array([-0.03, 0.0, -0.13]), 0.15, unique_name(link_name)))
        collection.append((np.array([-0.03, 0.0, -0.03]), 0.15, unique_name(link_name)))
        collection.append((np.array([-0.03, 0.0, 0.07]), 0.16, unique_name(link_name)))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection[link_name] = tmp

        h_base_sphere = 0.12
        base_link_sphere_collection = SphereCollection(
            [
                np.array([0.20, 0.20, h_base_sphere]),
                np.array([0.20, -0.20, h_base_sphere]),
                np.array([-0.20, 0.20, h_base_sphere]),
                np.array([-0.20, -0.20, h_base_sphere]),
                np.array([0.21, 0.0, h_base_sphere]),
                np.array([-0.21, 0.0, h_base_sphere]),
                np.array([0.0, 0.21, h_base_sphere]),
                np.array([0.0, -0.21, h_base_sphere]),
                np.array([0.245, 0.0, 0.26]),
            ],
            [0.16, 0.16, 0.16, 0.16, 0.15, 0.15, 0.15, 0.15, 0.07],
            ["base{}".format(i) for i in range(9)],
        )
        link_wise_sphere_collection["base_link"] = base_link_sphere_collection

        control_joint_names = self._get_control_joint_names()

        kinmap = ArticulatedCollisionKinematicsMap(
            self.urdf_path(),
            control_joint_names,
            link_wise_sphere_collection,
            base_type=self.base_type,
        )
        return kinmap

    def get_pairwise_selcol_consts(self, robot_model: PR2) -> PairWiseSelfCollFreeConst:
        # NOTE: this feature is not tested well
        colkin = self.get_collision_kin()

        rarm_group = [
            name for name in colkin.sphere_name_list if ("r_gripper" in name or "r_forearm" in name)
        ]
        larm_group = [
            name for name in colkin.sphere_name_list if ("l_gripper" in name or "l_forearm" in name)
        ]

        if self.selcol_mode == "easy":
            shoulder_sphere_names = [
                name
                for name in colkin.sphere_name_list
                if name.startswith("l_shoulder_pan_link") or name.startswith("r_shoulder_pan_link")
            ]
            anti_rarm_group_cand = shoulder_sphere_names
            anti_larm_group_cand = shoulder_sphere_names

        elif self.selcol_mode == "normal":
            anti_rarm_group_cand = [
                name for name in colkin.sphere_name_list if not name.startswith("r_")
            ]
            anti_larm_group_cand = [
                name for name in colkin.sphere_name_list if not name.startswith("l_")
            ]
        else:
            assert False

        # NOTE: we dont care larm-vs-anti if we are controlling only rarm. Same for rarm-vs-anti
        anti_rarm_group = []
        anti_larm_group = []
        if self.control_arm in ("rarm", "dual"):
            anti_rarm_group.extend(anti_rarm_group_cand)
        if self.control_arm in ("larm", "dual"):
            anti_larm_group.extend(anti_larm_group_cand)

        table = {name: fid for name, fid in zip(colkin.sphere_name_list, colkin.tinyfk_feature_ids)}
        pairs = set()
        for name1, name2 in product(rarm_group, anti_rarm_group):
            id1, id2 = table[name1], table[name2]
            pairs.add((min(id1, id2), max(id1, id2)))

        for name1, name2 in product(larm_group, anti_larm_group):
            id1, id2 = table[name1], table[name2]
            pairs.add((min(id1, id2), max(id1, id2)))

        return PairWiseSelfCollFreeConst(
            colkin, robot_model, id_pairs=list(pairs), only_closest_feature=True
        )
