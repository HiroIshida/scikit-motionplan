import copy
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
from skrobot.models import PR2
from tinyfk import BaseType

from skmp.collision import SphereCollection
from skmp.constraint import BoxConst, NeuralSelfCollFreeConst
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

        # r_upper_arm_link
        collection = []
        collection.append((np.array([0.0, 0.0, 0.0]), 0.2, str(uuid.uuid4())))
        collection.append((np.array([0.20, 0.0, 0.0]), 0.1, str(uuid.uuid4())))
        collection.append((np.array([0.26, 0.0, -0.03]), 0.082, str(uuid.uuid4())))
        collection.append((np.array([0.33, 0.0, -0.03]), 0.082, str(uuid.uuid4())))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection["r_upper_arm_link"] = tmp

        # l_upper_arm_link
        collection = []
        collection.append((np.array([0.0, 0.0, 0.0]), 0.2, str(uuid.uuid4())))
        collection.append((np.array([0.20, 0.0, 0.0]), 0.1, str(uuid.uuid4())))
        collection.append((np.array([0.26, 0.0, -0.03]), 0.082, str(uuid.uuid4())))
        collection.append((np.array([0.33, 0.0, -0.03]), 0.082, str(uuid.uuid4())))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection["l_upper_arm_link"] = tmp

        # r_forearm_link
        collection = []
        collection.append((np.array([0.0, 0.0, 0.0]), 0.1, str(uuid.uuid4())))
        collection.append((np.array([0.1, 0.0, 0.0]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.16, 0.0, 0.0]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.23, 0.02, -0.01]), 0.06, str(uuid.uuid4())))
        collection.append((np.array([0.23, -0.02, -0.01]), 0.06, str(uuid.uuid4())))
        collection.append((np.array([0.30, 0.02, 0.0]), 0.06, str(uuid.uuid4())))
        collection.append((np.array([0.30, -0.02, 0.0]), 0.06, str(uuid.uuid4())))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection["r_forearm_link"] = tmp

        # l_forearm_link
        collection = []
        collection.append((np.array([0.0, 0.0, 0.0]), 0.1, str(uuid.uuid4())))
        collection.append((np.array([0.1, 0.0, 0.0]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.16, 0.0, 0.0]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.23, 0.02, -0.01]), 0.06, str(uuid.uuid4())))
        collection.append((np.array([0.23, -0.02, -0.01]), 0.06, str(uuid.uuid4())))
        collection.append((np.array([0.30, 0.02, 0.0]), 0.06, str(uuid.uuid4())))
        collection.append((np.array([0.30, -0.02, 0.0]), 0.06, str(uuid.uuid4())))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection["l_forearm_link"] = tmp

        # r_gripper_palm_link
        collection = []
        collection.append((np.array([0.07, 0.02, 0.0]), 0.04, str(uuid.uuid4())))
        collection.append((np.array([0.06, 0.00, 0.0]), 0.04, str(uuid.uuid4())))
        collection.append((np.array([0.07, -0.02, 0.0]), 0.04, str(uuid.uuid4())))
        collection.append((np.array([0.1, 0.025, 0.0]), 0.04, str(uuid.uuid4())))
        collection.append((np.array([0.1, -0.025, 0.0]), 0.04, str(uuid.uuid4())))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection["r_gripper_palm_link"] = tmp

        collection = []
        collection.append((np.array([0.105, 0.0, 0.0]), 0.02, str(uuid.uuid4())))
        collection.append((np.array([0.0805, -0.01, 0.0]), 0.02, str(uuid.uuid4())))
        collection.append((np.array([0.06, -0.02, 0.0]), 0.03, str(uuid.uuid4())))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection["r_gripper_r_finger_link"] = tmp

        collection = []
        collection.append((np.array([0.105, 0.0, 0.0]), 0.02, str(uuid.uuid4())))
        collection.append((np.array([0.0805, 0.01, 0.0]), 0.02, str(uuid.uuid4())))
        collection.append((np.array([0.06, 0.02, 0.0]), 0.03, str(uuid.uuid4())))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection["r_gripper_l_finger_link"] = tmp

        # l_gripper_palm_link
        collection = []
        collection.append((np.array([0.07, 0.02, 0.0]), 0.04, str(uuid.uuid4())))
        collection.append((np.array([0.06, 0.00, 0.0]), 0.04, str(uuid.uuid4())))
        collection.append((np.array([0.07, -0.02, 0.0]), 0.04, str(uuid.uuid4())))
        collection.append((np.array([0.1, 0.025, 0.0]), 0.04, str(uuid.uuid4())))
        collection.append((np.array([0.1, -0.025, 0.0]), 0.04, str(uuid.uuid4())))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection["l_gripper_palm_link"] = tmp

        collection = []
        collection.append((np.array([0.105, 0.0, 0.0]), 0.02, str(uuid.uuid4())))
        collection.append((np.array([0.0805, -0.01, 0.0]), 0.02, str(uuid.uuid4())))
        collection.append((np.array([0.06, -0.02, 0.0]), 0.03, str(uuid.uuid4())))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection["l_gripper_r_finger_link"] = tmp

        collection = []
        collection.append((np.array([0.105, 0.0, 0.0]), 0.02, str(uuid.uuid4())))
        collection.append((np.array([0.0805, 0.01, 0.0]), 0.02, str(uuid.uuid4())))
        collection.append((np.array([0.06, 0.02, 0.0]), 0.03, str(uuid.uuid4())))
        tmp = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection["l_gripper_l_finger_link"] = tmp

        r_shoulder_collection = SphereCollection(
            [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -0.25]), np.array([0.0, 0.0, -0.4])],
            [0.2, 0.15, 0.15],
            ["r_shoudler{}".format(i) for i in range(3)],
        )

        l_shoulder_collection = SphereCollection(
            [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -0.25]), np.array([0.0, 0.0, -0.4])],
            [0.2, 0.15, 0.15],
            ["l_shoudler{}".format(i) for i in range(3)],
        )
        link_wise_sphere_collection["r_shoulder_pan_link"] = r_shoulder_collection
        link_wise_sphere_collection["l_shoulder_pan_link"] = l_shoulder_collection

        base_link_sphere_collection = SphereCollection(
            [
                np.array([0.23, 0.23, 0.1]),
                np.array([0.23, -0.23, 0.1]),
                np.array([0.16, 0.0, 0.1]),
                np.array([-0.23, 0.23, 0.1]),
                np.array([-0.23, -0.23, 0.1]),
                np.array([-0.16, 0.0, 0.1]),
                np.array([0.0, 0.16, 0.1]),
                np.array([0.0, -0.16, 0.1]),
            ],
            [0.15, 0.15, 0.23, 0.15, 0.15, 0.23, 0.23, 0.23],
            ["base{}".format(i) for i in range(8)],
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

    def get_neural_selcol_const(self, robot_model: PR2) -> NeuralSelfCollFreeConst:
        return NeuralSelfCollFreeConst.load(
            self.urdf_path(), self._get_control_joint_names(), robot_model, self.base_type
        )
