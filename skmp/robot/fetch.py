from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

import pkg_resources
from skrobot.model import Link
from skrobot.model.primitives import Box, Cylinder
from skrobot.models import Fetch
from tinyfk import BaseType

from skmp.constraint import BoxConst, FCLSelfCollFreeConst
from skmp.kinematics import ArticulatedCollisionKinematicsMap
from skmp.robot.utils import load_collision_spheres


@dataclass
class FetchConfig:
    base_type: BaseType = BaseType.FIXED
    use_torso: bool = True

    @classmethod
    def urdf_path(cls) -> Path:
        return Path("~/.skrobot/fetch_description/fetch.urdf").expanduser()

    def get_control_joint_names(self) -> List[str]:
        joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]

        if self.use_torso:
            joint_names = ["torso_lift_joint"] + joint_names
        return joint_names

    def get_box_const(self, eps: float = 1e-4) -> BoxConst:
        # set eps to satisfy that default postion is in the bounds
        bounds = BoxConst.from_urdf(self.urdf_path(), self.get_control_joint_names())
        bounds.lb -= eps
        bounds.ub += eps
        return bounds

    def get_collision_kin(self) -> ArticulatedCollisionKinematicsMap:
        collision_config_path = pkg_resources.resource_filename(
            "skmp", "robot/fetch_coll_spheres.yaml"
        )
        link_wise_sphere_collection = load_collision_spheres(collision_config_path)
        control_joint_names = self.get_control_joint_names()
        kinmap = ArticulatedCollisionKinematicsMap(
            self.urdf_path(),
            control_joint_names,
            link_wise_sphere_collection,
            base_type=self.base_type,
        )
        return kinmap

    def get_self_body_obstacles(self) -> List[Link]:
        base = Box([0.57, 0.55, 0.32], face_colors=[255, 255, 255, 200])
        base.translate([0.005, 0.0, 0.2])
        torso = Box([0.16, 0.16, 1.0], face_colors=[255, 255, 255, 200])
        torso.translate([-0.12, 0.0, 0.5])

        neck_lower = Box([0.1, 0.18, 0.08], face_colors=[255, 255, 255, 200])
        neck_lower.translate([0.0, 0.0, 0.97])
        neck_upper = Box([0.05, 0.17, 0.15], face_colors=[255, 255, 255, 200])
        neck_upper.translate([-0.035, 0.0, 0.92])

        torso_left = Cylinder(0.1, 1.0, face_colors=[255, 255, 255, 200])
        torso_left.translate([-0.143, 0.09, 0.5])
        torso_right = Cylinder(0.1, 1.0, face_colors=[255, 255, 255, 200])
        torso_right.translate([-0.143, -0.09, 0.5])

        head = Cylinder(0.28, 0.12, face_colors=[255, 255, 255, 200])
        head.translate([0.0, 0.0, 1.04])
        self_body_obstacles = [base, torso, neck_lower, neck_upper, torso_left, torso_right, head]
        return self_body_obstacles

    def get_selcol_consts(self, robot_model: Fetch):
        arm_links = [
            "shoulder_pan_link",
            "shoulder_lift_link",
            "upperarm_roll_link",
            "elbow_flex_link",
            "forearm_roll_link",
            "wrist_flex_link",
            "wrist_roll_link",
            "gripper_link",
            "r_gripper_finger_link",
            "l_gripper_finger_link",
        ]
        return FCLSelfCollFreeConst(
            robot_model, arm_links, self.get_control_joint_names(), self.ignore_pairs
        )

    @property
    def ignore_pairs(self) -> Set[Tuple[str, str]]:
        pairs = {
            ("shoulder_lift_link", "wrist_roll_link"),
            ("elbow_flex_link", "upperarm_roll_link"),
            ("base_link", "bellows_link2"),
            ("forearm_roll_link", "l_wheel_link"),
            ("estop_link", "forearm_roll_link"),
            ("bellows_link2", "shoulder_pan_link"),
            ("head_tilt_link", "shoulder_lift_link"),
            ("shoulder_lift_link", "shoulder_pan_link"),
            ("bellows_link2", "upperarm_roll_link"),
            ("forearm_roll_link", "l_gripper_finger_link"),
            ("base_link", "torso_fixed_link"),
            ("base_link", "estop_link"),
            ("head_tilt_link", "r_wheel_link"),
            ("estop_link", "l_wheel_link"),
            ("head_pan_link", "laser_link"),
            ("estop_link", "head_pan_link"),
            ("r_wheel_link", "torso_fixed_link"),
            ("l_wheel_link", "torso_fixed_link"),
            ("shoulder_lift_link", "upperarm_roll_link"),
            ("laser_link", "torso_fixed_link"),
            ("bellows_link2", "torso_lift_link"),
            ("torso_fixed_link", "torso_lift_link"),
            ("estop_link", "torso_fixed_link"),
            ("l_gripper_finger_link", "upperarm_roll_link"),
            ("head_pan_link", "shoulder_pan_link"),
            ("elbow_flex_link", "r_wheel_link"),
            ("forearm_roll_link", "wrist_roll_link"),
            ("elbow_flex_link", "r_gripper_finger_link"),
            ("base_link", "l_wheel_link"),
            ("gripper_link", "wrist_flex_link"),
            ("base_link", "head_pan_link"),
            ("base_link", "laser_link"),
            ("bellows_link2", "shoulder_lift_link"),
            ("l_wheel_link", "wrist_flex_link"),
            ("elbow_flex_link", "estop_link"),
            ("r_wheel_link", "wrist_roll_link"),
            ("gripper_link", "wrist_roll_link"),
            ("l_wheel_link", "laser_link"),
            ("l_wheel_link", "wrist_roll_link"),
            ("head_tilt_link", "l_wheel_link"),
            ("bellows_link2", "r_wheel_link"),
            ("l_wheel_link", "torso_lift_link"),
            ("elbow_flex_link", "forearm_roll_link"),
            ("forearm_roll_link", "shoulder_pan_link"),
            ("bellows_link2", "head_tilt_link"),
            ("forearm_roll_link", "shoulder_lift_link"),
            ("estop_link", "laser_link"),
            ("estop_link", "wrist_roll_link"),
            ("shoulder_pan_link", "torso_fixed_link"),
            ("head_pan_link", "torso_lift_link"),
            ("estop_link", "torso_lift_link"),
            ("gripper_link", "l_gripper_finger_link"),
            ("head_tilt_link", "torso_fixed_link"),
            ("elbow_flex_link", "gripper_link"),
            ("forearm_roll_link", "upperarm_roll_link"),
            ("elbow_flex_link", "wrist_flex_link"),
            ("l_wheel_link", "shoulder_pan_link"),
            ("base_link", "upperarm_roll_link"),
            ("elbow_flex_link", "l_wheel_link"),
            ("l_gripper_finger_link", "r_gripper_finger_link"),
            ("estop_link", "shoulder_pan_link"),
            ("elbow_flex_link", "wrist_roll_link"),
            ("head_pan_link", "shoulder_lift_link"),
            ("r_wheel_link", "upperarm_roll_link"),
            ("gripper_link", "upperarm_roll_link"),
            ("estop_link", "shoulder_lift_link"),
            ("l_wheel_link", "upperarm_roll_link"),
            ("shoulder_pan_link", "wrist_flex_link"),
            ("laser_link", "upperarm_roll_link"),
            ("base_link", "torso_lift_link"),
            ("estop_link", "upperarm_roll_link"),
            ("head_pan_link", "r_wheel_link"),
            ("elbow_flex_link", "l_gripper_finger_link"),
            ("r_wheel_link", "torso_lift_link"),
            ("head_pan_link", "head_tilt_link"),
            ("estop_link", "head_tilt_link"),
            ("head_tilt_link", "laser_link"),
            ("laser_link", "torso_lift_link"),
            ("base_link", "shoulder_pan_link"),
            ("base_link", "shoulder_lift_link"),
            ("l_gripper_finger_link", "wrist_flex_link"),
            ("bellows_link2", "torso_fixed_link"),
            ("r_gripper_finger_link", "wrist_flex_link"),
            ("bellows_link2", "estop_link"),
            ("r_wheel_link", "shoulder_pan_link"),
            ("wrist_flex_link", "wrist_roll_link"),
            ("r_wheel_link", "shoulder_lift_link"),
            ("l_gripper_finger_link", "l_wheel_link"),
            ("l_wheel_link", "shoulder_lift_link"),
            ("laser_link", "shoulder_pan_link"),
            ("forearm_roll_link", "r_gripper_finger_link"),
            ("head_tilt_link", "shoulder_pan_link"),
            ("l_gripper_finger_link", "wrist_roll_link"),
            ("laser_link", "shoulder_lift_link"),
            ("r_gripper_finger_link", "wrist_roll_link"),
            ("base_link", "r_wheel_link"),
            ("shoulder_lift_link", "torso_fixed_link"),
            ("base_link", "head_tilt_link"),
            ("l_wheel_link", "r_wheel_link"),
            ("upperarm_roll_link", "wrist_flex_link"),
            ("gripper_link", "r_gripper_finger_link"),
            ("laser_link", "r_wheel_link"),
            ("l_wheel_link", "r_gripper_finger_link"),
            ("head_pan_link", "l_wheel_link"),
            ("laser_link", "r_gripper_finger_link"),
            ("estop_link", "r_wheel_link"),
            ("upperarm_roll_link", "wrist_roll_link"),
            ("bellows_link2", "l_wheel_link"),
            ("shoulder_pan_link", "torso_lift_link"),
            ("bellows_link2", "head_pan_link"),
            ("estop_link", "r_gripper_finger_link"),
            ("bellows_link2", "laser_link"),
            ("elbow_flex_link", "shoulder_pan_link"),
            ("elbow_flex_link", "shoulder_lift_link"),
            ("shoulder_lift_link", "wrist_flex_link"),
            ("head_tilt_link", "torso_lift_link"),
            ("forearm_roll_link", "gripper_link"),
            ("head_pan_link", "torso_fixed_link"),
            ("forearm_roll_link", "wrist_flex_link"),
            ("r_gripper_finger_link", "upperarm_roll_link"),
        }
        return pairs
