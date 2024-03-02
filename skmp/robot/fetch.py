from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

from skrobot.models import Fetch

from skmp.constraint import BoxConst, FCLSelfCollFreeConst


@dataclass
class FetchConfig:
    with_torso: bool = True

    @classmethod
    def urdf_path(cls) -> Path:
        return Path("~/.skrobot/fetch_description/fetch.urdf").expanduser()

    @property
    def joint_names(self) -> List[str]:
        joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]

        if self.with_torso:
            joint_names = ["torso_lift_joint"] + joint_names
        return joint_names

    def get_box_const(self, eps: float = 1e-4) -> BoxConst:
        # set eps to satisfy that default postion is in the bounds
        bounds = BoxConst.from_urdf(self.urdf_path(), self.joint_names)
        bounds.lb -= eps
        bounds.ub += eps
        return bounds

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
        return FCLSelfCollFreeConst(robot_model, arm_links, self.joint_names, self.ignore_pairs)

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
