import copy
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pkg_resources
from skrobot.models import PR2
from tinyfk import BaseType, RotationType

from skmp.constraint import BoxConst, PairWiseSelfCollFreeConst
from skmp.kinematics import CollSphereKinematicsMap, EndEffectorKinematicsMap
from skmp.robot.utils import load_collision_spheres


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
        # !!solely for backward compatibility
        return self.get_control_joint_names()

    def get_control_joint_names(self) -> List[str]:
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

    def get_endeffector_kin(
        self, rot_type: RotationType = RotationType.RPY
    ) -> EndEffectorKinematicsMap:
        kinmap = EndEffectorKinematicsMap(
            self.urdf_path(),
            self.get_control_joint_names(),
            self._get_endeffector_names(),
            base_type=self.base_type,
            rot_type=rot_type,
        )
        return kinmap

    def get_default_motion_step_box(self) -> np.ndarray:
        table = {
            "r_shoulder_pan_joint": 0.05,
            "r_shoulder_lift_joint": 0.05,
            "r_upper_arm_roll_joint": 0.1,
            "r_elbow_flex_joint": 0.1,
            "r_forearm_roll_joint": 0.2,
            "r_wrist_flex_joint": 0.2,
            "r_wrist_roll_joint": 0.2,
            "l_shoulder_pan_joint": 0.05,
            "l_shoulder_lift_joint": 0.05,
            "l_upper_arm_roll_joint": 0.1,
            "l_elbow_flex_joint": 0.1,
            "l_forearm_roll_joint": 0.2,
            "l_wrist_flex_joint": 0.2,
            "l_wrist_roll_joint": 0.2,
            "torso_lift_joint": 0.05,  # 5cm
        }
        joint_names = self.get_control_joint_names()
        motion_step_box = [table[joint_name] for joint_name in joint_names]
        if self.base_type == BaseType.FLOATING:
            motion_step_box += [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        elif self.base_type == BaseType.PLANER:
            motion_step_box += [0.05, 0.05, 0.05]
        else:
            assert self.base_type == BaseType.FIXED
        return np.array(motion_step_box)

    def get_box_const(self, base_bound: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> BoxConst:
        if self.base_type == BaseType.PLANER:
            if base_bound is None:
                base_bound = np.array([-1.0, -2.0, -1.0]), np.array([2.0, 2.0, 1.0])
        elif self.base_type == BaseType.FLOATING:
            if base_bound is None:
                base_bound = -np.ones(6), np.ones(6)
        else:
            assert self.base_type == BaseType.FIXED
            assert base_bound is None
        lb_list_, ub_list_ = self._get_box_const_without_base(tuple(self.get_control_joint_names()))
        lb_list = copy.deepcopy(lb_list_)
        ub_list = copy.deepcopy(ub_list_)
        if base_bound is not None:
            lb, ub = base_bound
            n_dof_base = len(lb)
            assert n_dof_base in (3, 6)
            for i in range(n_dof_base):
                lb_list.append(lb[i])
                ub_list.append(ub[i])
        box_const = BoxConst(np.array(lb_list), np.array(ub_list))
        return box_const

    @staticmethod
    @lru_cache
    def _get_box_const_without_base(joint_names) -> Tuple[List[float], List[float]]:
        # this quite slow
        pr2 = PR2()
        lb_list = []
        ub_list = []
        for joint_name in joint_names:
            joint = pr2.__dict__[joint_name]
            min_angle = joint.min_angle
            max_angle = joint.max_angle
            if not np.isfinite(min_angle):
                min_angle = -2 * np.pi
            if not np.isfinite(max_angle):
                max_angle = 2 * np.pi
            lb_list.append(min_angle)
            ub_list.append(max_angle)
        return lb_list, ub_list

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

    def get_collision_kin(self, whole_body: bool = False) -> CollSphereKinematicsMap:
        collision_config_path = pkg_resources.resource_filename(
            "skmp", "robot/pr2_coll_spheres.yaml"
        )
        link_wise_sphere_collection = load_collision_spheres(collision_config_path)

        if self.base_type != BaseType.FIXED:
            whole_body = True

        if not whole_body:
            # remove irrelevant spheres
            if self.control_arm == "rarm":
                for key in list(link_wise_sphere_collection.keys()):
                    if key.startswith("l_"):
                        del link_wise_sphere_collection[key]

            if self.control_arm == "larm":
                for key in list(link_wise_sphere_collection.keys()):
                    if key.startswith("r_"):
                        del link_wise_sphere_collection[key]

            if self.base_type == BaseType.FIXED:
                for key in list(link_wise_sphere_collection.keys()):
                    if key.startswith("base_link"):
                        del link_wise_sphere_collection[key]

        control_joint_names = self.get_control_joint_names()

        kinmap = CollSphereKinematicsMap(
            self.urdf_path(),
            control_joint_names,
            link_wise_sphere_collection,
            base_type=self.base_type,
        )
        return kinmap

    def get_pairwise_selcol_consts(self, robot_model: PR2) -> PairWiseSelfCollFreeConst:
        # NOTE: this feature is not tested well
        colkin = self.get_collision_kin(whole_body=True)

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
            base_link_sphere_names = [
                name for name in colkin.sphere_name_list if name.startswith("base_link")
            ]
            anti_rarm_group_cand = shoulder_sphere_names + base_link_sphere_names
            anti_larm_group_cand = shoulder_sphere_names + base_link_sphere_names

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
