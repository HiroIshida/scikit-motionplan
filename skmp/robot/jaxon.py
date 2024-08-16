import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from robot_descriptions.jaxon_description import URDF_PATH as JAXON_URDF_PATH
from skrobot.coordinates import CascadedCoords
from skrobot.coordinates.math import rotation_matrix, rpy_angle
from skrobot.model.primitives import Box
from skrobot.models.urdf import RobotModelFromURDF
from tinyfk import BaseType, KinematicModel, RotationType

from skmp.collision import SphereCollection
from skmp.constraint import BoxConst, COMStabilityConst, NeuralSelfCollFreeConst
from skmp.kinematics import (
    AttachedObstacleCollPointsKinematicsMap,
    CollSphereKinematicsMap,
    EndEffectorKinematicsMap,
)


class Jaxon(RobotModelFromURDF):
    rarm_end_coords: CascadedCoords
    larm_end_coords: CascadedCoords
    rleg_end_coords: CascadedCoords
    lleg_end_coords: CascadedCoords

    def __init__(self):
        super().__init__(urdf_file=JAXON_URDF_PATH)
        matrix = rotation_matrix(np.pi * 0.5, [0, 0, 1.0])

        self.rarm_end_coords = CascadedCoords(self.RARM_LINK7, name="rarm_end_coords")
        self.rarm_end_coords.translate([0, 0, -0.220])
        self.rarm_end_coords.rotate_with_matrix(matrix, wrt="local")

        self.rarm_tip_coords = CascadedCoords(self.RARM_LINK7, name="rarm_end_coords")
        self.rarm_tip_coords.translate([0, 0, -0.3])
        self.rarm_tip_coords.rotate_with_matrix(matrix, wrt="local")

        self.larm_end_coords = CascadedCoords(self.LARM_LINK7, name="larm_end_coords")
        self.larm_end_coords.translate([0, 0, -0.220])
        self.larm_end_coords.rotate_with_matrix(matrix, wrt="local")

        self.rleg_end_coords = CascadedCoords(self.RLEG_LINK5, name="rleg_end_coords")
        self.rleg_end_coords.translate([0, 0, -0.1])

        self.lleg_end_coords = CascadedCoords(self.LLEG_LINK5, name="lleg_end_coords")
        self.lleg_end_coords.translate([0, 0, -0.1])

    def default_urdf_path(self):
        return JAXON_URDF_PATH

    def reset_manip_pose(self):
        angle_table = {
            "RLEG": [0.0, 0.0, -0.349066, 0.698132, -0.349066, 0.0],
            "LLEG": [0.0, 0.0, -0.349066, 0.698132, -0.349066, 0.0],
            "CHEST": [0.0, 0.0, 0.0],
            "RARM": [0.0, 0.959931, -0.349066, -0.261799, -1.74533, -0.436332, 0.0, -0.785398],
            "LARM": [0.0, 0.959931, 0.349066, 0.261799, -1.74533, 0.436332, 0.0, -0.785398],
        }
        for key, values in angle_table.items():
            for i, angle in enumerate(values):
                self.__dict__["{}_JOINT{}".format(key, i)].joint_angle(angle)


@dataclass
class JaxonConfig:
    @classmethod
    def urdf_path(cls) -> Path:
        return Path(JAXON_URDF_PATH)

    @staticmethod
    def add_end_coords(kin: KinematicModel) -> None:
        rarm_id, larm_id = kin.get_link_ids(["RARM_LINK7", "LARM_LINK7"])
        matrix = rotation_matrix(np.pi * 0.5, [0, 0, 1.0])
        rpy = np.flip(rpy_angle(matrix)[0])
        kin.add_new_link("rarm_end_coords", rarm_id, np.array([0, 0, -0.220]), rpy=rpy)
        kin.add_new_link("larm_end_coords", larm_id, np.array([0, 0, -0.220]), rpy=rpy)
        kin.add_new_link("rarm_tip_coords", rarm_id, np.array([0, 0, -0.30]), rpy=rpy)

        rleg_id, lleg_id = kin.get_link_ids(["RLEG_LINK5", "LLEG_LINK5"])
        kin.add_new_link("rleg_end_coords", rleg_id, np.array([0, 0, -0.1]))
        kin.add_new_link("lleg_end_coords", lleg_id, np.array([0, 0, -0.1]))

    def _get_control_joint_names(self) -> List[str]:
        joint_names = []
        for i in range(8):
            joint_names.append("RARM_JOINT{}".format(i))
            joint_names.append("LARM_JOINT{}".format(i))
        for i in range(6):
            joint_names.append("RLEG_JOINT{}".format(i))
            joint_names.append("LLEG_JOINT{}".format(i))
        for i in range(3):
            joint_names.append("CHEST_JOINT{}".format(i))
        return joint_names

    def get_endeffector_kin(
        self,
        rleg: bool = True,
        lleg: bool = True,
        rarm: bool = True,
        larm: bool = True,
        rot_type: RotationType = RotationType.XYZW,
    ):
        endeffector_names = []
        if rleg:
            endeffector_names.append("rleg_end_coords")
        if lleg:
            endeffector_names.append("lleg_end_coords")
        if rarm:
            endeffector_names.append("rarm_end_coords")
        if larm:
            endeffector_names.append("larm_end_coords")

        kinmap = EndEffectorKinematicsMap(
            self.urdf_path(),
            self._get_control_joint_names(),
            endeffector_names,
            base_type=BaseType.FLOATING,
            rot_type=rot_type,
            fksolver_init_hook=self.add_end_coords,
        )
        return kinmap

    def get_attached_obstacle_kin(
        self, relative_position: np.ndarray, shape: Box
    ) -> AttachedObstacleCollPointsKinematicsMap:
        kinmap = AttachedObstacleCollPointsKinematicsMap(
            self.urdf_path(),
            self._get_control_joint_names(),
            "rarm_end_coords",
            relative_position,
            shape,
            base_type=BaseType.FLOATING,
            fksolver_init_hook=self.add_end_coords,
        )
        return kinmap

    def get_collision_kin(
        self, rsole: bool = True, lsole: bool = True, rgripper: bool = True, lgripper: bool = True
    ) -> CollSphereKinematicsMap:
        link_wise_sphere_collection: Dict[str, SphereCollection] = {}

        collision_link_names = []

        if rsole:
            link_name = "RLEG_LINK5"
            collection = []
            collection.append((np.array([-0.08, 0.035, -0.075]), 0.04, str(uuid.uuid4())))
            collection.append((np.array([-0.08, -0.05, -0.075]), 0.04, str(uuid.uuid4())))
            collection.append((np.array([0.105, 0.035, -0.075]), 0.04, str(uuid.uuid4())))
            collection.append((np.array([0.105, -0.05, -0.075]), 0.04, str(uuid.uuid4())))
            collection.append((np.array([-0.0, 0.0, -0.075]), 0.04, str(uuid.uuid4())))
            collection.append((np.array([0.0, 0.0, 0.0]), 0.07, str(uuid.uuid4())))
            link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
            collision_link_names.append(link_name)

        if lsole:
            link_name = "LLEG_LINK5"
            collection = []
            collection.append((np.array([-0.08, -0.035, -0.075]), 0.04, str(uuid.uuid4())))
            collection.append((np.array([-0.08, +0.05, -0.075]), 0.04, str(uuid.uuid4())))
            collection.append((np.array([0.105, -0.035, -0.075]), 0.04, str(uuid.uuid4())))
            collection.append((np.array([0.105, +0.05, -0.075]), 0.04, str(uuid.uuid4())))
            collection.append((np.array([-0.0, 0.0, -0.075]), 0.04, str(uuid.uuid4())))
            collection.append((np.array([0.0, 0.0, 0.0]), 0.07, str(uuid.uuid4())))
            link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
            collision_link_names.append(link_name)

        # leg link3
        link_name = "RLEG_LINK3"
        collection = []
        collection.append((np.array([0.0, -0.015, 0.0]), 0.09, str(uuid.uuid4())))
        collection.append((np.array([0.02, -0.015, -0.07]), 0.09, str(uuid.uuid4())))
        collection.append((np.array([0.02, -0.015, -0.14]), 0.09, str(uuid.uuid4())))
        collection.append((np.array([0.02, -0.015, -0.21]), 0.09, str(uuid.uuid4())))
        collection.append((np.array([0.0, -0.015, -0.3]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.0, -0.015, -0.35]), 0.08, str(uuid.uuid4())))
        link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
        collision_link_names.append(link_name)

        link_name = "LLEG_LINK3"
        collection = []
        collection.append((np.array([0.0, +0.015, 0.0]), 0.09, str(uuid.uuid4())))
        collection.append((np.array([0.02, +0.015, -0.07]), 0.09, str(uuid.uuid4())))
        collection.append((np.array([0.02, +0.015, -0.14]), 0.09, str(uuid.uuid4())))
        collection.append((np.array([0.02, +0.015, -0.21]), 0.09, str(uuid.uuid4())))
        collection.append((np.array([0.0, +0.015, -0.3]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.0, +0.015, -0.35]), 0.08, str(uuid.uuid4())))
        link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
        collision_link_names.append(link_name)

        # leg link2
        link_name = "RLEG_LINK2"
        collection = []
        collection.append((np.array([-0.02, -0.01, 0.0]), 0.12, str(uuid.uuid4())))
        collection.append((np.array([-0.02, -0.01, -0.08]), 0.12, str(uuid.uuid4())))
        collection.append((np.array([0.02, -0.01, -0.16]), 0.12, str(uuid.uuid4())))
        collection.append((np.array([0.02, -0.02, -0.24]), 0.1, str(uuid.uuid4())))
        collection.append((np.array([0.02, -0.02, -0.32]), 0.1, str(uuid.uuid4())))
        link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
        collision_link_names.append(link_name)

        link_name = "LLEG_LINK2"
        collection = []
        collection.append((np.array([-0.02, +0.01, 0.0]), 0.12, str(uuid.uuid4())))
        collection.append((np.array([-0.02, +0.01, -0.08]), 0.12, str(uuid.uuid4())))
        collection.append((np.array([0.02, +0.01, -0.16]), 0.12, str(uuid.uuid4())))
        collection.append((np.array([0.02, +0.02, -0.24]), 0.1, str(uuid.uuid4())))
        collection.append((np.array([0.02, +0.02, -0.32]), 0.1, str(uuid.uuid4())))
        link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
        collision_link_names.append(link_name)

        # arm finger
        if rgripper:
            link_name = "RARM_FINGER0"
            collection = []
            collection.append((np.array([0.04, 0.01, 0.0]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.06, 0.02, 0.0]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.1, 0.02, 0.0]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.14, 0.01, 0.0]), 0.02, str(uuid.uuid4())))
            link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
            collision_link_names.append(link_name)

            link_name = "RARM_FINGER1"
            collection = []
            collection.append((np.array([0.04, -0.01, 0.02]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.06, -0.02, 0.02]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.1, -0.02, 0.02]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.14, -0.01, 0.02]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.04, -0.01, -0.02]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.06, -0.02, -0.02]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.1, -0.02, -0.02]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.14, -0.01, -0.02]), 0.02, str(uuid.uuid4())))
            link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
            collision_link_names.append(link_name)

            link_name = "RARM_LINK7"
            collection = []
            collection.append((np.array([0.0, 0.04, -0.16]), 0.06, str(uuid.uuid4())))
            collection.append((np.array([0.0, -0.04, -0.16]), 0.06, str(uuid.uuid4())))
            collection.append((np.array([0.0, 0.0, -0.10]), 0.06, str(uuid.uuid4())))
            collection.append((np.array([0.03, 0.0, -0.15]), 0.06, str(uuid.uuid4())))
            link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
            collision_link_names.append(link_name)

        if lgripper:
            link_name = "LARM_FINGER0"
            collection = []
            collection.append((np.array([0.04, 0.01, 0.0]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.06, 0.02, 0.0]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.1, 0.02, 0.0]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.14, 0.01, 0.0]), 0.02, str(uuid.uuid4())))
            link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
            collision_link_names.append(link_name)

            link_name = "LARM_FINGER1"
            collection = []
            collection.append((np.array([0.04, -0.01, 0.02]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.06, -0.02, 0.02]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.1, -0.02, 0.02]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.14, -0.01, 0.02]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.04, -0.01, -0.02]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.06, -0.02, -0.02]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.1, -0.02, -0.02]), 0.02, str(uuid.uuid4())))
            collection.append((np.array([0.14, -0.01, -0.02]), 0.02, str(uuid.uuid4())))
            link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
            collision_link_names.append(link_name)

            link_name = "LARM_LINK7"
            collection = []
            collection.append((np.array([0.0, 0.04, -0.16]), 0.06, str(uuid.uuid4())))
            collection.append((np.array([0.0, -0.04, -0.16]), 0.06, str(uuid.uuid4())))
            collection.append((np.array([0.0, 0.0, -0.10]), 0.06, str(uuid.uuid4())))
            collection.append((np.array([0.03, 0.0, -0.15]), 0.06, str(uuid.uuid4())))
            link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
            collision_link_names.append(link_name)

        link_name = "RARM_LINK5"
        collection = []
        collection.append((np.array([0.0, 0.0, +0.14]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.0, 0.0, +0.04]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.0, 0.0, -0.06]), 0.09, str(uuid.uuid4())))
        collection.append((np.array([0.0, 0.0, -0.16]), 0.09, str(uuid.uuid4())))
        link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
        collision_link_names.append(link_name)

        link_name = "LARM_LINK5"
        collection = []
        collection.append((np.array([0.0, 0.0, +0.14]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.0, 0.0, +0.04]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.0, 0.0, -0.06]), 0.09, str(uuid.uuid4())))
        collection.append((np.array([0.0, 0.0, -0.16]), 0.09, str(uuid.uuid4())))
        link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
        collision_link_names.append(link_name)

        link_name = "RARM_LINK3"
        collection = []
        collection.append((np.array([0.0, 0.0, +0.16]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.0, 0.0, +0.06]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.0, 0.0, -0.04]), 0.08, str(uuid.uuid4())))
        link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
        collision_link_names.append(link_name)

        link_name = "LARM_LINK3"
        collection = []
        collection.append((np.array([0.0, 0.0, +0.16]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.0, 0.0, +0.06]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.0, 0.0, -0.04]), 0.08, str(uuid.uuid4())))
        link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
        collision_link_names.append(link_name)

        link_name = "CHEST_LINK2"
        collection = []
        collection.append((np.array([-0.06, 0.0, -0.05]), 0.25, str(uuid.uuid4())))
        collection.append((np.array([-0.17, 0.0, -0.03]), 0.25, str(uuid.uuid4())))
        collection.append((np.array([-0.08, 0.0, -0.27]), 0.25, str(uuid.uuid4())))
        link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
        collision_link_names.append(link_name)

        link_name = "HEAD_LINK0"
        collection = []
        collection.append((np.array([0.0, 0.0, 0.1]), 0.15, str(uuid.uuid4())))
        collection.append((np.array([0.0, 0.0, 0.2]), 0.15, str(uuid.uuid4())))
        link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
        collision_link_names.append(link_name)

        kinmap = CollSphereKinematicsMap(
            self.urdf_path(),
            self._get_control_joint_names(),
            link_wise_sphere_collection,
            base_type=BaseType.FLOATING,
        )
        return kinmap

    def get_box_const(self) -> BoxConst:
        base_bounds = np.array([-1.0, -1.0, 0.0, -1.0, -1.0, -1.0]), np.array(
            [2.0, 1.0, 3.0, 1.0, 1.0, 1.0]
        )
        bounds = BoxConst.from_urdf(
            self.urdf_path(), self._get_control_joint_names(), base_bounds=base_bounds
        )
        return bounds

    def get_close_box_const(
        self,
        q: Optional[np.ndarray] = None,
        joint_margin: float = 1.0,
        base_pos_margin: float = 0.8,
        base_rot_margin: float = 1.0,
    ) -> BoxConst:
        box_const = self.get_box_const()
        if q is None:
            return box_const
        q_max = np.zeros(q.shape)
        q_min = np.zeros(q.shape)

        slices = [slice(None, -6), slice(-6, -3), slice(-3, None)]
        margins = [joint_margin, base_pos_margin, base_rot_margin]
        do_clips = [True, False, True]
        for slice_indices, margin, do_clip in zip(slices, margins, do_clips):
            if do_clip:
                q_max[slice_indices] = np.minimum(
                    q[slice_indices] + margin, box_const.ub[slice_indices]
                )
                q_min[slice_indices] = np.maximum(
                    q[slice_indices] - margin, box_const.lb[slice_indices]
                )
            else:
                q_max[slice_indices] = q[slice_indices] + margin
                q_min[slice_indices] = q[slice_indices] - margin
        return BoxConst(q_min, q_max)

    def get_motion_step_box(self) -> np.ndarray:
        name_to_width_table = {jn: 0.2 for jn in self._get_control_joint_names()}

        name_to_width_table["RARM_JOINT0"] = 0.1
        name_to_width_table["RARM_JOINT1"] = 0.1
        name_to_width_table["RARM_JOINT2"] = 0.1

        name_to_width_table["LARM_JOINT0"] = 0.1
        name_to_width_table["LARM_JOINT1"] = 0.1
        name_to_width_table["LARM_JOINT2"] = 0.1

        name_to_width_table["RLEG_JOINT0"] = 0.1
        name_to_width_table["RLEG_JOINT1"] = 0.1
        name_to_width_table["RLEG_JOINT2"] = 0.1

        name_to_width_table["LLEG_JOINT0"] = 0.1
        name_to_width_table["LLEG_JOINT1"] = 0.1
        name_to_width_table["LLEG_JOINT2"] = 0.1

        name_to_width_table["CHEST_JOINT0"] = 0.1
        name_to_width_table["CHEST_JOINT1"] = 0.1
        name_to_width_table["CHEST_JOINT2"] = 0.1

        joint_step_width = np.array(list(name_to_width_table.values()))
        base_step_width = np.ones(6) * 0.1
        return np.hstack([joint_step_width, base_step_width])

    def get_neural_selcol_const(self, robot_model: Jaxon) -> NeuralSelfCollFreeConst:
        return NeuralSelfCollFreeConst.load(
            self.urdf_path(), self._get_control_joint_names(), robot_model, BaseType.FLOATING
        )

    def get_com_stability_const(
        self,
        robot_model: Jaxon,
        com_box: Box,
        action_link_names: Optional[List[str]] = None,
        action_forces: Optional[List[float]] = None,
    ) -> COMStabilityConst:
        fksolver = KinematicModel(self.urdf_path())
        fksolver.get_joint_ids(self._get_control_joint_names())

        const = COMStabilityConst(
            self.urdf_path(),
            self._get_control_joint_names(),
            BaseType.FLOATING,
            robot_model,
            com_box,
            action_link_names,
            action_forces,
        )
        return const
