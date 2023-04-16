import copy
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from robot_descriptions.jaxon_description import URDF_PATH as JAXON_URDF_PATH
from skrobot.coordinates import CascadedCoords
from skrobot.coordinates.math import rotation_matrix, rpy_angle
from skrobot.models.urdf import RobotModelFromURDF
from tinyfk import BaseType, RobotModel, RotationType

from skmp.collision import SphereCollection
from skmp.constraint import BoxConst, NeuralSelfCollFreeConst
from skmp.kinematics import (
    ArticulatedCollisionKinematicsMap,
    ArticulatedEndEffectorKinematicsMap,
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

        self.larm_end_coords = CascadedCoords(self.LARM_LINK7, name="larm_end_coords")
        self.larm_end_coords.translate([0, 0, -0.220])
        self.larm_end_coords.rotate_with_matrix(matrix, wrt="local")

        self.rleg_end_coords = CascadedCoords(self.RLEG_LINK5, name="rleg_end_coords")
        self.rleg_end_coords.translate([0, 0, -0.1])

        self.lleg_end_coords = CascadedCoords(self.LLEG_LINK5, name="lleg_end_coords")
        self.lleg_end_coords.translate([0, 0, -0.1])

    def default_urdf_path(self):
        return JAXON_URDF_PATH


@dataclass
class JaxonConfig:
    @classmethod
    def urdf_path(cls) -> Path:
        return Path(JAXON_URDF_PATH)

    @staticmethod
    def add_end_coords(robot_model: RobotModel) -> None:
        rarm_id, larm_id = robot_model.get_link_ids(["RARM_LINK7", "LARM_LINK7"])
        matrix = rotation_matrix(np.pi * 0.5, [0, 0, 1.0])
        rpy = np.flip(rpy_angle(matrix)[0])
        robot_model.add_new_link("rarm_end_coords", rarm_id, [0, 0, -0.220], rotation=rpy)
        robot_model.add_new_link("larm_end_coords", larm_id, [0, 0, -0.220], rotation=rpy)

        rleg_id, lleg_id = robot_model.get_link_ids(["RLEG_LINK5", "LLEG_LINK5"])
        robot_model.add_new_link("rleg_end_coords", rleg_id, [0, 0, -0.1])
        robot_model.add_new_link("lleg_end_coords", lleg_id, [0, 0, -0.1])

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
        self, rleg: bool = True, lleg: bool = True, rarm: bool = True, larm: bool = True
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

        kinmap = ArticulatedEndEffectorKinematicsMap(
            self.urdf_path(),
            self._get_control_joint_names(),
            endeffector_names,
            base_type=BaseType.FLOATING,
            rot_type=RotationType.XYZW,
            fksolver_init_hook=self.add_end_coords,
        )
        return kinmap

    def get_collision_kin(
        self, rsole: bool = True, lsole: bool = True
    ) -> ArticulatedCollisionKinematicsMap:
        link_wise_sphere_creator = {}
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
            sc_rleg5 = copy.deepcopy(SphereCollection(*list(zip(*collection))))
            link_wise_sphere_creator[link_name] = lambda mesh: sc_rleg5
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
            sc_lleg5 = copy.deepcopy(SphereCollection(*list(zip(*collection))))
            link_wise_sphere_creator[link_name] = lambda mesh: sc_lleg5
            collision_link_names.append(link_name)

        # link3
        link_name = "RLEG_LINK3"
        collection = []
        collection.append((np.array([0.0, 0.0, 0.0]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.02, 0.0, -0.1]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.02, 0.0, -0.2]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.0, 0.0, -0.3]), 0.08, str(uuid.uuid4())))
        sc_rleg4 = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_creator[link_name] = lambda mesh: sc_rleg4
        collision_link_names.append(link_name)

        link_name = "LLEG_LINK3"
        collection = []
        collection.append((np.array([0.0, 0.0, 0.0]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.02, 0.0, -0.1]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.02, 0.0, -0.2]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.0, 0.0, -0.3]), 0.08, str(uuid.uuid4())))
        sc_lleg4 = copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_creator[link_name] = lambda mesh: sc_lleg4
        collision_link_names.append(link_name)

        kinmap = ArticulatedCollisionKinematicsMap(
            self.urdf_path(),
            self._get_control_joint_names(),
            collision_link_names,
            link_wise_sphere_creator=link_wise_sphere_creator,
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

    def get_neural_selcol_const(self, robot_model: Jaxon) -> NeuralSelfCollFreeConst:
        return NeuralSelfCollFreeConst.load(
            self.urdf_path(), self._get_control_joint_names(), robot_model, BaseType.FLOATING
        )
