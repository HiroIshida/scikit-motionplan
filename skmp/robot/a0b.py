import copy
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Union

import numpy as np
from skrobot.coordinates import CascadedCoords
from skrobot.model.primitives import Box
from skrobot.models.urdf import RobotModelFromURDF
from tinyfk import BaseType, KinematicModel, RotationType
from trimesh import Trimesh

from skmp.collision import SphereCollection
from skmp.constraint import BoxConst
from skmp.kinematics import (
    ArticulatedCollisionKinematicsMap,
    ArticulatedEndEffectorKinematicsMap,
)

END_COORDS_TRANSLATION = 0.25


class A0B(RobotModelFromURDF):
    def __init__(self, urdf_path):
        # model is not publicaly available..?
        super().__init__(urdf_file=urdf_path)
        self.rarm_end_coords = CascadedCoords(self.RARM_LINK5, name="rarm_end_coords")
        self.rarm_end_coords.translate([END_COORDS_TRANSLATION, 0, 0.0])


class A0BSurrounding:
    pole: Box
    table: Box

    def __init__(self):
        pass

        pole = Box([0.2, 0.2, 1.5], with_sdf=True)
        pole.translate([-0.2, 0, -0.75])
        pole.visual_mesh.visual.face_colors = [255, 255, 255, 120]
        self.pole = pole

        table = Box([2.0, 2.0, 0.1], with_sdf=True)
        table.translate([0.0, 0.0, -0.5])
        table.visual_mesh.visual.face_colors = [255, 255, 255, 120]
        self.table = table


class A0BConfig:
    urdf_path: Path

    def __init__(self, urdf_path: str):
        self.urdf_path = Path(urdf_path).expanduser()

    def _get_control_joint_names(self) -> List[str]:
        names = []
        for i in range(6):
            names.append("RARM_JOINT{}".format(i))
        return names

    @staticmethod
    def add_end_coords(robot_model: KinematicModel) -> None:
        rarm_id = robot_model.get_link_ids(["RARM_LINK5"])[0]
        robot_model.add_new_link("rarm_end_coords", rarm_id, [END_COORDS_TRANSLATION, 0, 0.0])

    def get_endeffector_kin(self):
        kinmap = ArticulatedEndEffectorKinematicsMap(
            self.urdf_path,
            self._get_control_joint_names(),
            ["rarm_end_coords"],
            base_type=BaseType.FIXED,
            rot_type=RotationType.XYZW,
            fksolver_init_hook=self.add_end_coords,
        )
        return kinmap

    def get_collision_kin(self) -> ArticulatedCollisionKinematicsMap:
        collision_link_names = ["RARM_LINK{}".format(i) for i in range(6)]

        link_wise_sphere_collection: Dict[
            str, Union[Callable[[Trimesh], SphereCollection], SphereCollection]
        ] = {}

        link_name = "RARM_LINK0"
        collection = []
        collection.append((np.array([0.0, 0.0, 0.0]), 0.08, str(uuid.uuid4())))
        link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
        collision_link_names.append(link_name)

        link_name = "RARM_LINK1"
        collection = []
        collection.append((np.array([0.1, 0.0, 0.0]), 0.05, str(uuid.uuid4())))
        collection.append((np.array([0.2, 0.0, 0.0]), 0.09, str(uuid.uuid4())))
        link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
        collision_link_names.append(link_name)

        link_name = "RARM_LINK2"
        collection = []
        collection.append((np.array([0.0, 0.0, 0.0]), 0.095, str(uuid.uuid4())))
        copy.deepcopy(SphereCollection(*list(zip(*collection))))
        link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
        collision_link_names.append(link_name)

        link_name = "RARM_LINK3"
        collection = []
        collection.append((np.array([0.1, 0.0, 0.0]), 0.05, str(uuid.uuid4())))
        collection.append((np.array([0.2, 0.0, 0.0]), 0.09, str(uuid.uuid4())))
        link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
        collision_link_names.append(link_name)

        link_name = "RARM_LINK4"
        collection = []
        collection.append((np.array([0.0, 0.0, 0.0]), 0.09, str(uuid.uuid4())))
        link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
        collision_link_names.append(link_name)

        link_name = "RARM_LINK5"
        collection = []
        collection.append((np.array([0.1, 0.0, 0.0]), 0.05, str(uuid.uuid4())))
        collection.append((np.array([0.18, 0.0, 0.0]), 0.08, str(uuid.uuid4())))
        collection.append((np.array([0.18, 0.06, 0.025]), 0.045, str(uuid.uuid4())))
        collection.append((np.array([0.18, -0.06, 0.025]), 0.045, str(uuid.uuid4())))
        collection.append((np.array([0.18, 0.06, -0.025]), 0.045, str(uuid.uuid4())))
        collection.append((np.array([0.18, -0.06, -0.025]), 0.045, str(uuid.uuid4())))
        link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*collection)))
        collision_link_names.append(link_name)

        kinmap = ArticulatedCollisionKinematicsMap(
            self.urdf_path,
            self._get_control_joint_names(),
            ["RARM_LINK{}".format(i) for i in range(6)],
            link_wise_sphere_collection=link_wise_sphere_collection,
            base_type=BaseType.FIXED,
        )
        return kinmap

    def get_box_const(self) -> BoxConst:
        bounds = BoxConst.from_urdf(self.urdf_path, self._get_control_joint_names())
        return bounds
