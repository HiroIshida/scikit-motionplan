import copy
import uuid
import numpy as np
from pathlib import Path
from typing import List, Dict

from tinyfk import BaseType
from skmp.constraint import BoxConst
from skmp.collision import SphereCollection
from skmp.kinematics import CollSphereKinematicsMap

from skmp.robot.robot import RobotConfig, EndEffectorList


class A0BConfig(RobotConfig):
    urdf_path: Path
    end_effector_list: EndEffectorList

    def __init__(self, urdf_path: str, end_effector_list: EndEffectorList):
        super().__init__(urdf_path, end_effector_list)

    def get_control_joint_names(self) -> List[str]:
        names = []
        for i in range(6):
            names.append("RARM_JOINT{}".format(i))
        return names

    def get_collision_kin(self) -> CollSphereKinematicsMap:
        collision_link_names = ["RARM_LINK{}".format(i) for i in range(6)]

        link_wise_sphere_collection: Dict[str, SphereCollection] = {}

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

        kinmap = CollSphereKinematicsMap(
            self.urdf_path,
            self.get_control_joint_names(),
            link_wise_sphere_collection=link_wise_sphere_collection,
            base_type=BaseType.FIXED,
        )
        return kinmap

    def get_box_const(self) -> BoxConst:
        bounds = BoxConst.from_urdf(self.urdf_path, self.get_control_joint_names())
        return bounds
