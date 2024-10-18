import importlib
from typing import List
from pathlib import Path
from dataclasses import dataclass

from skrobot.model import Link
from skrobot.sdf import UnionSDF
from skrobot.coordinates import CascadedCoords
from skrobot.models.urdf import RobotModelFromURDF

from skmp.constraint import BoxConst
from skmp.kinematics import CollSphereKinematicsMap, EndEffectorKinematicsMap

from tinyfk import BaseType, KinematicModel, RotationType


@dataclass
class EndEffectorList:
    link_names: List[str]
    end_effector_names: List[str]
    positions: List[List[float]]
    rpys: List[List[float]]


@dataclass
class SurroundingList:
    name: List[str]
    shape: List[str]
    size: List[List[float]]
    position: List[List[float]]
    rpy: List[List[float]]
    color: List[List[int]]


class Robot(RobotModelFromURDF):
    urdf_path: Path
    end_effector_list: EndEffectorList

    def __init__(self, urdf_path: str, end_effector_list: EndEffectorList):
        super().__init__(urdf_file=urdf_path)

        for link_name, end_effector_name, position, rpy in zip(
            end_effector_list.link_names,
            end_effector_list.end_effector_names,
            end_effector_list.positions,
            end_effector_list.rpys,
        ):
            setattr(self, end_effector_name, CascadedCoords(getattr(self, link_name), name=end_effector_name))
            getattr(self, end_effector_name).rotate(rpy[0], "x")
            getattr(self, end_effector_name).rotate(rpy[1], "y")
            getattr(self, end_effector_name).rotate(rpy[2], "z")
            getattr(self, end_effector_name).translate(position)


class RobotSurrounding:
    surrounding_list: SurroundingList

    def __init__(self, surrounding_list: SurroundingList):
        package_name = "skrobot.model.primitives"
        module = importlib.import_module(package_name)
        self.sdf_list = []
        self.surrounding_list = surrounding_list
        for name, shape, size, position, rpy, color in zip(
            surrounding_list.name,
            surrounding_list.shape,
            surrounding_list.size,
            surrounding_list.position,
            surrounding_list.rpy,
            surrounding_list.color,
        ):
            try:
                shapeClass = getattr(module, shape)
                setattr(self, name, shapeClass(size, with_sdf=True))
                getattr(self, name).rotate(rpy[0], "x")
                getattr(self, name).rotate(rpy[1], "y")
                getattr(self, name).rotate(rpy[2], "z")
                getattr(self, name).translate(position)
                getattr(self, name).set_color(color)
                self.sdf_list.append(getattr(self, name).sdf)
            except ImportError as e:
                print(f"Error importing {shape}: {e}")
            except AttributeError as e:
                print(f"{shape} does not exist in {package_name}.")

    def append_sdf(self, objects: List[Link]):
        self.sdf_list.extend([obj.sdf for obj in objects])

    def get_sdf_list(self):
        sdf = UnionSDF(self.sdf_list)
        return sdf

    def get_object_list(self):
        return [getattr(self, name) for name in self.surrounding_list.name]


class RobotConfig:
    urdf_path: Path
    end_effector_list: EndEffectorList

    def __init__(self, urdf_path: str, end_effector_list: EndEffectorList):
        self.urdf_path = Path(urdf_path).expanduser()
        self.end_effector_list = end_effector_list

    def add_end_coords(self, robot_model: KinematicModel) -> None:
        for link_name, end_effector_name, position, rpy in zip(
            self.end_effector_list.link_names,
            self.end_effector_list.end_effector_names,
            self.end_effector_list.positions,
            self.end_effector_list.rpys,
        ):
            link_id = robot_model.get_link_ids([link_name])[0]
            try:
                robot_model.add_new_link(end_effector_name, link_id, position, rpy)
            except Exception as e:
                print(f"[Warning] Failed to add {end_effector_name}: {e}")

    def get_endeffector_kin(
            self,
            base_type: BaseType = BaseType.FIXED,
            rot_type: RotationType = RotationType.XYZW
    ) -> EndEffectorKinematicsMap:
        kinmap = EndEffectorKinematicsMap(
            self.urdf_path,
            self.get_control_joint_names(),
            end_effector_names=self.end_effector_list.end_effector_names,
            base_type=base_type,
            rot_type=rot_type,
            fksolver_init_hook=self.add_end_coords,
        )
        return kinmap

    def get_box_const(self) -> BoxConst:
        raise NotImplementedError

    def get_control_joint_names(self) -> List[str]:
        raise NotImplementedError

    def get_collision_kin(self) -> CollSphereKinematicsMap:
        raise NotImplementedError
