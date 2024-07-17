import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import fcl
import numpy as np
import yaml
from fcl import BVHModel, CollisionObject, Transform
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rpy_angle, rpy_matrix
from skrobot.model.link import Link
from skrobot.model.robot_model import RobotModel
from tinyfk import BaseType
from trimesh import Trimesh

from skmp.kinematics import SphereCollection


def load_collision_spheres(yaml_file_path: Path) -> Dict[str, SphereCollection]:
    with open(yaml_file_path, "r") as f:
        collision_config = yaml.safe_load(f)
    d = collision_config["collision_spheres"]

    def unique_name(link_name) -> str:
        return link_name + str(uuid.uuid4())[:13]

    link_wise_sphere_collection: Dict[str, SphereCollection] = {}
    for link_name, vals in d.items():
        spheres_d = vals["spheres"]
        tmp_list = []
        for spec in spheres_d:
            vals = np.array(spec)
            center, r = vals[:3], vals[3]
            tmp_list.append((center, r, unique_name(link_name)))
        link_wise_sphere_collection[link_name] = SphereCollection(*list(zip(*tmp_list)))
    return link_wise_sphere_collection


def set_robot_state(
    robot_model: RobotModel,
    joint_names: List[str],
    angles: np.ndarray,
    base_type: BaseType = BaseType.FIXED,
) -> None:
    if base_type == BaseType.PLANER:
        assert len(joint_names) + 3 == len(angles)
        av_joint, av_base = angles[:-3], angles[-3:]
        x, y, theta = av_base
        co = Coordinates(pos=[x, y, 0.0], rot=rpy_matrix(theta, 0.0, 0.0))
        robot_model.newcoords(co)
    elif base_type == BaseType.FLOATING:
        assert len(joint_names) + 6 == len(angles)
        av_joint, av_base = angles[:-6], angles[-6:]
        xyz, rpy = av_base[:3], av_base[3:]
        co = Coordinates(pos=xyz, rot=rpy_matrix(*np.flip(rpy)))
        robot_model.newcoords(co)
    else:
        assert len(joint_names) == len(angles)
        av_joint = angles

    for joint_name, angle in zip(joint_names, av_joint):
        robot_model.__dict__[joint_name].joint_angle(angle)


def get_robot_state(
    robot_model: RobotModel, joint_names: List[str], base_type: BaseType = BaseType.FIXED
) -> np.ndarray:
    av_joint = np.array([robot_model.__dict__[jn].joint_angle() for jn in joint_names])
    if base_type == BaseType.PLANER:
        x, y, _ = robot_model.translation
        rpy = rpy_angle(robot_model.rotation)[0]
        theta = rpy[0]
        base_pose_vec = np.array([x, y, theta])
    elif base_type == BaseType.FLOATING:
        xyz = robot_model.translation
        rpy = np.flip(rpy_angle(robot_model.rotation)[0])
        base_pose_vec = np.hstack([xyz, rpy])
    elif base_type == BaseType.FIXED:
        base_pose_vec = np.array([])
    else:
        assert False
    q = np.hstack([av_joint, base_pose_vec])
    return q


class FCLCollisionManager:
    models: List[BVHModel]
    link_name_id_table: Dict[str, int]
    collision_check_link_id_pairs: Set[Tuple[int, int]]

    def __init__(
        self,
        robot: RobotModel,
        link_name_group: List[str],
        ignore_pairs: Optional[Set[Tuple[str, str]]] = None,
    ):

        if ignore_pairs is None:
            ignore_pairs = set()

        self._initialize_models(robot)

        link_id_group1 = set([self.link_name_id_table[name] for name in link_name_group])
        link_id_group2 = set(self.link_name_id_table.values()) - link_id_group1

        pairs = [(i, j) for i in link_id_group1 for j in link_id_group2]
        collision_check_link_id_pairs = set([(i, j) if i < j else (j, i) for i, j in pairs])

        tmp = set(
            [(self.link_name_id_table[n1], self.link_name_id_table[n2]) for n1, n2 in ignore_pairs]
        )
        ignore_link_id_pairs = set([(i, j) if i < j else (j, i) for i, j in tmp])

        collision_check_link_id_pairs = collision_check_link_id_pairs - ignore_link_id_pairs
        self.collision_check_link_id_pairs = collision_check_link_id_pairs

    def _initialize_models(self, robot: RobotModel):
        models: List[BVHModel] = []
        link_name_to_id = {}
        model_id = 0
        for link in robot.link_list:
            collmesh: Trimesh = link.collision_mesh
            if collmesh is not None:
                # extract verts and tries from collmesh
                verts = collmesh.vertices
                tris = collmesh.faces
                m = fcl.BVHModel()
                m.beginModel(len(verts), len(tris))
                m.addSubModel(verts, tris)
                m.endModel()
                col_obj = CollisionObject(m)
                models.append(col_obj)
                link_name_to_id[link.name] = model_id
                model_id += 1
        self.models = models
        self.link_name_id_table = link_name_to_id

    def set_pose(self, link_id: Union[str, int], position: np.ndarray, rotmat: np.ndarray):
        tf = Transform(rotmat, position)
        if isinstance(link_id, str):
            link_id = self.link_name_id_table[link_id]
        self.models[link_id].setTransform(tf)

    def reflect_skrobot(self, robot: RobotModel):
        for link_name in self.link_name_id_table.keys():
            link: Link = robot.__dict__[link_name]
            rot = link.worldrot()
            pos = link.worldpos()
            self.set_pose(link_name, pos, rot)

    def check_collision(self, link_id1: Union[int, str], link_id2: Union[int, str]) -> bool:
        if isinstance(link_id1, str):
            link_id1 = self.link_name_id_table[link_id1]
        if isinstance(link_id2, str):
            link_id2 = self.link_name_id_table[link_id2]
        return fcl.collide(self.models[link_id1], self.models[link_id2])

    def check_self_collision(self) -> bool:
        for link_id1, link_id2 in self.collision_check_link_id_pairs:
            if self.check_collision(link_id1, link_id2):
                return True
        return False
