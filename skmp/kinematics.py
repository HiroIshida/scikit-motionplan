import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
from skrobot.coordinates.math import rpy_angle
from skrobot.model import RobotModel
from tinyfk import BaseType, KinematicModel, RotationType
from trimesh import Trimesh

from skmp.collision import SphereCollection, create_sphere_collection
from skmp.utils import load_urdf_model_using_cache


class KinematicsMapProtocol(Protocol):
    @property
    def dim_cspace(self) -> int:
        ...

    @property
    def dim_tspace(self) -> int:
        ...

    @property
    def n_feature(self) -> int:
        ...

    def map(self, points_cspace: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """maps points in C-space to points in the task space.
        points_cspace: R^(n_points, n_feature)
        return: R^(n_points, n_feature, n_task), R^(n_points, n_feature, n_task, n_dof)
        """
        ...


class CollisionKinmaticsMapProtocol(KinematicsMapProtocol, Protocol):
    @property
    def radius_list(self) -> List[float]:
        ...


@dataclass
class TrivialKinmaticsMap:
    """Map when C-space and task space are equal, and thus the map is identical"""

    dim: int
    radius: float

    @property
    def dim_cspace(self) -> int:
        return self.dim

    @property
    def dim_tspace(self) -> int:
        return self.dim

    @property
    def n_feature(self) -> int:
        return 1

    @property
    def radius_list(self) -> List[float]:
        return [self.radius]

    def map(self, points_cspace: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """maps points in C-space to points in the task space."""
        n_point, n_dim_cspace = points_cspace.shape
        points_tspace = points_cspace.reshape(n_point, self.n_feature, self.dim_tspace)

        tmp = np.stack([np.eye(self.dim) for _ in range(n_point)])
        jacobians = tmp.reshape(n_point, self.n_feature, self.dim_tspace, self.dim_cspace)
        return points_tspace, jacobians


class ArticulatedKinematicsMapBase:
    dim_cspace: int
    dim_tspace: int
    n_feature: int
    fksolver: KinematicModel
    tinyfk_joint_ids: List[int]
    tinyfk_feature_ids: List[int]
    control_joint_names: List[str]
    _rot_type: RotationType
    _base_type: BaseType
    _map_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @property
    def rot_type(self) -> RotationType:
        return self._rot_type

    @property
    def base_type(self) -> BaseType:
        return self._base_type

    def update_joint_angles(
        self, joint_name_to_angle_table: Dict[str, float], base_pose_6d: np.ndarray
    ) -> None:
        """Update internal joint angles"""

        assert len(base_pose_6d) == 6

    def map(self, points_cspace: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_point, n_dim_cspace = points_cspace.shape

        f_tmp, j_tmp = self.fksolver.solve_fk(
            points_cspace,
            self.tinyfk_feature_ids,
            self.tinyfk_joint_ids,
            base_type=self._base_type,
            with_jacobian=True,
            rot_type=self._rot_type,
        )
        points_tspace = f_tmp.reshape(n_point, self.n_feature, self.dim_tspace)
        jacobians = j_tmp.reshape(n_point, self.n_feature, self.dim_tspace, self.dim_cspace)
        self._map_cache = (points_tspace, jacobians)
        return points_tspace, jacobians

    def map_skrobot_model(self, robot_model: RobotModel) -> Tuple[np.ndarray, np.ndarray]:
        joint_list = [robot_model.__dict__[name] for name in self.control_joint_names]
        av_joint = np.array([j.joint_angle() for j in joint_list])
        if self._base_type == BaseType.PLANER:
            x, y, _ = robot_model.translation
            ypr = rpy_angle(robot_model.rotation)[0]
            theta = ypr[0]
            np.hstack((av_joint, [x, y, theta]))
            av_whole = np.hstack((av_joint, [x, y, theta]))
        elif self._base_type == BaseType.FLOATING:
            x, y, z = robot_model.translation
            ypr = rpy_angle(robot_model.rotation)[0]
            av_whole = np.hstack((av_joint, [x, y, z], np.flip(ypr)))
        elif self._base_type == BaseType.FIXED:
            av_whole = av_joint
        else:
            assert False
        return self.map(np.expand_dims(av_whole, axis=0))

    def reflect_skrobot_model(self, robot_model: RobotModel):
        """reflecting skrobot model configuratin to tinyfk solver configuration"""
        xyz = robot_model.translation
        ypr = rpy_angle(robot_model.rotation)[0]
        base_pose = np.hstack([xyz, np.flip(ypr)])
        joint_angles = [
            robot_model.__dict__[name].joint_angle() for name in robot_model.joint_names
        ]
        angles = joint_angles + base_pose.tolist()
        joint_ids = self.fksolver.get_joint_ids(robot_model.joint_names)
        self.fksolver.set_q(joint_ids, angles, base_type=BaseType.FLOATING)

    def add_new_feature_point(
        self,
        link_like: Union[str, int],
        position: np.ndarray,
        rotation: Optional[np.ndarray] = None,
    ) -> None:
        """add relative point to specified link as a new feature point"""
        if isinstance(link_like, str):
            parent_link_id = self.fksolver.get_link_ids([link_like])[0]
        else:
            parent_link_id = link_like
        new_link_name = "new_feature_{}".format(uuid.uuid4())
        self.fksolver.add_new_link(new_link_name, parent_link_id, position, rotation)

        new_feature_id = self.fksolver.get_link_ids([new_link_name])[0]
        self.tinyfk_feature_ids.append(new_feature_id)
        self.n_feature += 1


class ArticulatedEndEffectorKinematicsMap(ArticulatedKinematicsMapBase):
    def __init__(
        self,
        urdfpath: Path,
        joint_names: List[str],
        end_effector_names: List[str],
        base_type: BaseType = BaseType.FIXED,
        rot_type: RotationType = RotationType.RPY,
        fksolver_init_hook: Optional[Callable[[KinematicModel], None]] = None,
    ):

        dim_cspace = (
            len(joint_names)
            + (base_type == BaseType.PLANER) * 3
            + (base_type == BaseType.FLOATING) * 6
        )

        urdfpath_str = str(urdfpath.expanduser())
        fksolver = KinematicModel(urdfpath_str)
        if fksolver_init_hook is not None:
            fksolver_init_hook(fksolver)

        tinyfk_ef_ids = fksolver.get_link_ids(end_effector_names)

        n_feature = len(tinyfk_ef_ids)
        tinyfk_joint_ids = fksolver.get_joint_ids(joint_names)

        self.dim_cspace = dim_cspace
        self.n_feature = n_feature
        self.fksolver = fksolver
        self.tinyfk_joint_ids = tinyfk_joint_ids
        self.control_joint_names = joint_names
        self._base_type = base_type
        self.update_rotation_type(rot_type)
        self.tinyfk_feature_ids = tinyfk_ef_ids

    def update_rotation_type(self, rot_type: RotationType) -> None:
        if rot_type == RotationType.IGNORE:
            self.dim_tspace = 3
        elif rot_type == RotationType.RPY:
            self.dim_tspace = 6
        elif rot_type == RotationType.XYZW:
            self.dim_tspace = 7
        else:
            assert False
        self._rot_type = rot_type


class ArticulatedCollisionKinematicsMap(ArticulatedKinematicsMapBase):
    radius_list: List[float]
    sphere_name_list: List[str]
    sphere_center_list: List[np.ndarray]

    def __init__(
        self,
        urdfpath: Path,
        joint_names: List[str],
        collision_link_names: List[str],
        base_type: BaseType = BaseType.FIXED,
        link_wise_sphere_collection: Optional[
            Dict[str, Union[Callable[[Trimesh], SphereCollection], SphereCollection]]
        ] = None,
        fksolver_init_hook: Optional[Callable[[KinematicModel], None]] = None,
    ):
        if link_wise_sphere_collection is None:
            link_wise_sphere_collection = {}
        for ln in collision_link_names:
            if ln not in link_wise_sphere_collection:
                link_wise_sphere_collection[ln] = create_sphere_collection

        dim_cspace = (
            len(joint_names)
            + (base_type == BaseType.PLANER) * 3
            + (base_type == BaseType.FLOATING) * 6
        )
        dim_tspace = 3
        rot_type = RotationType.IGNORE

        urdfpath_str = str(urdfpath.expanduser())
        fksolver = KinematicModel(urdfpath_str)
        if fksolver_init_hook is not None:
            fksolver_init_hook(fksolver)

        require_urdf_with_geometry = False
        for sphere_collection in link_wise_sphere_collection.values():
            if isinstance(sphere_collection, Callable):  # type: ignore
                require_urdf_with_geometry = True
        urdf = load_urdf_model_using_cache(urdfpath, with_geometry=require_urdf_with_geometry)

        radius_list = []
        sphere_name_list = []
        sphere_center_list = []

        for ln in collision_link_names:
            mesh: Trimesh = urdf.link_map[ln].collision_mesh
            sphere_collection_like = link_wise_sphere_collection[ln]
            if isinstance(sphere_collection_like, SphereCollection):
                sphere_collection = sphere_collection_like
            else:
                sphere_collection = sphere_collection_like(mesh)

            coll_link_id = fksolver.get_link_ids([ln])[0]

            for i in range(len(sphere_collection)):
                radius = sphere_collection.radius_list[i]
                name = sphere_collection.name_list[i]
                center = sphere_collection.center_list[i]

                radius_list.append(radius)
                sphere_name_list.append(name)
                sphere_center_list.append(center)

                fksolver.add_new_link(name, coll_link_id, center)
        n_feature = len(radius_list)

        tinyfk_joint_ids = fksolver.get_joint_ids(joint_names)
        tinyfk_sphere_ids = fksolver.get_link_ids(sphere_name_list)

        self.dim_cspace = dim_cspace
        self.dim_tspace = dim_tspace
        self.n_feature = n_feature
        self.radius_list = radius_list
        self.sphere_name_list = sphere_name_list
        self.sphere_center_list = sphere_center_list
        self.tinyfk_feature_ids = tinyfk_sphere_ids
        self.tinyfk_joint_ids = tinyfk_joint_ids
        self.control_joint_names = joint_names
        self.fksolver = fksolver
        self._base_type = base_type
        self._rot_type = rot_type
