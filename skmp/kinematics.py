import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from skrobot.coordinates.math import rpy_angle
from skrobot.model import RobotModel
from skrobot.model.primitives import Box
from tinyfk import BaseType, KinematicModel, RotationType

from skmp.collision import SphereCollection
from skmp.utils import load_urdf_model_using_cache


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


class ArticulatedCollisionSpheresKinematicsMapBase(ArticulatedKinematicsMapBase, ABC):
    @abstractmethod
    def get_radius_list(self) -> List[float]:
        pass


class AttachedObstacleCollisionKinematicsMap(ArticulatedCollisionSpheresKinematicsMapBase):
    # if some obstacle is attached to the link (e.g. a box attached when robot is holding it)

    def __init__(
        self,
        urdfpath: Path,
        joint_names: List[str],
        attach_link_name: str,
        relative_position: np.ndarray,
        shape: Box,
        n_grid: int = 6,
        base_type: BaseType = BaseType.FIXED,
        fksolver_init_hook: Optional[Callable[[KinematicModel], None]] = None,
        margin=0.0,
    ):

        urdfpath_str = str(urdfpath.expanduser())
        fksolver = KinematicModel(urdfpath_str)
        if fksolver_init_hook is not None:
            fksolver_init_hook(fksolver)

        # sample grid points from the shape and add them as feature points
        assert shape.sdf is not None
        extent = shape._extents
        grid = np.meshgrid(
            np.linspace(-0.5 * extent[0], 0.5 * extent[0], n_grid),
            np.linspace(-0.5 * extent[1], 0.5 * extent[1], n_grid),
            np.linspace(-0.5 * extent[2], 0.5 * extent[2], n_grid),
        )
        grid_points = np.stack([g.flatten() for g in grid], axis=1)
        grid_points = shape.transform_vector(grid_points)
        grid_points = grid_points[shape.sdf(grid_points) > -1e-2]

        points_from_center = grid_points - shape.worldpos()
        points_from_link = points_from_center + relative_position
        attach_link_id = fksolver.get_link_ids([attach_link_name])[0]
        feature_names = []
        for pt in points_from_link:
            feature_name = "pt_{}".format(uuid.uuid4())
            feature_names.append(feature_name)
            fksolver.add_new_link(feature_name, attach_link_id, pt)

        dim_cspace = (
            len(joint_names)
            + (base_type == BaseType.PLANER) * 3
            + (base_type == BaseType.FLOATING) * 6
        )

        self.dim_cspace = dim_cspace
        self.dim_tspace = 3
        self.n_feature = len(grid_points)
        self.radius_list = [margin] * len(grid_points)
        self.fksolver = fksolver
        self.tinyfk_joint_ids = fksolver.get_joint_ids(joint_names)
        self.control_joint_names = joint_names
        self._base_type = base_type
        self._rot_type = RotationType.IGNORE
        self.tinyfk_feature_ids = fksolver.get_link_ids(feature_names)

    def get_radius_list(self) -> List[float]:
        return self.radius_list


class ArticulatedCollisionKinematicsMap(ArticulatedCollisionSpheresKinematicsMapBase):
    radius_list: List[float]
    sphere_name_list: List[str]
    sphere_center_list: List[np.ndarray]

    def __init__(
        self,
        urdfpath: Path,
        joint_names: List[str],
        link_wise_sphere_collection: Dict[str, SphereCollection],
        base_type: BaseType = BaseType.FIXED,
        fksolver_init_hook: Optional[Callable[[KinematicModel], None]] = None,
    ):

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

        load_urdf_model_using_cache(urdfpath)

        radius_list = []
        sphere_name_list = []
        sphere_center_list = []

        for ln in link_wise_sphere_collection.keys():
            sphere_collection = link_wise_sphere_collection[ln]
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

    def get_radius_list(self) -> List[float]:
        return self.radius_list
