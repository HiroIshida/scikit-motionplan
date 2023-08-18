import copy
import itertools
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Callable,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
from selcol.file import default_pretrained_basepath
from selcol.runtime import OrtSelColInferencer
from skrobot.coordinates import Coordinates, matrix2quaternion, rpy_angle
from skrobot.coordinates.math import wxyz2xyzw
from skrobot.model import RobotModel
from skrobot.model.primitives import Box
from skrobot.utils.urdf import JointLimit
from tinyfk import BaseType, KinematicModel, RotationType

from skmp.kinematics import (
    ArticulatedCollisionKinematicsMap,
    ArticulatedEndEffectorKinematicsMap,
)
from skmp.utils import load_urdf_model_using_cache


class AbstractConst(ABC):
    reflect_robot_flag: bool = False
    id_value: str

    def assign_id_value(self):
        self.id_value = str(uuid.uuid4())

    def evaluate(self, qs: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        if not self.reflect_robot_flag:
            message = "{}: you need to call reflect_skrobot_model beforehand".format(
                type(self).__name__
            )
            raise RuntimeError(message)

        return self._evaluate(qs, with_jacobian)

    def evaluate_single(self, q: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        assert q.ndim == 1
        f, jac = self.evaluate(np.expand_dims(q, axis=0), with_jacobian)
        return f[0], jac[0]

    def dummy_jacobian(self) -> np.ndarray:
        return np.array([[np.nan]])

    @abstractmethod
    def _evaluate(self, qs: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def reflect_skrobot_model(self, robot_model: Optional[RobotModel]) -> None:
        """reflect skrobot model state to internal state
        Although constraints does not necessarily require to reflect robot model,
        I believe defensive programming is always better.
        For constraints that does not require robot, you can pass None.
        """
        self._reflect_skrobot_model(robot_model)
        self.reflect_robot_flag = True

    @abstractmethod
    def _reflect_skrobot_model(self, robot_model: Optional[RobotModel]) -> None:
        ...

    @classmethod
    @abstractmethod
    def is_equality(cls) -> bool:
        ...


ConstraintT = TypeVar("ConstraintT", bound=AbstractConst)


class AbstractIneqConst(AbstractConst):
    @classmethod
    def is_equality(cls) -> bool:
        return False

    def is_valid(self, q: np.ndarray) -> bool:
        value, _ = self.evaluate_single(q, False)
        return bool(np.all(value > 0))


class AbstractEqConst(AbstractConst):
    @classmethod
    def is_equality(cls) -> bool:
        return True

    def is_approx_satisfied(self, q: np.ndarray, eps: float = 0.01) -> bool:
        # TODO: this is experimental!!
        values, _ = self.evaluate_single(q, False)
        print(values)
        return bool(np.all(np.abs(values) < eps))


@runtime_checkable
class VectorDescriptable(Protocol):
    def get_description(self) -> np.ndarray:
        ...


CompositeConstT = TypeVar("CompositeConstT", bound="_CompositeConst")
InnerConstT = TypeVar("InnerConstT", bound=Union[AbstractIneqConst, AbstractEqConst])


class _CompositeConst(AbstractConst, Generic[InnerConstT]):
    const_list: List[InnerConstT]

    def __init__(self, const_list: List[InnerConstT]) -> None:
        for const in const_list:
            assert const.is_equality() == self.is_equality()
        self.const_list = const_list

        # NOTE: composite constraint is special case that does not need
        # call of _reflect_skrobot_model, because all const_list
        # is supposed to already reflect robot state
        self.reflect_robot_flag = True
        self.assign_id_value()

        self._remove_duplication()

    def _evaluate(self, qs: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        valuess_list = []
        jacs_list = []

        for const in self.const_list:
            values, jac = const.evaluate(qs, with_jacobian=with_jacobian)
            valuess_list.append(values)
            jacs_list.append(jac)

        valuess_out = np.hstack(valuess_list)

        if not with_jacobian:
            return valuess_out, self.dummy_jacobian()

        jacs_out = np.concatenate(jacs_list, axis=1)
        return valuess_out, jacs_out

    def _reflect_skrobot_model(self, robot_model: Optional[RobotModel]) -> None:
        for const in self.const_list:
            const.reflect_skrobot_model(robot_model)

    def _remove_duplication(self) -> None:
        table = {}
        for const in self.const_list:
            table[const.id_value] = const
        const_list = [value for value in table.values()]
        self.const_list = const_list


class IneqCompositeConst(AbstractIneqConst, _CompositeConst[AbstractIneqConst]):
    """
    Order of the list of constraints affects lot to performance.
    See the comment in is_valid function.
    """

    def is_valid(self, q: np.ndarray) -> bool:
        # NOTE: return False as soon as possible when
        # finding invalidity found for some constraint.
        # therefore, the order of const_list is "really" important
        # to speed up planning
        for const in self.const_list:
            values, _ = const.evaluate_single(q, False)
            if np.any(values < 0.0):
                return False
        return True


class EqCompositeConst(AbstractEqConst, _CompositeConst[AbstractEqConst]):
    ...


class BoxConst(AbstractIneqConst):
    lb: np.ndarray
    ub: np.ndarray

    def __init__(self, lb: np.ndarray, ub: np.ndarray) -> None:
        self.lb = lb
        self.ub = ub
        self.reflect_skrobot_model(None)
        self.assign_id_value()

    @classmethod
    def from_urdf(
        cls,
        urdf_path: Path,
        joint_names: List[str],
        base_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):

        urdf = load_urdf_model_using_cache(urdf_path.expanduser())
        b_min = []
        b_max = []
        for joint_name in joint_names:
            limit: JointLimit = urdf.joint_map[joint_name].limit

            if limit.lower in [-np.inf, np.nan, None]:
                b_min.append(-2 * np.pi)
            else:
                b_min.append(limit.lower)

            if limit.upper in [+np.inf, np.nan, None]:
                b_max.append(2 * np.pi)
            else:
                b_max.append(limit.upper)

        if base_bounds is not None:
            lb, ub = base_bounds
            n_dof_base = len(lb)
            assert n_dof_base in (3, 6)
            for i in range(n_dof_base):
                b_min.append(lb[i])
                b_max.append(ub[i])

        return cls(np.array(b_min), np.array(b_max))

    def _evaluate(self, qs: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        n_point, dim = qs.shape
        f_lower = (qs - self.lb).flatten()
        f_upper = (self.ub - qs).flatten()
        f = np.hstack((f_lower, f_upper)).reshape(n_point, -1)
        if with_jacobian:
            jac_single = np.vstack((np.eye(dim), -np.eye(dim)))
            jac = np.array([jac_single for _ in range(n_point)])
        else:
            jac = self.dummy_jacobian()
        return f, jac

    def _reflect_skrobot_model(self, robot_model: Optional[RobotModel]) -> None:
        pass

    def sample(self) -> np.ndarray:
        w = self.ub - self.lb
        return np.random.rand(len(w)) * w + self.lb


class CollFreeConst(AbstractIneqConst):
    colkin: ArticulatedCollisionKinematicsMap
    sdf: Callable[[np.ndarray], np.ndarray]
    only_closest_feature: bool

    def __init__(
        self,
        colkin: ArticulatedCollisionKinematicsMap,
        sdf: Callable[[np.ndarray], np.ndarray],
        robot_model: RobotModel,
        only_closest_feature: bool = False,
    ) -> None:
        self.colkin = colkin
        self.sdf = sdf
        self.reflect_skrobot_model(robot_model)
        self.only_closest_feature = only_closest_feature
        self.assign_id_value()

    def _evaluate(self, qs: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        if self.only_closest_feature:
            return self._evaluate_closest(qs, with_jacobian)
        else:
            return self._evaluate_all(qs, with_jacobian)

    def _evaluate_closest(
        self, qs: np.ndarray, with_jacobian: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.sdf is not None
        n_point, dim_cspace = qs.shape
        self.colkin.n_feature
        dim_tspace = self.colkin.dim_tspace

        # xss: R^{n_point, n_feature * dim_tspace}
        # jacss: R^{n_point, n_feature, dim_tspace, dim_cspace}
        xss, jacss = self.colkin.map(qs)  # ss refere to points of points

        # NOTE: 1 because we focus single closest feature point
        values = np.zeros([n_point, 1])
        Js = np.zeros([n_point, 1, dim_cspace])

        for i in range(n_point):
            xs = xss[i]
            sds_stacked = self.sdf(xs)
            idx_closest = np.argmin(sds_stacked)
            sd_closest = sds_stacked[idx_closest]
            sd_with_margin = sd_closest - self.colkin.radius_list[idx_closest]

            values[i] = sd_with_margin

            if with_jacobian:
                eps = 1e-7
                xs_closest = xs[idx_closest]
                jac_closest = jacss[i][idx_closest]

                grad = np.zeros(dim_tspace)

                for j in range(dim_tspace):
                    xs_closest_plus = copy.deepcopy(xs_closest)
                    xs_closest_plus[j] += eps
                    sd_closest_plus = self.sdf(np.expand_dims(xs_closest_plus, axis=0))[0]
                    grad[j] = (sd_closest_plus - sd_closest) / eps
                Js[i, 0, :] = grad.dot(jac_closest)

        return values, Js

    def _evaluate_all(self, qs: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        """compute signed distance of feature points and its jacobian
        input:
            qs: R^{n_point, n_feature}
        output:
            fss: R^{n_point, n_feature}
            Jss: R^{n_point, n_feature, dim_cspace}
        """
        assert self.sdf is not None
        n_point, dim_cspace = qs.shape
        n_feature = self.colkin.n_feature
        dim_tspace = self.colkin.dim_tspace

        # xss: R^{n_point, n_feature * dim_tspace}
        # jacss: R^{n_point, n_feature, dim_tspace, dim_cspace}
        xss, jacss = self.colkin.map(qs)  # ss refere to points of points

        xs_stacked = xss.reshape((n_point * n_feature, dim_tspace))
        sds_stacked = self.sdf(xs_stacked)

        # compute sd_vals_stacked
        margin_radius = np.tile(np.array(self.colkin.radius_list), n_point)
        fs_stacked = sds_stacked - margin_radius
        fss = fs_stacked.reshape(n_point, n_feature)

        # compute jacobian by chain rule
        if with_jacobian:
            eps = 1e-7
            grads_stacked = np.zeros((n_feature * n_point, dim_tspace))

            for i in range(dim_tspace):
                xs_stacked_plus = copy.deepcopy(xs_stacked)
                xs_stacked_plus[:, i] += eps
                sds_stacked_plus = self.sdf(xs_stacked_plus)
                grads_stacked[:, i] = (sds_stacked_plus - sds_stacked) / eps
            gradss = grads_stacked.reshape((n_point, n_feature, dim_tspace))
            Jss = np.einsum("ijk,ijkl->ijl", gradss, jacss)
        else:
            Jss = self.dummy_jacobian()
        return fss, Jss

    def _reflect_skrobot_model(self, robot_model: Optional[RobotModel]) -> None:
        assert robot_model, "robot_model must not be None"
        self.colkin.reflect_skrobot_model(robot_model)


class ReducedCollisionFreeConst(CollFreeConst):
    def _evaluate(self, qs: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        fss, jss = super()._evaluate(qs, with_jacobian)

        fss_new_tmp = []
        jss_new_tmp = []

        for fs, js in zip(fss, jss):
            # fs: R^{n_feature}, js: R^{n_feature, n_cspace}
            idx_closest_feature = np.argmin(fs)
            fs_min = fs[idx_closest_feature]
            fs_new = np.array([fs_min])
            fss_new_tmp.append(fs_new)

            if with_jacobian:
                jac_min = js[idx_closest_feature]  # grad
                jac_new = np.expand_dims(jac_min, axis=0)
                jss_new_tmp.append(jac_new)

        fss_new = np.array(fss_new_tmp)

        if with_jacobian:
            jss_new = np.array(jss_new_tmp)
            return fss_new, jss_new
        else:
            return fss_new, self.dummy_jacobian()


class PointCollFreeConst(AbstractIneqConst):
    """
    Collision free constraint for point-shaped robot
    """

    sdf: Callable[[np.ndarray], np.ndarray]
    eps: float = 1e-6

    def __init__(self, sdf: Callable[[np.ndarray], np.ndarray]):
        self.sdf = sdf
        self.reflect_skrobot_model(None)
        self.assign_id_value()

    def _evaluate(self, qs: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        n_point, n_dim = qs.shape
        jacs_stacked = np.zeros((n_point, 1, n_dim))
        fs = self.sdf(qs)
        for i in range(n_dim):
            qs1 = copy.deepcopy(qs)
            qs1[:, i] += self.eps
            jacs_stacked[:, :, i] = (self.sdf(qs1) - fs) / self.eps
        return fs.reshape(-1, 1), jacs_stacked

    def _reflect_skrobot_model(self, robot_model: Optional[RobotModel]) -> None:
        return None


class ConfigPointConst(AbstractEqConst):
    desired_angles: np.ndarray

    def __init__(self, desired_angles: np.ndarray) -> None:
        self.desired_angles = desired_angles
        self.reflect_skrobot_model(None)
        self.assign_id_value()

    def _evaluate(self, qs: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        n_point, dim = qs.shape
        val = qs - self.desired_angles
        if with_jacobian:
            jac = np.array([np.eye(dim) for _ in range(n_point)])
        else:
            jac = self.dummy_jacobian()
        return val, jac

    def _reflect_skrobot_model(self, robot_model: Optional[RobotModel]) -> None:
        pass

    def get_description(self) -> np.ndarray:
        return self.desired_angles


class PoseConstraint(AbstractEqConst):
    efkin: ArticulatedEndEffectorKinematicsMap
    desired_poses: List[np.ndarray]
    debug_rank_deficiency: bool = False

    def __init__(
        self,
        desired_poses: List[np.ndarray],
        efkin: ArticulatedEndEffectorKinematicsMap,
        robot_model: RobotModel,
        debug_rank_deficiency: bool,
    ) -> None:
        assert len(desired_poses) == efkin.n_feature
        self.desired_poses = desired_poses
        self.efkin = efkin
        self.reflect_skrobot_model(robot_model)
        self.debug_rank_deficiency = debug_rank_deficiency
        self.assign_id_value()

    def _evaluate(
        self, qs: np.ndarray, with_jacobian: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_point, n_dim = qs.shape
        xs_tmp, jacs_tmp = self.efkin.map(qs)
        xs = xs_tmp.reshape(n_point, -1)
        jacs = jacs_tmp.reshape(n_point, -1, n_dim)
        n_point, dim = qs.shape

        target = np.hstack(self.desired_poses)
        values = xs - target
        if self.debug_rank_deficiency:
            for i in range(len(jacs)):
                jac = jacs[i]
                rank_diff = jac.shape[0] - np.linalg.matrix_rank(jac)
                if rank_diff > 0:
                    row_sums = np.sum(jac, axis=1)
                    message = "rank diference: {}".format(rank_diff)
                    message += ", row sums: {}".format(row_sums)
                    assert False, message

        return values, jacs

    @classmethod
    def from_skrobot_coords(
        cls,
        co_list: List[Coordinates],
        efkin: ArticulatedEndEffectorKinematicsMap,
        robot_model: RobotModel,
        debug_rank_deficiency: bool = False,
    ) -> "PoseConstraint":
        vector_list = []
        for co in co_list:
            pos = co.worldpos()
            if efkin.rot_type == RotationType.RPY:
                ypr = rpy_angle(co.worldrot())[0]
                rpy = np.flip(ypr)
                vector = np.hstack([pos, rpy])
            elif efkin.rot_type == RotationType.XYZW:
                xyzw = wxyz2xyzw(matrix2quaternion(co.worldrot()))
                vector = np.hstack([pos, xyzw])
            elif efkin.rot_type == RotationType.IGNORE:
                vector = pos
            else:
                assert False

            vector_list.append(vector)
        return cls(vector_list, efkin, robot_model, debug_rank_deficiency)

    def _reflect_skrobot_model(self, robot_model: Optional[RobotModel]) -> None:
        assert robot_model is not None
        self.efkin.reflect_skrobot_model(robot_model)

    def get_description(self) -> np.ndarray:
        return np.hstack(self.desired_poses)


class RelativePoseConstraint(AbstractEqConst):
    desired_relative_position: np.ndarray  # todo: extend this to pose
    efkin: ArticulatedEndEffectorKinematicsMap

    def __init__(
        self,
        desired_relative_position: np.ndarray,
        efkin: ArticulatedEndEffectorKinematicsMap,
        robot_model: RobotModel,
    ):
        assert efkin.rot_type == RotationType.RPY  # TODO: support xyzw

        efkin = copy.deepcopy(efkin)

        # we consider relative pose from feature-1 to feature-2
        assert efkin.n_feature == 2
        feature_1_id = efkin.tinyfk_feature_ids[0]
        efkin.add_new_feature_point(feature_1_id, desired_relative_position, None)

        # now we have feature-3 which is relative transformed pose from feature-1
        # and we are going to try to match feature 2 and feature 3
        assert efkin.n_feature == 3

        self.desired_relative_position = desired_relative_position
        self.efkin = efkin
        self.reflect_skrobot_model(robot_model)
        self.assign_id_value()

    def _evaluate(self, qs: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        n_point, n_dim = qs.shape
        xs, jacs = self.efkin.map(
            qs
        )  # xs: R^(n_point, n_task), jacs: R^(n_point, n_feature, n_task, n_dof)

        points_feature2 = xs[:, 1, :]
        points_feature3 = xs[:, 2, :]

        diffs = points_feature2 - points_feature3  # R^(n_point, n_task)

        if not with_jacobian:
            return diffs, self.dummy_jacobian()
        else:
            jacs_feature2 = jacs[:, 1, :, :]
            jacs_feature3 = jacs[:, 2, :, :]
            jacs_diff = jacs_feature2 - jacs_feature3  # R^(n_point, n_task, n_dof)
            return diffs, jacs_diff

    def _reflect_skrobot_model(self, robot_model: Optional[RobotModel]) -> None:
        assert robot_model is not None
        self.efkin.reflect_skrobot_model(robot_model)


class PairWiseSelfCollFreeConst(AbstractIneqConst):
    colkin: ArticulatedCollisionKinematicsMap
    check_sphere_id_pairs: List[Tuple[int, int]]
    check_sphere_pair_sqdists: np.ndarray  # pair sqdist means (r1 + r2) ** 2
    only_closest_feature: bool

    def __init__(
        self,
        colkin: ArticulatedCollisionKinematicsMap,
        robot_model: RobotModel,
        id_pairs: Optional[List[Tuple[int, int]]] = None,
        only_closest_feature: bool = False,
    ) -> None:

        # create sphere_id_raius_table
        sphere_id_raius_table = {}
        for sphere_id, radius in zip(colkin.tinyfk_feature_ids, colkin.radius_list):
            sphere_id_raius_table[sphere_id] = radius

        all_index_pairs = list(itertools.combinations(colkin.tinyfk_feature_ids, 2))
        pair_pair_dist_table = {}
        for pair in all_index_pairs:
            sphere_id1 = pair[0]
            sphere_id2 = pair[1]
            r1 = sphere_id_raius_table[sphere_id1]
            r2 = sphere_id_raius_table[sphere_id2]
            pair_pair_dist_table[pair] = r1 + r2

        if id_pairs is None:
            # here in this constructor, we will filter out collision pair which is already collide
            # at the initial pose np.zeros(n_dof)

            # compute inter-sphere distances when q = np.zeros(n_dof)
            q_init = np.zeros(colkin.dim_cspace)
            sqdists, _ = colkin.fksolver.compute_inter_link_sqdists(
                [q_init], all_index_pairs, colkin.tinyfk_joint_ids, base_type=colkin.base_type
            )
            dists = np.sqrt(sqdists)

            # determine collision pairs
            # because for cpython >= 3.6, dict is orderd...
            rs = np.array(list(pair_pair_dist_table.values()))

            # multiplying rs seems good heuristics.
            # for example, if sphere is large, the margin must be large
            # but for small one, we don't need large margin.
            rs_with_margin = rs * 3
            collision_pair_indices = np.where(dists - rs_with_margin < 0)[0]
            collision_pairs = [all_index_pairs[idx] for idx in collision_pair_indices]

            # subtract collision pairs from the all pairs
            valid_sphere_id_pair_set = set(all_index_pairs).difference(set(collision_pairs))
            valid_sphere_id_pairs = list(valid_sphere_id_pair_set)
            valid_sphere_pair_dists = np.array(
                [pair_pair_dist_table[pair] for pair in valid_sphere_id_pairs]
            )
            id_pairs = valid_sphere_id_pairs
            pair_dists = valid_sphere_pair_dists
        else:
            pair_dists = np.array([pair_pair_dist_table[pair] for pair in id_pairs])

        self.colkin = colkin
        self.check_sphere_id_pairs = id_pairs
        self.check_sphere_pair_sqdists = pair_dists**2
        self.reflect_skrobot_model(robot_model)
        self.only_closest_feature = only_closest_feature
        self.assign_id_value()

    def _evaluate(self, qs: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        n_wp, n_dim = qs.shape

        sqdists_stacked, grads_stacked = self.colkin.fksolver.compute_inter_link_sqdists(
            qs,
            self.check_sphere_id_pairs,
            self.colkin.tinyfk_joint_ids,
            base_type=self.colkin.base_type,
            with_jacobian=with_jacobian,
        )
        sqdistss = sqdists_stacked.reshape(n_wp, -1)
        valuess = sqdistss - self.check_sphere_pair_sqdists

        if self.only_closest_feature:
            min_valuess = np.expand_dims(np.min(valuess, axis=1), axis=1)
            if not with_jacobian:
                return min_valuess, self.dummy_jacobian()
            else:
                min_jacs = np.zeros((n_wp, 1, n_dim))
                gradss = grads_stacked.reshape(n_wp, -1, n_dim)
                for i in range(n_wp):
                    values, grads = valuess[i], gradss[i]
                    idx_min = np.argmin(values)
                    min_jacs[i, 0, :] = grads[idx_min]
            return min_valuess, min_jacs
        else:
            if not with_jacobian:
                return valuess, self.dummy_jacobian()
            gradss = grads_stacked.reshape(n_wp, -1, n_dim)
            return valuess, gradss

    def _reflect_skrobot_model(self, robot_model: Optional[RobotModel]) -> None:
        assert robot_model is not None
        self.colkin.reflect_skrobot_model(robot_model)


class NeuralSelfCollFreeConst(AbstractIneqConst):
    model: OrtSelColInferencer  # type: ignore
    base_type: BaseType
    threshold: float = 0.5

    def __init__(self, infer_model: OrtSelColInferencer, robot_Model: RobotModel, base_type: BaseType) -> None:  # type: ignore
        self.model = infer_model  # type: ignore
        self.base_type = base_type
        self.reflect_skrobot_model(robot_Model)
        self.assign_id_value()

    @classmethod
    def load(
        cls,
        urdf_path: Path,
        control_joint_names: List[str],
        robot_model: RobotModel,
        base_type: BaseType,
    ) -> "NeuralSelfCollFreeConst":
        cache_basepath = default_pretrained_basepath()
        model = OrtSelColInferencer.load(
            cache_basepath, urdf_path=urdf_path, eval_joint_names=control_joint_names
        )
        return cls(model, robot_model, base_type)

    def _evaluate(
        self, qs: np.ndarray, with_jacobian: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_point, n_dim = qs.shape

        val_list = []
        grad_list = []
        for q in qs:
            if self.base_type == BaseType.PLANER:
                q = q[:-3]  # because base pose is irrelevant to self collision
            if self.base_type == BaseType.FLOATING:
                q = q[:-6]  # because base pose is irrelevant to self collision

            val, grad = self.model.infer(q, with_grad=with_jacobian)
            val_list.append(self.threshold - val)
            if self.base_type == BaseType.PLANER:
                grad = np.hstack((grad, np.zeros(3)))
            if self.base_type == BaseType.FLOATING:
                grad = np.hstack((grad, np.zeros(6)))
            grad_list.append(-grad)

        valss = np.array(val_list).reshape(n_point, 1)

        if not with_jacobian:
            return valss, self.dummy_jacobian()

        grads = np.array(grad_list)
        jacs = grads.reshape(n_point, 1, n_dim)
        return valss, jacs

    def _reflect_skrobot_model(self, robot_model: Optional[RobotModel]) -> None:
        angles = [robot_model.__dict__[jn].joint_angle() for jn in self.model.joint_names]
        self.model.set_context(np.array(angles))


class COMStabilityConst(AbstractIneqConst):
    # highly experimental feature
    # will be used in humanoid planning only
    dim_cspace: int
    fksolver: KinematicModel
    tinyfk_joint_ids: List[int]
    base_type: BaseType
    model: RobotModel
    com_box: Box

    def __init__(
        self,
        urdfpath: Path,
        joint_names: List[str],
        base_type: BaseType,
        robot_model: RobotModel,
        com_box: Box,
    ):

        dim_cspace = (
            len(joint_names)
            + (base_type == BaseType.PLANER) * 3
            + (base_type == BaseType.FLOATING) * 6
        )

        urdfpath_str = str(urdfpath.expanduser())
        fksolver = KinematicModel(urdfpath_str)
        tinyfk_joint_ids = fksolver.get_joint_ids(joint_names)

        self.dim_cspace = dim_cspace
        self.fksolver = fksolver
        self.tinyfk_joint_ids = tinyfk_joint_ids
        self.base_type = base_type
        self.model = robot_model
        self.com_box = com_box
        self.reflect_skrobot_model(robot_model)
        self.assign_id_value()

    def _evaluate(
        self, qs: np.ndarray, with_jacobian: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:

        n_point, n_dim = qs.shape
        xs, jacs_tmp = self.fksolver.solve_com_fk(
            qs, self.tinyfk_joint_ids, self.base_type, with_jacobian
        )
        jacs = jacs_tmp.reshape(n_point, 1, 3, n_dim)
        # xss: R^{n_point, 3}
        # jacs: R^{3, n_point, dim_cspace}

        assert self.com_box is not None
        sds = -self.com_box.sdf(xs)
        if with_jacobian:
            eps = 1e-7
            grads = np.zeros((n_point, 1, 3))

            for i in range(3):
                xs_plus = copy.deepcopy(xs)
                xs_plus[:, i] += eps
                sds_plus = -self.com_box.sdf(xs_plus)
                grads[:, 0, i] = (sds_plus - sds) / eps

            Js = np.einsum("ijk,ijkl->ijl", grads, jacs)
        else:
            Js = self.dummy_jacobian()
        return sds.reshape((n_point, 1)), Js

    def _reflect_skrobot_model(self, robot_model: Optional[RobotModel]) -> None:
        pass
