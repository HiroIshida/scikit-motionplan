import copy
import importlib
import itertools
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Type, TypeVar

import numpy as np
from skrobot.coordinates import Coordinates, rpy_angle
from skrobot.model import RobotModel

from skmp.kinematics import (
    ArticulatedCollisionKinematicsMap,
    CollisionKinmaticsMapProtocol,
    KinematicsMapProtocol,
)
from skmp.utils.urdf import URDF, JointLimit

if importlib.find_loader("selcol") is not None:
    SELCOL_FOUND = True
    from selcol.file import default_cache_basepath
    from selcol.runtime import OrtSelColInferencer
else:
    SELCOL_FOUND = False


class AbstractConst:
    @abstractmethod
    def evaluate(self, qs: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def evaluate_single(self, q: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        assert q.ndim == 1
        f, jac = self.evaluate(np.expand_dims(q, axis=0), with_jacobian)
        return f[0], jac[0]

    def dummy_jacobian(self) -> np.ndarray:
        return np.array([[np.nan]])

    @classmethod
    @abstractmethod
    def is_equality(cls) -> bool:
        ...


ConstraintT = TypeVar("ConstraintT", bound=AbstractConst)


class AbstractIneqConst(AbstractConst):
    @classmethod
    def is_equality(cls) -> bool:
        return False


class AbstractEqConst(AbstractConst):
    @classmethod
    def is_equality(cls) -> bool:
        return True


CompositeConstT = TypeVar("CompositeConstT", bound="_CompositeConst")


@dataclass
class _CompositeConst(AbstractConst):
    const_list: List[AbstractConst]

    @classmethod
    def composite(cls: Type[CompositeConstT], const_list: List[AbstractConst]) -> CompositeConstT:
        for const in const_list:
            assert const.is_equality() == cls.is_equality()
        return cls(const_list)

    def evaluate(self, qs: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
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


class IneqCompositeConst(AbstractIneqConst, _CompositeConst):
    ...


class EqCompositeConst(AbstractEqConst, _CompositeConst):
    ...


@dataclass
class BoxConst(AbstractIneqConst):
    lb: np.ndarray
    ub: np.ndarray

    @classmethod
    def from_urdf(
        cls,
        urdf_path: Path,
        joint_names: List[str],
        base_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):

        urdfpath_str = str(urdf_path.expanduser())
        urdf = URDF.load(urdfpath_str)
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
            for i in range(3):
                b_min.append(lb[i])
                b_max.append(ub[i])

        return cls(np.array(b_min), np.array(b_max))

    def evaluate(self, qs: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
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

    def sample(self) -> np.ndarray:
        w = self.ub - self.lb
        return np.random.rand(len(w)) * w + self.lb


@dataclass
class CollFreeConst(AbstractIneqConst):
    colkin: CollisionKinmaticsMapProtocol
    sdf: Callable[[np.ndarray], np.ndarray]
    dim_tspace: int
    clearance: float = 0.0

    def evaluate(self, qs: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        """compute signed distance of feature points and its jacobian
        input:
            qs: R^{n_point, dim_cspace}
            clearance: clearance
        output:
            fss: R^{n_point, n_feature}
            Jss: R^{n_point, n_feature, dim_cspace}
        """
        assert self.sdf is not None
        n_point, dim_cspace = qs.shape
        n_feature = self.colkin.n_feature

        # xss: R^{n_point, n_feature * dim_tspace}
        # jss: R^{n_point, n_feature, dim_tspace, dim_cspace}
        xss, jacss = self.colkin.map(qs)  # ss refere to points of points

        xs_stacked = xss.reshape((n_point * n_feature, self.dim_tspace))
        sds_stacked = self.sdf(xs_stacked)

        # compute sd_vals_stacked
        margin_radius = np.tile(np.array(self.colkin.radius_list), n_point) + self.clearance
        fs_stacked = sds_stacked - margin_radius
        fss = fs_stacked.reshape(n_point, n_feature)

        # compute jacobian by chain rule
        if with_jacobian:
            eps = 1e-7
            grads_stacked = np.zeros((n_feature * n_point, self.dim_tspace))

            for i in range(self.dim_tspace):
                xs_stacked_plus = copy.deepcopy(xs_stacked)
                xs_stacked_plus[:, i] += eps
                sds_stacked_plus = self.sdf(xs_stacked_plus)
                grads_stacked[:, i] = (sds_stacked_plus - sds_stacked) / eps
            gradss = grads_stacked.reshape((n_point, n_feature, self.dim_tspace))
            Jss = np.einsum("ijk,ijkl->ijl", gradss, jacss)
        else:
            Jss = self.dummy_jacobian()
        return fss, Jss


@dataclass
class ConfigPointConst(AbstractEqConst):
    desired_angles: np.ndarray

    def evaluate(self, qs: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        n_point, dim = qs.shape
        val = qs - self.desired_angles
        if with_jacobian:
            jac = np.array([np.eye(dim) for _ in range(n_point)])
        else:
            jac = self.dummy_jacobian()
        return val, jac


@dataclass
class PoseConstraint(AbstractEqConst):
    desired_poses: List[np.ndarray]
    efkin: KinematicsMapProtocol

    def evaluate(
        self, qs: np.ndarray, with_jacobian: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_point, n_dim = qs.shape
        xs_tmp, jacs_tmp = self.efkin.map(qs)
        xs = xs_tmp.reshape(n_point, -1)
        jacs = jacs_tmp.reshape(n_point, -1, n_dim)
        n_point, dim = qs.shape

        target = np.hstack(self.desired_poses)
        values = xs - target
        return values, jacs

    @classmethod
    def from_skrobot_coords(
        cls, co_list: List[Coordinates], efkin: KinematicsMapProtocol
    ) -> "PoseConstraint":
        vector_list = []
        for co in co_list:
            pos = co.worldpos()
            ypr = rpy_angle(co.worldrot())[0]
            rpy = np.flip(ypr)
            vector = np.hstack([pos, rpy])
            vector_list.append(vector)
        return cls(vector_list, efkin)


@dataclass
class PairWiseSelfCollFreeConst(AbstractIneqConst):
    colkin: ArticulatedCollisionKinematicsMap
    check_sphere_id_pairs: List[Tuple[int, int]]
    check_sphere_pair_sqdists: np.ndarray  # pair sqdist means (r1 + r2) ** 2

    @classmethod
    def from_colkin(cls, colkin: ArticulatedCollisionKinematicsMap) -> "PairWiseSelfCollFreeConst":
        # here in this constructor, we will filter out collision pair which is already collide
        # at the initial pose np.zeros(n_dof)

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

        # compute inter-sphere distances when q = np.zeros(n_dof)
        q_init = np.zeros(colkin.dim_cspace)
        sqdists, _ = colkin.fksolver.compute_inter_link_sqdists(
            [q_init], all_index_pairs, colkin.tinyfk_joint_ids, with_base=colkin.with_base
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

        return cls(colkin, valid_sphere_id_pairs, valid_sphere_pair_dists**2)

    def evaluate(self, qs: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        n_sample, n_dim = qs.shape

        sqdists_stacked, grads_stacked = self.colkin.fksolver.compute_inter_link_sqdists(
            qs,
            self.check_sphere_id_pairs,
            self.colkin.tinyfk_joint_ids,
            with_base=self.colkin.with_base,
            with_jacobian=with_jacobian,
        )
        sqdistss = sqdists_stacked.reshape(n_sample, -1)
        valuess = sqdistss - self.check_sphere_pair_sqdists

        if not with_jacobian:
            return valuess, self.dummy_jacobian()

        gradss = grads_stacked.reshape(n_sample, -1, n_dim)
        return valuess, gradss


@dataclass
class NeuralSelfCollFreeConst(AbstractIneqConst):
    model: OrtSelColInferencer  # type: ignore
    threshold: float = 0.5

    @classmethod
    def load(cls, urdf_path: Path, control_joint_names: List[str]) -> "NeuralSelfCollFreeConst":
        assert SELCOL_FOUND
        cache_basepath = default_cache_basepath()
        model = OrtSelColInferencer.load(
            cache_basepath, urdf_path=urdf_path, eval_joint_names=control_joint_names
        )
        return cls(model)

    def reflect_skrobot_model(self, robot_model: RobotModel):
        angles = [robot_model.__dict__[jn].joint_angle() for jn in self.model.joint_names]
        self.model.set_context(np.array(angles))

    def evaluate(
        self, qs: np.ndarray, with_jacobian: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_point, n_dim = qs.shape

        val_list = []
        grad_list = []
        for q in qs:
            val, grad = self.model.infer(q, with_grad=with_jacobian)
            val_list.append(self.threshold - val)
            grad_list.append(-grad)

        valss = np.array(val_list).reshape(n_point, 1)

        if not with_jacobian:
            return valss, self.dummy_jacobian()

        grads = np.array(grad_list)
        jacs = grads.reshape(n_point, 1, n_dim)
        return valss, jacs
