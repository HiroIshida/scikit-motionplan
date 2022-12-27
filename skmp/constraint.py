import copy
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from skmp.kinematics import CollisionKinmaticsMapProtocol
from skmp.utils.urdf import URDF, JointLimit


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


class AbstractIneqConst(AbstractConst):
    ...


class AbstractEqConst(AbstractConst):
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
            jac = np.vstack((np.eye(len(f_lower)), -np.eye(len(f_upper))))
            jac = jac.reshape(n_point, -1, dim)
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

    def evaluate(
        self, qs: np.ndarray, with_jacobian: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def evaluate(
        self, qs: np.ndarray, with_jacobian: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_point, dim = qs.shape
        val = qs - self.desired_angles
        if with_jacobian:
            jac = np.array([np.eye(dim) for _ in range(n_point)])
        else:
            jac = self.dummy_jacobian()
        return val, jac
