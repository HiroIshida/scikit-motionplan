from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import (
    Dict,
    Generic,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Protocol,
    Tuple,
    Union,
)

import numpy as np

from skmp.constraint import (
    AbstractEqConst,
    AbstractIneqConst,
    ConstraintT,
    EqCompositeConst,
    IneqCompositeConst,
)


class GlobalConstraintProtocol(Protocol):
    def evaluate(self, __qs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...


@dataclass
class TrajectoryConstraint(ABC, Mapping, Generic[ConstraintT]):
    n_dof: int
    n_wp: int
    local_constraint_table: MutableMapping[int, ConstraintT]  # constraint on sigle waypoint
    global_constraint_table: List[GlobalConstraintProtocol]

    def add(self, idx: int, constraint: ConstraintT, force: bool = False) -> None:
        assert idx > -1

        if idx > self.n_wp - 1:
            message = "index {} exceeds the waypoint number {}".format(idx, self.n_wp)
            raise ValueError(message)

        if idx not in self.local_constraint_table:
            self.local_constraint_table[idx] = constraint
        else:
            if force:
                self.local_constraint_table[idx] = constraint  # overwrite
            else:
                self.composite(idx, constraint)

    @abstractmethod
    def composite(self, idx: int, constraint: ConstraintT) -> None:
        ...

    def add_goal_constraint(self, constraint: ConstraintT):
        self.local_constraint_table[self.n_wp - 1] = constraint

    def get_goal_constraint(self) -> ConstraintT:
        return self.local_constraint_table[self.n_wp - 1]

    def is_homogeneous(self) -> bool:
        """check if the same constraint is enforced over all waypoints"""
        no_local_constraint = len(self.local_constraint_table) == 0
        if no_local_constraint:
            return True

        if len(self) < self.n_wp:
            return False
        ids = [id(cons) for cons in self.local_constraint_table.values()]
        is_all_same_cons = len(set(ids)) == 1
        return is_all_same_cons

    def is_start_goal(self) -> bool:
        """check if constraint is enforced only both at start and goal
        Most of path planning problem falls into this type.
        """
        if len(self) != 2:
            return False
        if 0 not in self:
            return False
        if (self.n_wp - 1) not in self:
            return False
        return True

    def __getitem__(self, key: int) -> ConstraintT:
        return self.local_constraint_table.__getitem__(key)

    def __iter__(self) -> Iterator[int]:
        return self.local_constraint_table.__iter__()

    def __len__(self) -> int:
        return len(self.local_constraint_table)

    def evaluate(self, traj_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # first evaluate local constraints
        traj = traj_vector.reshape(-1, self.n_dof)

        local_table = self.local_constraint_table

        if len(local_table) > 0:
            value_list = []
            jacobi_dict: Dict[int, np.ndarray] = {}

            for i in local_table.keys():
                # compute constraint at waypoint at idx
                const = local_table[i]
                q = traj[i]
                value, jacobi = const.evaluate_single(q, True)  # unzip
                value_list.append(value)
                jacobi_dict[i] = jacobi
            local_value_total = np.hstack(value_list)

        elif len(local_table) == 0:
            local_value_total = np.array([])

        local_dim_codomain_total = len(local_value_total)

        local_jacobi_total = np.zeros((local_dim_codomain_total, len(traj_vector)))
        head = 0
        for i in local_table.keys():
            idx_start = i * self.n_dof
            idx_end = (i + 1) * self.n_dof
            jacobi = jacobi_dict[i]
            dim_codomain = jacobi.shape[0]
            local_jacobi_total[head : head + dim_codomain, idx_start:idx_end] = jacobi
            head += dim_codomain

        # then evaluate global constraints
        if len(self.global_constraint_table) > 0:
            values, jacobis = zip(
                *[cons.evaluate(traj_vector) for cons in self.global_constraint_table]
            )
            global_value_total = np.hstack(values)
            global_jacobi_total = np.vstack(jacobis)
        else:
            global_value_total = np.empty(0)
            global_jacobi_total = np.empty((0, len(traj_vector)))

        value_total = np.hstack([local_value_total, global_value_total])
        jacobi_total = np.vstack([local_jacobi_total, global_jacobi_total])

        return value_total, jacobi_total


@dataclass
class TrajectoryEqualityConstraint(TrajectoryConstraint[AbstractEqConst]):
    def composite(self, idx: int, constraint: AbstractEqConst) -> None:
        c_new = EqCompositeConst([self.local_constraint_table[idx], constraint])
        self.local_constraint_table[idx] = c_new


@dataclass
class TrajectoryInequalityConstraint(TrajectoryConstraint[AbstractIneqConst]):
    def __post_init__(self):
        assert self.is_homogeneous()  # temporary limitation

    def composite(self, idx: int, constraint: AbstractIneqConst) -> None:
        c_new = IneqCompositeConst([self.local_constraint_table[idx], constraint])
        self.local_constraint_table[idx] = c_new

    @classmethod
    def create_homogeneous(
        cls,
        n_wp: int,
        n_dof: int,
        local_const: Optional[AbstractIneqConst] = None,
        global_consts: Optional[List[GlobalConstraintProtocol]] = None,
    ) -> "TrajectoryInequalityConstraint":
        table: Dict[int, AbstractIneqConst] = {}
        if local_const is not None:
            for i in range(n_wp):
                table[i] = local_const

        if global_consts is None:
            global_consts = []
        return cls(n_dof, n_wp, table, global_consts)

    def evaluate(self, traj_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return super().evaluate(traj_vector)


@dataclass
class MotionStepInequalityConstraint:
    """NOTE: this is not TrajectoryConstraint in the sense that
    this doesn't have the table
    """

    n_dof: int
    n_wp: int
    motion_step_box: Union[float, np.ndarray, None] = None

    def __post_init__(self):
        self.box_constraint_matrix
        self.box_width_repeated

    @cached_property
    def box_constraint_matrix(self) -> np.ndarray:
        base = -np.eye(self.n_wp)
        for i in range(self.n_wp - 1):
            base[i, i + 1] = 1
        base = base[:-1, :]
        mat = np.kron(base, np.eye(self.n_dof))
        return mat

    @cached_property
    def box_width_repeated(self) -> np.ndarray:
        assert self.motion_step_box is not None
        if isinstance(self.motion_step_box, float):
            width = np.ones(self.n_dof) * self.motion_step_box
        else:
            width = self.motion_step_box
        W = np.hstack([width] * (self.n_wp - 1))
        return W

    def dim_codomain(self):
        return self.n_dof * (self.n_wp - 1)

    def evaluate(self, traj_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # original constratint -W < AX < W will be split into the two
        # left: 0 < AX + W
        # right: 0 < -AX + W
        A = self.box_constraint_matrix
        W = self.box_width_repeated
        AX = A @ traj_vector

        left_eval = AX + W
        left_jac = A

        right_eval = -AX + W
        right_jac = -A

        eval_concat = np.hstack((left_eval, right_eval))
        jac_concat = np.vstack((left_jac, right_jac))
        return eval_concat, jac_concat
