from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol, Type, TypeVar, Union

import numpy as np

from skmp.constraint import AbstractEqConst, AbstractIneqConst, BoxConst
from skmp.solver.motion_step_box import is_valid_motion_step
from skmp.trajectory import Trajectory

GoalConstT = TypeVar("GoalConstT")
GlobalIneqConstT = TypeVar("GlobalIneqConstT")
GlobalEqConstT = TypeVar("GlobalEqConstT")
SolverT = TypeVar("SolverT", bound="AbstractSolver")
ProblemT = TypeVar("ProblemT", bound="Problem")


@dataclass
class Problem:
    start: np.ndarray
    box_const: BoxConst
    goal_const: AbstractEqConst
    global_ineq_const: Optional[AbstractIneqConst]
    global_eq_const: Optional[AbstractEqConst]
    eqconst_admissible_mse: float = 1e-6
    motion_step_box: Union[float, np.ndarray] = 0.1

    def is_constrained(self) -> bool:
        return self.global_eq_const is not None

    def is_satisfied(self, traj: Trajectory) -> bool:
        # check goal satsifaction
        vals, _ = self.goal_const.evaluate_single(traj[-1], with_jacobian=False)
        if vals.dot(vals) > self.eqconst_admissible_mse:
            return False

        # check ineq satisfaction
        if self.global_ineq_const is not None:
            valss, _ = self.global_ineq_const.evaluate(traj.numpy(), with_jacobian=False)
            if not np.all(valss > 0):
                return False

        # check motion step box
        if self.global_ineq_const is not None or self.global_eq_const is not None:
            assert self.global_eq_const is None, "currently not implemented yet"
            for i in range(len(traj) - 1):
                q1, q2 = traj[i], traj[i + 1]
                if not is_valid_motion_step(self.motion_step_box, q1, q2, self.global_ineq_const):  # type: ignore[arg-type]
                    return False
        return True


class ConfigProtocol(Protocol):
    n_max_eval: int
    motion_step_box: Union[np.ndarray, float]


class ResultProtocol(Protocol):
    traj: Optional[Trajectory]
    time_elapsed: float


class AbstractSolver(ABC):
    @classmethod
    @abstractmethod
    def setup(cls: Type[SolverT], problem: Problem, config: ConfigProtocol) -> SolverT:
        ...

    @abstractmethod
    def solve(self, init_traj: Optional[Trajectory] = None) -> ResultProtocol:
        ...
