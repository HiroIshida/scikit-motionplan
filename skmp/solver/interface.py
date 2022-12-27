from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol, Type, TypeVar, Union

import numpy as np

from skmp.constraint import AbstractEqConst, AbstractIneqConst, BoxConst
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

    def is_constrained(self) -> bool:
        return self.global_eq_const is not None

    def is_satisfied(self, traj: Trajectory) -> bool:
        # check goal satsifaction
        vals, _ = self.goal_const.evaluate_single(traj[-1], with_jacobian=False)
        if vals.dot(vals) > self.eqconst_admissible_mse:
            return False

        # check ineq satisfaction
        assert self.global_ineq_const is not None
        valss, _ = self.global_ineq_const.evaluate(traj.numpy(), with_jacobian=False)
        return bool(np.all(valss > 0))


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
