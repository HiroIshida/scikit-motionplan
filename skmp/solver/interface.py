from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol, Type, TypeVar

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

    def is_constrained(self) -> bool:
        return self.global_eq_const is not None


class ConfigProtocol(Protocol):
    n_max_eval: int
    motion_step_box: np.ndarray


class ResultProtocol(Protocol):
    traj: Optional[Trajectory]
    time_elapsed: float


class AbstractSolver(ABC):
    @classmethod
    @abstractmethod
    def setup(cls: Type[SolverT], problem: Problem, config: ConfigProtocol) -> SolverT:
        ...

    @abstractmethod
    def solve(self) -> ResultProtocol:
        ...
