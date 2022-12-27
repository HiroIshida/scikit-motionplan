import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Type, TypeVar

import numpy as np
from ompl import Algorithm, Planner

from skmp.constraint import AbstractEqConst, AbstractIneqConst, BoxConst
from skmp.solver_interface import AbstractSolver, ConfigProtocol, Problem
from skmp.trajectory import Trajectory


@dataclass
class OMPLSolverConfig(ConfigProtocol):
    n_max_eval: int
    motion_step_box: np.ndarray
    algorithm: Algorithm = Algorithm.RRTConnect


OMPLSolverT = TypeVar("OMPLSolverT", bound="OMPLSolverBase")


@dataclass
class OMPLSolverBase(AbstractSolver):
    problem: Problem
    planner: Planner

    @dataclass
    class Result:
        traj: Optional[Trajectory]
        time_elapsed: float

    @classmethod
    def setup(cls: Type[OMPLSolverT], problem: Problem, config: OMPLSolverConfig) -> OMPLSolverT:
        assert not problem.is_constrained()

        def is_valid(x: np.ndarray) -> bool:
            if problem.global_ineq_const is None:
                return True
            else:
                val, _ = problem.global_ineq_const.evaluate_single(x, with_jacobian=False)
                return val > 0

        lb = problem.box_constraint.lb
        ub = problem.box_constraint.ub

        planner = Planner(
            lb,
            ub,
            is_valid,
            config.n_max_eval,
            validation_box=config.n_max_eval,
            algo=config.algorithm,
        )
        return cls(problem, planner)

    def solve(self):
        ts = time.time()
        goal = self.sample_goal(
            self.problem.goal_const, self.problem.box_constraint, self.problem.global_ineq_const
        )
        plan_result = self.planner.solve(self.problem.start, goal)
        elapsed = time.time() - ts
        if plan_result is not None:
            traj = Trajectory(plan_result)
        else:
            traj = None
        return self.Result(traj, elapsed)

    @staticmethod
    @abstractmethod
    def sample_goal(
        eq_const: AbstractEqConst, box_const: BoxConst, ineq_const: Optional[AbstractIneqConst]
    ) -> np.ndarray:
        ...
