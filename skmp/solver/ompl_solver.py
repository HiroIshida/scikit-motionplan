import time
from dataclasses import dataclass
from typing import List, Optional, Type, TypeVar

import numpy as np
from ompl import Algorithm, Planner

from skmp.satisfy import SatisfactionResult, satisfy_by_optimization
from skmp.solver.interface import AbstractSolver, Problem
from skmp.trajectory import Trajectory


@dataclass
class OMPLSolverConfig:
    n_max_eval: int = 2000
    n_max_satisfaction_trial: int = 100
    algorithm: Algorithm = Algorithm.RRTConnect


OMPLSolverT = TypeVar("OMPLSolverT", bound="OMPLSolver")


@dataclass
class OMPLSolver(AbstractSolver):
    problem: Problem
    config: OMPLSolverConfig
    planner: Planner

    @dataclass
    class Result:
        traj: Optional[Trajectory]
        time_elapsed: float

    @classmethod
    def setup(cls: Type[OMPLSolverT], problem: Problem, config: Optional[OMPLSolverConfig] = None) -> OMPLSolverT:  # type: ignore[override]

        if config is None:
            config = OMPLSolverConfig()

        assert not problem.is_constrained()

        def is_valid(q_: List[float]) -> bool:
            q = np.array(q_)
            if problem.global_ineq_const is None:
                return True
            else:
                val, _ = problem.global_ineq_const.evaluate_single(q, with_jacobian=False)
                return bool(np.all(val > 0))

        lb = problem.box_const.lb
        ub = problem.box_const.ub

        planner = Planner(
            lb,
            ub,
            is_valid,
            config.n_max_eval,
            validation_box=problem.motion_step_box,
            algo=config.algorithm,
        )
        return cls(problem, config, planner)

    def solve(self, init_traj: Optional[Trajectory] = None):
        ts = time.time()

        result: Optional[SatisfactionResult] = None
        for _ in range(self.config.n_max_satisfaction_trial):
            result = satisfy_by_optimization(
                self.problem.goal_const,
                self.problem.box_const,
                self.problem.global_ineq_const,
                None,
            )
            if result.success:
                break
        assert result is not None
        if not result.success:
            return self.Result(None, time.time() - ts)
        q_start = self.problem.start
        q_goal = result.q
        plan_result = self.planner.solve(q_start, q_goal)
        if plan_result is not None:
            traj = Trajectory(plan_result)
        else:
            traj = None
        return self.Result(traj, time.time() - ts)
