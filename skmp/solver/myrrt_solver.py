import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Type, TypeVar

import numpy as np

from skmp.satisfy import SatisfactionConfig, SatisfactionResult, satisfy_by_optimization
from skmp.solver._manifold_rrt_solver import ManifoldRRTConfig, ManifoldRRTConnect
from skmp.solver.interface import AbstractScratchSolver, Problem
from skmp.trajectory import Trajectory


@dataclass
class MyRRTConfig:
    n_max_call: int
    n_max_satisfaction_trial: int = 100
    satisfaction_conf: Optional[SatisfactionConfig] = SatisfactionConfig()

    @property
    def sample_goal_first(self) -> bool:
        return self.satisfaction_conf is not None


class TerminateState(Enum):
    SUCCESS = 1
    FAIL_SATISFACTION = 2
    FAIL_PLANNING = 3


@dataclass
class MyRRTResult:
    traj: Optional[Trajectory]
    time_elapsed: float
    n_call: int
    terminate_state: TerminateState

    @classmethod
    def abnormal(cls, time_elapsed: float) -> "MyRRTResult":
        return cls(None, time_elapsed, -1, TerminateState.FAIL_SATISFACTION)


MyRRTSolverT = TypeVar("MyRRTSolverT", bound="MyRRTSolverBase")


@dataclass
class MyRRTSolverBase(AbstractScratchSolver[MyRRTConfig, MyRRTResult]):
    config: MyRRTConfig
    problem: Optional[Problem] = None

    @classmethod
    def get_result_type(cls) -> Type[MyRRTResult]:
        return MyRRTResult

    @classmethod
    def init(cls: Type[MyRRTSolverT], config: MyRRTConfig) -> MyRRTSolverT:
        return cls(config)

    def setup(self, problem: Problem) -> None:
        self.problem = problem

    def project(self, q: np.ndarray) -> Optional[np.ndarray]:
        assert self.problem is not None

        if self.problem.global_eq_const is None:
            return q
        else:
            res = satisfy_by_optimization(
                self.problem.global_eq_const, self.problem.box_const, None, q
            )
            if res.success:
                return res.q
            else:
                return None

    def is_valid(self, q: np.ndarray) -> bool:
        assert self.problem is not None
        if self.problem.global_ineq_const is None:
            return True
        val, _ = self.problem.global_ineq_const.evaluate_single(q, False)
        return bool(np.all(val > 0))


@dataclass
class MyRRTConnectSolver(MyRRTSolverBase):
    def solve(self, init_traj: Optional[Trajectory] = None) -> MyRRTResult:
        """solve problem with maybe a solution guess"""

        assert init_traj is None, "don't support replanning"
        assert self.problem is not None

        ts = time.time()

        assert self.config.sample_goal_first, "goal must be sampled before in rrt-connect"
        satisfy_result: SatisfactionResult
        for _ in range(self.config.n_max_satisfaction_trial):
            satisfy_result = satisfy_by_optimization(
                self.problem.goal_const,
                self.problem.box_const,
                self.problem.global_ineq_const,
                None,
            )
            if satisfy_result.success:
                break

        if not satisfy_result.success:
            return MyRRTResult.abnormal(time.time() - ts)

        conf = ManifoldRRTConfig(self.config.n_max_call)

        rrtconnect = ManifoldRRTConnect(
            self.problem.start,
            satisfy_result.q,
            self.problem.box_const.lb,
            self.problem.box_const.ub,
            self.problem.motion_step_box,
            self.project,
            self.is_valid,
            conf,
        )
        is_success = rrtconnect.solve()
        if is_success:
            traj = Trajectory(list(rrtconnect.get_solution()))
            return MyRRTResult(
                traj, time.time() - ts, rrtconnect.n_extension_trial, TerminateState.SUCCESS
            )
        else:
            return MyRRTResult(
                None, time.time() - ts, self.config.n_max_call, TerminateState.FAIL_PLANNING
            )
