from dataclasses import dataclass
from enum import Enum
from typing import Optional, Type, TypeVar

import numpy as np

from skmp.satisfy import SatisfactionConfig, SatisfactionResult, satisfy_by_optimization
from skmp.solver._manifold_rrt_solver import (
    InvalidStartPosition,
    ManifoldRRT,
    ManifoldRRTConfig,
    ManifoldRRTConnect,
)
from skmp.solver.interface import AbstractScratchSolver, Problem
from skmp.trajectory import Trajectory


@dataclass
class MyRRTConfig:
    n_max_call: int
    n_max_satisfaction_trial: int = 100
    satisfaction_conf: Optional[SatisfactionConfig] = SatisfactionConfig()
    n_subgoal: int = 4  # used only when init_traj is specified
    timeout: Optional[int] = None

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
    time_elapsed: Optional[float]
    n_call: int
    terminate_state: TerminateState

    @classmethod
    def abnormal(cls) -> "MyRRTResult":
        return cls(None, None, -1, TerminateState.FAIL_SATISFACTION)


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

    def _setup(self, problem: Problem) -> None:
        pass

    def project(self, q: np.ndarray, collision_aware: bool = False) -> Optional[np.ndarray]:
        assert self.problem is not None

        if self.problem.global_eq_const is None:
            return q
        else:
            ineq_const = self.problem.global_ineq_const if collision_aware else None
            res = satisfy_by_optimization(
                self.problem.global_eq_const, self.problem.box_const, ineq_const, q
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
class MyRRTSolver(MyRRTSolverBase):
    def _solve(self, init_traj: Optional[Trajectory] = None) -> MyRRTResult:
        """solve problem with maybe a solution guess"""
        assert init_traj is None, "don't support replanning"
        assert self.problem is not None

        inner_conf = ManifoldRRTConfig(self.config.n_max_call)

        assert not self.config.sample_goal_first  # TODO

        def f_goal_project(q) -> Optional[np.ndarray]:
            assert self.problem is not None
            # box_const = copy.deepcopy(self.problem.box_const)
            # box_const.lb = np.maximum(box_const.lb, q - self.problem.motion_step_box * 10)
            # box_const.ub = np.minimum(box_const.ub, q + self.problem.motion_step_box * 10)

            goal_const = self.problem.goal_const
            satis_conf = SatisfactionConfig(n_max_eval=50, ftol=1e-3, acceptable_error=1e-3)
            res = satisfy_by_optimization(
                goal_const,
                self.problem.box_const,
                self.problem.global_ineq_const,
                q,
                config=satis_conf,
            )
            if res.success:
                return res.q
            else:
                return None

        rrt = ManifoldRRT(
            self.problem.start,
            f_goal_project,
            self.problem.box_const.lb,
            self.problem.box_const.ub,
            self.problem.motion_step_box,
            self.project,
            self.is_valid,
            config=inner_conf,
        )
        is_success = rrt.solve()
        if is_success:
            traj = Trajectory(list(rrt.get_solution()))
            return MyRRTResult(traj, None, rrt.n_extension_trial, TerminateState.SUCCESS)
        else:
            return MyRRTResult(None, None, self.config.n_max_call, TerminateState.FAIL_PLANNING)


@dataclass
class MyRRTConnectSolver(MyRRTSolverBase):
    def _solve(self, init_traj: Optional[Trajectory] = None) -> MyRRTResult:
        """solve problem with maybe a solution guess"""

        if init_traj is not None:
            return self.solve_with_initial_solution(init_traj)

        assert self.problem is not None

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
            return MyRRTResult.abnormal()

        conf = ManifoldRRTConfig(self.config.n_max_call)

        rrtconnect = ManifoldRRTConnect(
            self.problem.start,
            satisfy_result.q,
            self.problem.box_const.lb,
            self.problem.box_const.ub,
            self.problem.motion_step_box,
            self.project,
            self.is_valid,
            config=conf,
        )
        try:
            is_success = rrtconnect.solve()
        except InvalidStartPosition:
            return MyRRTResult.abnormal()

        if is_success:
            traj = Trajectory(list(rrtconnect.get_solution()))
            return MyRRTResult(traj, None, rrtconnect.n_extension_trial, TerminateState.SUCCESS)
        else:
            return MyRRTResult(None, None, self.config.n_max_call, TerminateState.FAIL_PLANNING)

    def solve_with_initial_solution(self, init_traj: Trajectory):
        n_subgoal = self.config.n_subgoal
        subgoal_cands = init_traj.resample(n_subgoal + 1)[1:]  # +1 for initial state

        assert self.problem is not None

        q_start = self.problem.start
        n_call_sofar = 0
        q_seq_list = []
        for i in range(n_subgoal):
            subgoal_cand = subgoal_cands[i]

            if i == n_subgoal - 1:
                satisfy_result = satisfy_by_optimization(
                    self.problem.goal_const,
                    self.problem.box_const,
                    self.problem.global_ineq_const,
                    subgoal_cand,
                )
            else:
                satisfy_result = satisfy_by_optimization(
                    None,
                    self.problem.box_const,
                    self.problem.global_ineq_const,
                    subgoal_cand,
                )

            if not satisfy_result.success:
                return MyRRTResult.abnormal()

            conf = ManifoldRRTConfig(self.config.n_max_call - n_call_sofar)

            rrtconnect = ManifoldRRTConnect(
                q_start,
                satisfy_result.q,
                self.problem.box_const.lb,
                self.problem.box_const.ub,
                self.problem.motion_step_box,
                self.project,
                self.is_valid,
                config=conf,
            )
            try:
                is_success = rrtconnect.solve()
            except InvalidStartPosition:
                return MyRRTResult.abnormal()

            n_call_sofar += rrtconnect.n_extension_trial

            if not is_success:
                return MyRRTResult(None, None, n_call_sofar, TerminateState.FAIL_PLANNING)
            q_seq = rrtconnect.get_solution()
            q_start = q_seq[-1]
            if i == 1:
                q_seq_list.append(q_seq)
            else:
                q_seq_list.append(q_seq[1:])

        traj = Trajectory(list(np.vstack(q_seq_list)))
        # all sub goal solved
        return MyRRTResult(traj, None, n_call_sofar, TerminateState.SUCCESS)
