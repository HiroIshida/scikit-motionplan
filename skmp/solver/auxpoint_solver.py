import copy
from dataclasses import dataclass
from typing import Optional, Type

import numpy as np
import tqdm

from skmp.constraint import ConfigPointConst
from skmp.satisfy import SatisfactionConfig, satisfy_by_optimization
from skmp.solver.interface import AbstractScratchSolver, Problem
from skmp.solver.nlp_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.trajectory import Trajectory


@dataclass
class AuxSolverConfig:
    n_call: int = 100  # connection trial
    n_sample: int = 300


@dataclass
class AuxSolverResult:
    traj: Optional[Trajectory]
    time_elapsed: float
    n_call: int


@dataclass
class AuxSolver(AbstractScratchSolver[AuxSolverConfig, AuxSolverResult]):
    config: AuxSolverConfig
    problem: Optional[Problem] = None

    @classmethod
    def get_result_type(cls) -> Type[AuxSolverResult]:
        return AuxSolverResult

    @classmethod
    def init(cls, config: AuxSolverConfig) -> "AuxSolver":
        return cls(config, None)

    def setup(self, problem: Problem) -> None:
        self.problem = problem

    def solve(self, init_traj: Optional[Trajectory] = None) -> AuxSolverResult:
        assert init_traj is None

        # adhoc
        assert self.problem is not None
        assert self.problem.global_eq_const is not None

        start_original = copy.deepcopy(self.problem.start)
        goal_original = copy.copy(self.problem.goal_const)

        def reset_problem():
            # I want to deepcopy the problem, but cannot because it wraps c++ stuff
            # so, before returning this function, we restore the problem
            self.problem.start = start_original
            self.problem.goal_const = goal_original

        # sample box
        q_from_start_list = []
        q_to_goal_list = []

        if isinstance(self.problem.goal_const, ConfigPointConst):
            ub_q_start_goal = (
                np.maximum(self.problem.start, self.problem.goal_const.desired_angles) + 0.5
            )
            lb_q_start_goal = (
                np.minimum(self.problem.start, self.problem.goal_const.desired_angles) - 0.5
            )
            ub = np.minimum(self.problem.box_const.ub, ub_q_start_goal)
            lb = np.maximum(self.problem.box_const.lb, lb_q_start_goal)
        else:
            ub = self.problem.box_const.ub
            lb = self.problem.box_const.lb

        w = ub - lb
        sqp_config = SQPBasedSolverConfig(
            50, n_max_call=60, motion_step_satisfaction="explicit", verbose=False, ctol_eq=1e-3
        )
        sqp = SQPBasedSolver.init(sqp_config)

        for _ in tqdm.tqdm(range(self.config.n_sample)):
            q = lb + np.random.rand(len(w)) * w
            conf = SatisfactionConfig(n_max_eval=100)
            res = satisfy_by_optimization(
                self.problem.global_eq_const,
                self.problem.box_const,
                self.problem.global_ineq_const,
                q,
                conf,
            )
            if res.success:
                print("success projection")
                # q_manifold_list.append(q)
                self.problem.start = start_original
                self.problem.goal_const = ConfigPointConst(res.q)

                sqp.setup(self.problem)
                res1 = sqp.solve()
                success1 = res1.traj is not None
                if success1:
                    q_from_start_list.append(q)
                    print("success1")

                self.problem.start = res.q
                self.problem.goal_const = goal_original
                sqp.setup(self.problem)
                res2 = sqp.solve()
                success2 = res2.traj is not None
                if success2:
                    q_to_goal_list.append(q)
                    print("success2")

                if success1 and success2:
                    traj = Trajectory(np.vstack([res1.traj.numpy()[:-1], res2.traj.numpy()]))  # type: ignore
                    reset_problem()
                    return AuxSolverResult(traj, 0, 100)
        return AuxSolverResult(None, 0, 100)
