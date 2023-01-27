import time
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np
from ompl import (
    Algorithm,
    ConstrainedPlanner,
    InvalidProblemError,
    LightningDB,
    LightningPlanner,
    Planner,
    RepairPlanner,
    _OMPLPlannerBase,
)

from skmp.satisfy import SatisfactionResult, satisfy_by_optimization
from skmp.solver.interface import (
    AbstractDataDrivenSolver,
    AbstractScratchSolver,
    AbstractSolver,
    Problem,
)
from skmp.trajectory import Trajectory


@dataclass
class OMPLSolverConfig:
    n_max_call: int = 2000
    n_max_satisfaction_trial: int = 100
    algorithm: Algorithm = Algorithm.RRTConnect
    algorithm_range: Optional[float] = None
    simplify: bool = False


class TerminateState(Enum):
    SUCCESS = 1
    FAIL_SATISFACTION = 2
    FAIL_PLANNING = 3


@dataclass
class OMPLSolverResult:
    traj: Optional[Trajectory]
    time_elapsed: float
    n_call: int
    terminate_state: TerminateState

    @classmethod
    def abnormal(cls, time_elapsed: float) -> "OMPLSolverResult":
        return cls(None, time_elapsed, -1, TerminateState.FAIL_SATISFACTION)


OMPLSolverT = TypeVar("OMPLSolverT", bound="OMPLSolver")


@dataclass
class OMPLSolverBase(AbstractSolver[OMPLSolverConfig, OMPLSolverResult]):
    config: OMPLSolverConfig
    problem: Optional[Problem]
    planner: Optional[_OMPLPlannerBase]
    repair_planner: Optional[RepairPlanner]  # used when init trajectory is given
    _n_call_dict: Dict[str, int]

    @classmethod
    def get_result_type(cls) -> Type[OMPLSolverResult]:
        return OMPLSolverResult

    def setup(self, problem: Problem) -> None:
        self._n_call_dict["count"] = 0  # reset count

        def is_valid(q_: List[float]) -> bool:
            self._n_call_dict["count"] += 1
            q = np.array(q_)
            if problem.global_ineq_const is None:
                return True
            else:
                val, _ = problem.global_ineq_const.evaluate_single(q, with_jacobian=False)
                return bool(np.all(val > 0))

        lb = problem.box_const.lb
        ub = problem.box_const.ub

        planner = self.create_planner(
            eq_const=problem.global_eq_const,
            lb=lb,
            ub=ub,
            is_valid=is_valid,
            n_max_is_valid=self.config.n_max_call,
            validation_box=problem.motion_step_box,
            algo=self.config.algorithm,
            algo_range=self.config.algorithm_range,
        )

        self.problem = problem
        self.planner = planner

        if self.problem.global_eq_const is None:
            # NOTE: lightning repair planner can handle only
            # planning in euclidean space
            repair_planner = RepairPlanner(
                lb,
                ub,
                is_valid,
                self.config.n_max_call,
                problem.motion_step_box,
                algo=self.config.algorithm,
                algo_range=self.config.algorithm_range,
            )
            self.repair_planner = repair_planner
        else:
            self.repair_planner = None

    @abstractmethod
    def create_planner(self, **kwargs) -> _OMPLPlannerBase:
        ...

    def solve(self, init_traj: Optional[Trajectory] = None):
        assert self.problem is not None, "setup is not called yet"
        assert self.planner is not None

        ts = time.time()

        planner: _OMPLPlannerBase
        satisfy_result: Optional[SatisfactionResult] = None
        if init_traj is not None:
            message = "replanner could not be defind for this problem."
            assert self.repair_planner is not None, message

            # use the trajectory last element as the guess
            q_ik_guess = init_traj.numpy()[-1]

            satisfy_result = satisfy_by_optimization(
                self.problem.goal_const,
                self.problem.box_const,
                self.problem.global_ineq_const,
                q_seed=q_ik_guess,
            )

            planner = self.repair_planner
            planner.set_heuristic(init_traj.numpy())
        else:
            # solve IK without initial guess
            for _ in range(self.config.n_max_satisfaction_trial):
                satisfy_result = satisfy_by_optimization(
                    self.problem.goal_const,
                    self.problem.box_const,
                    self.problem.global_ineq_const,
                    None,
                )
                if satisfy_result.success:
                    break
            assert satisfy_result is not None
            if not satisfy_result.success:
                return OMPLSolverResult(
                    None, time.time() - ts, -1, TerminateState.FAIL_SATISFACTION
                )

            planner = self.planner

        q_start = self.problem.start
        q_goal = satisfy_result.q
        try:
            plan_result = planner.solve(q_start, q_goal, self.config.simplify)
        except InvalidProblemError:
            # TODO: this should not happen...
            return OMPLSolverResult(None, time.time() - ts, -1, TerminateState.FAIL_SATISFACTION)

        if plan_result is not None:
            terminate_state = TerminateState.SUCCESS
            traj = Trajectory(plan_result)
        else:
            terminate_state = TerminateState.FAIL_PLANNING
            traj = None

        valid_func_call_count = self._n_call_dict["count"]
        self._n_call_dict["count"] = 0
        self.problem = None
        return OMPLSolverResult(traj, time.time() - ts, valid_func_call_count, terminate_state)


class OMPLSolver(AbstractScratchSolver[OMPLSolverConfig, OMPLSolverResult], OMPLSolverBase):
    @classmethod
    def init(cls, config: OMPLSolverConfig) -> "OMPLSolver":
        n_call_dict = {"count": 0}
        return cls(config, None, None, None, n_call_dict)

    def create_planner(self, **kwargs) -> _OMPLPlannerBase:
        if kwargs["eq_const"] is None:
            kwargs.pop("eq_const")
            return Planner(**kwargs)
        else:
            f = kwargs["eq_const"]

            def eq_const(x: List[float]):
                np_x = np.array(x)
                return f.evaluate_single(np_x, True)

            kwargs["eq_const"] = eq_const

            return ConstrainedPlanner(**kwargs)


@dataclass
class LightningSolver(AbstractDataDrivenSolver[OMPLSolverConfig, OMPLSolverResult], OMPLSolverBase):
    db: LightningDB

    @classmethod
    def init(
        cls, config: OMPLSolverConfig, dataset: List[Tuple[Problem, Trajectory]]
    ) -> "LightningSolver":
        n_call_dict = {"count": 0}

        trajectories = [pair[1] for pair in dataset]

        dim = len(trajectories[0][0])
        db = LightningDB(dim)
        for traj in trajectories:
            db.add_experience(list(traj.numpy()))

        return cls(config, None, None, None, n_call_dict, db)

    def create_planner(self, **kwargs) -> _OMPLPlannerBase:
        if kwargs["eq_const"] is not None:
            raise RuntimeError("lightning does not support global equality constraint")
        kwargs.pop("eq_const")
        kwargs["db"] = self.db
        return LightningPlanner(**kwargs)
