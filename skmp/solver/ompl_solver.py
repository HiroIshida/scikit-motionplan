from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from ompl import (
    Algorithm,
    ConstrainedPlanner,
    ConstStateType,
    ERTConnectPlanner,
    InvalidProblemError,
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
    expbased_planner_backend: Literal["ertconnect", "lightning"] = "lightning"
    ertconnect_eps: float = 5.0  # used only when ertconnect is selected
    const_state_type: ConstStateType = ConstStateType.PROJECTION
    timeout: Optional[int] = None


class TerminateState(Enum):
    SUCCESS = 1
    FAIL_SATISFACTION = 2
    FAIL_PLANNING = 3
    FAIL_SIMPLIFYING = 4


@dataclass
class OMPLSolverResult:
    traj: Optional[Trajectory]
    time_elapsed: Optional[float]
    n_call: int
    terminate_state: TerminateState

    @classmethod
    def abnormal(cls) -> "OMPLSolverResult":
        return cls(None, None, -1, TerminateState.FAIL_SATISFACTION)


OMPLSolverT = TypeVar("OMPLSolverT", bound="OMPLSolver")


@dataclass
class OMPLSolverBase(AbstractSolver[OMPLSolverConfig, OMPLSolverResult, Trajectory]):
    config: OMPLSolverConfig
    problem: Optional[Problem]
    planner: Optional[_OMPLPlannerBase]
    expbased_planner: Optional[
        Union[ERTConnectPlanner, RepairPlanner]
    ]  # used when init trajectory is given
    _n_call_dict: Dict[str, int]

    @classmethod
    def get_result_type(cls) -> Type[OMPLSolverResult]:
        return OMPLSolverResult

    def _setup(self, problem: Problem) -> None:
        self._n_call_dict["count"] = 0  # reset count

        def is_valid(q_: List[float]) -> bool:
            self._n_call_dict["count"] += 1
            q = np.array(q_)
            if problem.global_ineq_const is None:
                return True
            else:
                return problem.global_ineq_const.is_valid(q)

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
            cs_type=self.config.const_state_type,
        )

        self.planner = planner

        if problem.global_eq_const is None:
            # NOTE: lightning repair planner can handle only
            # planning in euclidean space
            if self.config.expbased_planner_backend == "ertconnect":
                expbased_planner_t = ERTConnectPlanner
            elif self.config.expbased_planner_backend == "lightning":
                expbased_planner_t = RepairPlanner  # type: ignore
            else:
                assert False
            expbased_planner = expbased_planner_t(
                lb,
                ub,
                is_valid,
                self.config.n_max_call,
                problem.motion_step_box,
                algo=self.config.algorithm,
                algo_range=self.config.algorithm_range,
            )

            if self.config.expbased_planner_backend == "ertconnect":
                expbased_planner.set_parameters(eps=self.config.ertconnect_eps)

            self.expbased_planner = expbased_planner
        else:
            self.expbased_planner = None

    @abstractmethod
    def create_planner(self, **kwargs) -> _OMPLPlannerBase:
        ...

    def _solve(self, init_traj: Optional[Trajectory] = None) -> OMPLSolverResult:
        assert self.problem is not None, "setup is not called yet"
        assert self.planner is not None

        planner: _OMPLPlannerBase
        satisfy_result: Optional[SatisfactionResult] = None

        if init_traj is not None:
            # use the trajectory last element as the guess
            q_ik_guess = init_traj.numpy()[-1]
            satisfy_result = satisfy_by_optimization(
                self.problem.goal_const,
                self.problem.box_const,
                self.problem.global_ineq_const,
                q_seed=q_ik_guess,
            )
        else:
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
            return OMPLSolverResult(None, None, -1, TerminateState.FAIL_SATISFACTION)

        if init_traj is not None:
            message = "replanner could not be defind for this problem."
            assert self.expbased_planner is not None, message
            planner = self.expbased_planner
            planner.set_heuristic(init_traj.numpy())
        else:
            planner = self.planner

        q_start = self.problem.start
        q_goal = satisfy_result.q
        try:
            plan_result = planner.solve(q_start, q_goal, self.config.simplify)
        except InvalidProblemError:
            # TODO: this should not happen...
            return OMPLSolverResult(None, None, -1, TerminateState.FAIL_SATISFACTION)

        traj: Optional[Trajectory] = None
        if plan_result is not None:
            traj = Trajectory(plan_result)
            if not self.config.simplify or traj.is_simplified():
                terminate_state = TerminateState.SUCCESS
            else:
                traj = None
                terminate_state = TerminateState.FAIL_SIMPLIFYING
        else:
            terminate_state = TerminateState.FAIL_PLANNING
            traj = None

        valid_func_call_count = self._n_call_dict["count"]
        self._n_call_dict["count"] = 0
        self.problem = None
        return OMPLSolverResult(traj, None, valid_func_call_count, terminate_state)


class OMPLSolver(AbstractScratchSolver[OMPLSolverConfig, OMPLSolverResult], OMPLSolverBase):
    @classmethod
    def init(cls, config: OMPLSolverConfig) -> "OMPLSolver":
        n_call_dict = {"count": 0}
        return cls(config, None, None, None, n_call_dict)

    def create_planner(self, **kwargs) -> _OMPLPlannerBase:
        is_unconstraind = kwargs["eq_const"] is None
        if is_unconstraind:
            kwargs.pop("eq_const")
            kwargs.pop("cs_type")
            return Planner(**kwargs)
        else:
            f = kwargs["eq_const"]

            def eq_const(x: List[float]):
                np_x = np.array(x)
                return f.evaluate_single(np_x, True)

            kwargs["eq_const"] = eq_const

            return ConstrainedPlanner(**kwargs)


@dataclass
class OMPLDataDrivenSolver(AbstractDataDrivenSolver[OMPLSolverConfig, OMPLSolverResult]):
    config: OMPLSolverConfig
    internal_solver: OMPLSolverBase
    vec_descs: np.ndarray
    trajectories: List[Trajectory]

    @classmethod
    def init(
        cls, config: OMPLSolverConfig, dataset: List[Tuple[np.ndarray, Trajectory]]
    ) -> "OMPLDataDrivenSolver":
        n_call_dict = {"count": 0}
        vec_descs = np.array([p[0] for p in dataset])
        trajectories = [p[1] for p in dataset]
        internal_solver = OMPLSolver(config, None, None, None, n_call_dict)
        return cls(config, internal_solver, vec_descs, trajectories)

    def _solve(self, query_desc: Optional[np.ndarray] = None) -> OMPLSolverResult:
        if query_desc is not None:
            sqdists = np.sum((self.vec_descs - query_desc) ** 2, axis=1)
            idx_closest = np.argmin(sqdists)
            reuse_traj = self.trajectories[idx_closest]
        else:
            reuse_traj = None
        result = self.internal_solver._solve(reuse_traj)
        return result

    @classmethod
    def get_result_type(cls) -> Type[OMPLSolverResult]:
        return OMPLSolverResult

    def _setup(self, problem: Problem) -> None:
        self.internal_solver.setup(problem)
