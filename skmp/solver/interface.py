import multiprocessing
import os
import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generic, List, Optional, Protocol, Tuple, Type, TypeVar, Union

import numpy as np
import threadpoolctl

from skmp.constraint import (
    AbstractEqConst,
    AbstractIneqConst,
    BoxConst,
    IneqCompositeConst,
)
from skmp.solver.motion_step_box import is_valid_motion_step
from skmp.trajectory import Trajectory

GoalConstT = TypeVar("GoalConstT")
GlobalIneqConstT = TypeVar("GlobalIneqConstT")
GlobalEqConstT = TypeVar("GlobalEqConstT")
SolverT = TypeVar("SolverT", bound="AbstractSolver")
DataDrivenSolverT = TypeVar("DataDrivenSolverT", bound="AbstractDataDrivenSolver")
ConfigT = TypeVar("ConfigT", bound="ConfigProtocol")
ResultT = TypeVar("ResultT", bound="ResultProtocol")
GuidingTrajT = TypeVar("GuidingTrajT", bound=Any)


@dataclass
class Problem:
    start: np.ndarray
    box_const: BoxConst
    goal_const: AbstractEqConst
    global_ineq_const: Optional[AbstractIneqConst]
    global_eq_const: Optional[AbstractEqConst]
    eqconst_admissible_mse: float = 1e-6
    motion_step_box_: Union[float, np.ndarray] = 0.1

    def check_init_feasibility(self) -> Tuple[bool, str]:
        msg_list = []
        if not np.all(self.start < self.box_const.ub):
            msg_list.append("q_start doesn't satisfy BoxConst upper")
        if not np.all(self.start > self.box_const.lb):
            msg_list.append("q_start doesn't satisfy BoxConst lower")
        if self.global_ineq_const is not None:
            if not self.global_ineq_const.is_valid(self.start):
                if isinstance(self.global_ineq_const, IneqCompositeConst):
                    for c in self.global_ineq_const.const_list:
                        if not c.is_valid(self.start):
                            msg_list.append("q_start doesn't satisfy {}".format(c))
                else:
                    msg_list.append("q_start doesn't satisfy {}".format(self.global_ineq_const))
        is_init_feasible = len(msg_list) == 0
        msg_concat = ", ".join(msg_list)
        return is_init_feasible, msg_concat

    @property
    def motion_step_box(self) -> np.ndarray:
        if isinstance(self.motion_step_box_, np.ndarray):
            return self.motion_step_box_

        n_dim = len(self.start)
        motion_step_box = np.ones(n_dim) * self.motion_step_box_
        return motion_step_box

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
        if self.global_ineq_const is not None:
            # note: we will not check eqality constraint because checking requires
            # traversing on manifold and its bit difficult to implement
            for i in range(len(traj) - 1):
                q1, q2 = traj[i], traj[i + 1]
                if not is_valid_motion_step(self.motion_step_box, q1, q2, self.global_ineq_const):  # type: ignore[arg-type]
                    return False
        return True


class ConfigProtocol(Protocol):
    n_max_call: int
    timeout: Optional[int]  # second


class ResultProtocol(Protocol):
    traj: Optional[Trajectory]
    time_elapsed: Optional[float]
    n_call: int

    @classmethod
    def abnormal(cls: Type[ResultT]) -> ResultT:
        """create result when solver failed without calling the core-solver
        and could not get n_call and other stuff"""
        ...


class AbstractSolver(ABC, Generic[ConfigT, ResultT, GuidingTrajT]):
    config: ConfigT
    problem: Optional[Problem]

    @abstractmethod
    def get_result_type(self) -> Type[ResultT]:
        ...

    def setup(self, problem: Problem) -> None:
        """setup solver for a paticular problem"""
        self._setup(problem)
        self.problem = problem

    @abstractmethod
    def _setup(self, problem: Problem):
        ...

    def solve(self, guiding_traj: Optional[GuidingTrajT] = None) -> ResultT:
        """solve problem with maybe a solution guess"""
        ts = time.time()

        class TimeoutException(Exception):
            ...

        if self.config.timeout is not None:
            assert self.config.timeout > 0

            def handler(sig, frame):
                raise TimeoutException()

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(self.config.timeout)

        try:
            assert self.problem is not None
            is_init_feasible, _ = self.problem.check_init_feasibility()
            if is_init_feasible:
                ret = self._solve(guiding_traj)
            else:
                ret = self.get_result_type().abnormal()
        except TimeoutException:
            ret = self.get_result_type().abnormal()

        if self.config.timeout is not None:
            signal.alarm(0)  # reset alarm

        ret.time_elapsed = time.time() - ts
        return ret

    @abstractmethod
    def _solve(self, guiding_traj: Optional[GuidingTrajT] = None) -> ResultT:
        ...

    def as_parallel_solver(self, n_process=4) -> "ParallelSolver[ConfigT, ResultT, GuidingTrajT]":
        return ParallelSolver(self.config, self, n_process)


@dataclass
class ParallelSolver(AbstractSolver, Generic[ConfigT, ResultT, GuidingTrajT]):
    config: ConfigT
    internal_solver: AbstractSolver[ConfigT, ResultT, GuidingTrajT]
    n_process: int = 4

    def get_result_type(self) -> Type[ResultT]:
        return self.internal_solver.get_result_type()

    def _setup(self, problem: Problem) -> None:
        self.internal_solver.setup(problem)

    def _parallel_solve_inner(self, replan_info: Optional[GuidingTrajT] = None) -> ResultT:
        """assume to be used in multi processing"""
        # prevend numpy from using multi-thread
        unique_seed = datetime.now().microsecond + os.getpid()
        np.random.seed(unique_seed)
        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            return self.internal_solver._solve(replan_info)

    def _solve(self, replan_info: Optional[GuidingTrajT] = None) -> ResultT:
        processes = []
        result_queue: multiprocessing.Queue[ResultT] = multiprocessing.Queue()

        for i in range(self.n_process):
            p = multiprocessing.Process(
                target=lambda: result_queue.put(self._parallel_solve_inner(replan_info))
            )
            processes.append(p)
            p.start()

        for _ in range(self.n_process):
            result = result_queue.get()
            if result.traj is not None:
                for p in processes:
                    p.terminate()
                for p in processes:
                    p.join()
                return result
        return result.abnormal()


class AbstractScratchSolver(AbstractSolver[ConfigT, ResultT, Trajectory]):
    @classmethod
    @abstractmethod
    def init(cls: Type[SolverT], config: ConfigT) -> SolverT:
        """common interface of constructor"""
        ...


class AbstractDataDrivenSolver(AbstractSolver[ConfigT, ResultT, np.ndarray]):
    @classmethod
    @abstractmethod
    def init(
        cls: Type[SolverT], config: ConfigT, dataset: List[Tuple[np.ndarray, Trajectory]]
    ) -> SolverT:
        """common interface of constructor"""
        ...
