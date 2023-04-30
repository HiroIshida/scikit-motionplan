import multiprocessing
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, List, Optional, Protocol, Tuple, Type, TypeVar, Union

import numpy as np
import threadpoolctl

from skmp.constraint import AbstractEqConst, AbstractIneqConst, BoxConst
from skmp.solver.motion_step_box import is_valid_motion_step
from skmp.trajectory import Trajectory

GoalConstT = TypeVar("GoalConstT")
GlobalIneqConstT = TypeVar("GlobalIneqConstT")
GlobalEqConstT = TypeVar("GlobalEqConstT")
SolverT = TypeVar("SolverT", bound="AbstractSolver")
DataDrivenSolverT = TypeVar("DataDrivenSolverT", bound="AbstractDataDrivenSolver")
ConfigT = TypeVar("ConfigT", bound="ConfigProtocol")
ResultT = TypeVar("ResultT", bound="ResultProtocol")
DataLikeT = TypeVar("DataLikeT")


@dataclass
class Problem:
    start: np.ndarray
    box_const: BoxConst
    goal_const_seq: List[AbstractEqConst]
    global_ineq_const: Optional[AbstractIneqConst]
    global_eq_const: Optional[AbstractEqConst]
    eqconst_admissible_mse: float
    motion_step_box_: np.ndarray

    def __init__(
        self,
        start: np.ndarray,
        box_const: BoxConst,
        goal_const_seq: Union[AbstractEqConst, List[AbstractEqConst]],
        global_ineq_const: Optional[AbstractIneqConst],
        global_eq_const: Optional[AbstractEqConst],
        eqconst_admissible_mse: float = 1e-6,
        motion_step_box: Union[float, np.ndarray] = 0.1,
    ):

        if isinstance(goal_const_seq, AbstractEqConst):
            goal_const_seq = [goal_const_seq]

        if not isinstance(self.motion_step_box_, np.ndarray):
            n_dim = len(start)
            np.ones(n_dim) * self.motion_step_box_

        self.start = start
        self.box_const = box_const
        self.goal_const_seq = goal_const_seq
        self.global_ineq_const = global_ineq_const
        self.global_eq_const = global_eq_const
        self.eqconst_admissible_mse = eqconst_admissible_mse

    @property
    def motion_step_box(self) -> np.ndarray:
        if isinstance(self.motion_step_box_, np.ndarray):
            return self.motion_step_box_

        n_dim = len(self.start)
        motion_step_box = np.ones(n_dim) * self.motion_step_box_
        return motion_step_box

    def is_constrained(self) -> bool:
        return self.global_eq_const is not None

    def is_satisfied(self, traj_seq: Union[Trajectory, List[Trajectory]]) -> bool:
        if isinstance(traj_seq, Trajectory):
            traj_seq = [traj_seq]

        for goal_const, traj in zip(self.goal_const_seq, traj_seq):
            vals, _ = goal_const.evaluate_single(traj[-1], with_jacobian=False)
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


class ResultProtocol(Protocol):
    traj: Optional[Trajectory]
    time_elapsed: float
    n_call: int

    @classmethod
    def abnormal(cls: Type[ResultT], time_elapsed: float) -> ResultT:
        """create result when solver failed without calling the core-solver
        and could not get n_call and other stuff"""
        ...


class AbstractSolver(ABC, Generic[ConfigT, ResultT]):
    @abstractmethod
    def get_result_type(self) -> Type[ResultT]:
        ...

    @abstractmethod
    def setup(self, problem: Problem) -> None:
        """setup solver for a paticular problem"""
        ...

    @abstractmethod
    def solve(self, init_traj: Optional[Trajectory] = None) -> ResultT:
        """solve problem with maybe a solution guess"""
        ...

    def as_parallel_solver(self, n_process=4) -> "ParallelSolver[ConfigT, ResultT]":
        return ParallelSolver(self, n_process)


@dataclass
class ParallelSolver(AbstractSolver, Generic[ConfigT, ResultT]):
    internal_solver: AbstractSolver[ConfigT, ResultT]
    n_process: int = 4

    def get_result_type(self) -> Type[ResultT]:
        return self.internal_solver.get_result_type()

    def setup(self, problem: Problem) -> None:
        self.internal_solver.setup(problem)

    def _parallel_solve_inner(self, init_traj: Optional[Trajectory] = None) -> ResultT:
        """assume to be used in multi processing"""
        # prevend numpy from using multi-thread
        unique_seed = datetime.now().microsecond + os.getpid()
        np.random.seed(unique_seed)
        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            return self.internal_solver.solve(init_traj)

    def solve(self, init_traj: Optional[Trajectory] = None) -> ResultT:
        processes = []
        result_queue = multiprocessing.Queue()  # type: ignore

        for i in range(self.n_process):
            p = multiprocessing.Process(
                target=lambda: result_queue.put(self._parallel_solve_inner(init_traj))
            )
            processes.append(p)
            p.start()
            print("start process")

        result = result_queue.get()
        for p in processes:
            p.terminate()
        return result


class AbstractScratchSolver(AbstractSolver[ConfigT, ResultT]):
    @classmethod
    @abstractmethod
    def init(cls: Type[SolverT], config: ConfigT) -> SolverT:
        """common interface of constructor"""
        ...


class AbstractDataDrivenSolver(AbstractSolver[ConfigT, ResultT]):
    @classmethod
    @abstractmethod
    def init(
        cls: Type[SolverT], config: ConfigT, dataset: List[Tuple[Problem, Trajectory]]
    ) -> SolverT:
        """common interface of constructor"""
        ...
