from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, TypeVar

import numpy as np

from skmp.solver.interface import (
    AbstractScratchSolver,
    AbstractSolver,
    ConfigProtocol,
    Problem,
    ResultProtocol,
)
from skmp.trajectory import Trajectory

ConfigT = TypeVar("ConfigT", bound="ConfigProtocol")
ResultT = TypeVar("ResultT", bound="ResultProtocol")


@dataclass
class NearestNeigborSolver(AbstractSolver[ConfigT, ResultT, np.ndarray]):
    config: ConfigT
    internal_solver: AbstractScratchSolver[ConfigT, ResultT]
    vec_descs: np.ndarray
    trajectories: List[Optional[Trajectory]]  # None means no trajectory is available
    knn: int
    infeasibility_threshold: int
    previous_est_positive: Optional[bool] = None
    previous_false_positive: Optional[bool] = None

    @classmethod
    def init(
        cls: Type["NearestNeigborSolver[ConfigT, ResultT]"],
        solver_type: Type[AbstractScratchSolver[ConfigT, ResultT]],
        config: ConfigT,  # for internal solver
        dataset: List[Tuple[np.ndarray, Optional[Trajectory]]],
        knn: int = 1,
        infeasibility_threshold: Optional[int] = None,
    ) -> "NearestNeigborSolver[ConfigT, ResultT]":
        assert knn > 0

        tmp, trajectories = zip(*dataset)
        vec_descs = np.array(tmp)
        internal_solver = solver_type.init(config)

        if infeasibility_threshold is None:
            # do leave-one-out cross validation
            # see the V.A of the following paper for the detail
            # Hauser, Kris. "Learning the problem-optimum map: Analysis and application to global optimization in robotics." IEEE Transactions on Robotics 33.1 (2016): 141-152.
            # actually, the threshold can be tuned wrt specified fp-rate, but what we vary is only integer
            # we have little control over the fp-rate. So just use the best threshold in terms of the accuracy
            errors = []
            thresholds = list(range(1, knn + 1))
            for threshold in thresholds:
                error = 0
                for i, (desc, traj) in enumerate(dataset):
                    sqdists = np.sum((vec_descs - desc) ** 2, axis=1)
                    k_nearests = np.argsort(sqdists)[1 : knn + 1]  # +1 because itself is included
                    none_count_in_knn = sum(1 for idx in k_nearests if trajectories[idx] is None)
                    seems_infeasible = none_count_in_knn >= threshold
                    actually_infeasible = traj is None
                    if seems_infeasible != actually_infeasible:
                        error += 1
                errors.append(error)
            print(f"t-error pairs: {list(zip(thresholds, errors))}")
            infeasibility_threshold = thresholds[np.argmin(errors)]
        return cls(
            config, internal_solver, vec_descs, list(trajectories), knn, infeasibility_threshold
        )

    def _knn_trajectories(self, query_desc: np.ndarray) -> List[Optional[Trajectory]]:
        sqdists = np.sum((self.vec_descs - query_desc) ** 2, axis=1)
        k_nearests = np.argsort(sqdists)[: self.knn]
        return [self.trajectories[i] for i in k_nearests]

    def _solve(self, query_desc: Optional[np.ndarray] = None) -> ResultT:
        if query_desc is not None:
            trajs = self._knn_trajectories(query_desc)
            trajs_without_none = [traj for traj in trajs if traj is not None]
            count_none = len(trajs) - len(trajs_without_none)
            seems_infeasible = count_none >= self.infeasibility_threshold
            self.previous_est_positive = not seems_infeasible
            if seems_infeasible:
                return self.get_result_type().abnormal()

            for guiding_traj in trajs_without_none:
                if guiding_traj is not None:
                    result = self.internal_solver._solve(guiding_traj)
                    if result.traj is not None:
                        self.previous_false_positive = False
                        return result
            self.previous_false_positive = True
            return self.get_result_type().abnormal()
        else:
            reuse_traj = None
            result = self.internal_solver._solve(reuse_traj)
            return result

    def get_result_type(self) -> Type[ResultT]:
        return self.internal_solver.get_result_type()

    def _setup(self, problem: Problem) -> None:
        self.internal_solver.setup(problem)
        self.previous_est_positive = None
        self.previous_false_positive = None
