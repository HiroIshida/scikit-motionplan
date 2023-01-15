from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Type, TypeVar, Union

import GPy
import numpy as np
from sklearn.decomposition import PCA

from skmp.constraint import VectorDescriptable
from skmp.solver.interface import AbstractDataDrivenSolver, Problem
from skmp.solver.nlp_solver import (
    SQPBasedSolver,
    SQPBasedSolverConfig,
    SQPBasedSolverResult,
)
from skmp.trajectory import Trajectory

RegressorT = TypeVar("RegressorT", bound="Regressor")
MemmoSolverT = TypeVar("MemmoSolverT", bound="AbstractMemmoSolver")


class Regressor(ABC):
    @classmethod
    def fit_from_trajectories(cls: Type[RegressorT], traj_list: List[Trajectory]) -> RegressorT:
        x_list = []
        y_list = []
        for traj in traj_list:
            start, goal = traj[0], traj[-1]
            x = np.hstack([start, goal])
            y = traj.numpy()
            x_list.append(x)
            y_list.append(y)
        X = np.array(x_list)
        Y = np.array(y_list)
        return cls.fit(X, Y)

    @classmethod
    @abstractmethod
    def fit(cls: Type[RegressorT], X: np.ndarray, Y: np.ndarray) -> RegressorT:
        ...

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class NNRegressor(Regressor):
    X: np.ndarray
    Y: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray) -> "NNRegressor":  # type: ignore[override]
        return cls(X, Y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        dists = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
        idx = np.argmin(dists)
        return self.Y[idx]


@dataclass(frozen=True)
class StraightRegressor(Regressor):
    n_wp: int

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray) -> "StraightRegressor":  # type: ignore[override]
        _, n_wp, _ = Y.shape
        return cls(n_wp)

    def predict(self, x: np.ndarray) -> np.ndarray:
        # assume that x is composed of q_init and q_goal
        q_init, q_goal = x.reshape(2, -1)
        width = (q_goal - q_init) / (self.n_wp - 1)
        traj = np.array([q_init + i * width for i in range(self.n_wp)])
        return traj


@dataclass(frozen=True)
class GPRRegressorBase(Regressor):
    gp: Union[GPy.models.GPRegression, GPy.models.SparseGPRegression]
    pca: Optional[PCA] = None

    @classmethod
    def _fit(cls, X: np.ndarray, Y: np.ndarray, pca_dim: Optional[int] = None):
        n_data, n_input_dim = X.shape
        Y_flatten = Y.reshape(n_data, -1)

        use_pca = pca_dim is not None
        if use_pca:
            pca = PCA(pca_dim)
            pca.fit(Y_flatten)
            Y_flatten = pca.transform(Y_flatten)
        else:
            pca = None

        if n_data < 200:
            kernel = GPy.kern.RBF(
                input_dim=n_input_dim, variance=0.1, lengthscale=0.3, ARD=True
            ) + GPy.kern.White(input_dim=n_input_dim)
            gp = GPy.models.GPRegression(X, Y_flatten, kernel)
            num_restarts = 10
            gp.optimize_restarts(num_restarts=num_restarts)
        else:
            Z = X[:100]
            gp = GPy.models.SparseGPRegression(X, Y_flatten, Z=Z)
            gp.optimize("bfgs")
        return gp, pca

    def predict(self, x: np.ndarray):
        y, cov = self.gp.predict(np.expand_dims(x, axis=0))
        if self.pca is not None:
            y = self.pca.inverse_transform(np.expand_dims(y, axis=0))[0]
        return


@dataclass(frozen=True)
class GPRRegressor(GPRRegressorBase):
    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray) -> "GPRRegressor":
        return cls(*cls._fit(X, Y, None))


@dataclass(frozen=True)
class PCAGPRRegressor(GPRRegressorBase):
    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray) -> "PCAGPRRegressor":
        dim_pca = 50  # same as the paper
        return cls(*cls._fit(X, Y, dim_pca))


@dataclass
class AbstractMemmoSolver(AbstractDataDrivenSolver[SQPBasedSolverConfig, SQPBasedSolverResult]):
    solver: SQPBasedSolver
    regressor: Regressor
    problem_vector_description: Optional[np.ndarray]

    @classmethod
    @abstractmethod
    def get_regressor_type(cls) -> Type[Regressor]:
        ...

    @classmethod
    def init(
        cls: Type[MemmoSolverT], config: SQPBasedSolverConfig, trajectories: List[Trajectory]
    ) -> MemmoSolverT:
        solver = SQPBasedSolver.init(config)
        regressor_type = cls.get_regressor_type()
        regressor = regressor_type.fit_from_trajectories(trajectories)
        return cls(solver, regressor, None)

    @classmethod
    def get_result_type(cls) -> Type[SQPBasedSolverResult]:
        return SQPBasedSolverResult

    def setup(self, problem: Problem) -> None:
        self.solver.setup(problem)
        goal_const = problem.goal_const
        assert isinstance(goal_const, VectorDescriptable)
        goal_decription = goal_const.get_description()

        # NOTE: memmo encode only start and goal constraint
        problem_vector_description = np.hstack((problem.start, goal_decription))
        self.problem_vector_description = problem_vector_description

    def solve(self, init_traj: Optional[Trajectory] = None) -> SQPBasedSolverResult:
        if not init_traj:
            pass
        return self.solver.solve(init_traj)


class NnMemmoSolver(AbstractMemmoSolver):
    @classmethod
    def get_regressor_type(cls) -> Type[NNRegressor]:
        return NNRegressor


class StraightLineMemmoSolver(AbstractMemmoSolver):
    @classmethod
    def get_regressor_type(cls) -> Type[StraightRegressor]:
        return StraightRegressor


class GprMemmoSolver(AbstractMemmoSolver):
    @classmethod
    def get_regressor_type(cls) -> Type[GPRRegressor]:
        return GPRRegressor


class PcaGprMemmoSolver(AbstractMemmoSolver):
    @classmethod
    def get_regressor_type(cls) -> Type[PCAGPRRegressor]:
        return PCAGPRRegressor
