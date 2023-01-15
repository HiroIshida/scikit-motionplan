from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, TypeVar, Union

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


def get_memmo_problem_description(problem: Problem) -> np.ndarray:
    """get vector-form description of a problem"""
    start_desc = problem.start
    assert isinstance(problem.goal_const, VectorDescriptable)
    goal_desc = problem.goal_const.get_description()
    return np.hstack([start_desc, goal_desc])


@dataclass(frozen=True)
class Regressor(ABC):
    n_dim: int

    @classmethod
    def fit_from_dataset(
        cls: Type[RegressorT], dataset: List[Tuple[Problem, Trajectory]]
    ) -> RegressorT:
        x_list = []
        y_list = []
        for problem, trajectory in dataset:
            x = get_memmo_problem_description(problem)
            y = trajectory.numpy()
            x_list.append(x)
            y_list.append(y)
        X = np.array(x_list)
        Y = np.array(y_list)
        return cls.fit(X, Y)

    @classmethod
    @abstractmethod
    def fit(cls: Type[RegressorT], X: np.ndarray, Y: np.ndarray) -> RegressorT:
        ...

    def predict(self, x: np.ndarray) -> Trajectory:
        y = self._predict(x)
        points = list(y.reshape((-1, self.n_dim)))
        return Trajectory(points)

    @abstractmethod
    def _predict(self, x: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class NNRegressor(Regressor):
    X: np.ndarray
    Y: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray) -> "NNRegressor":  # type: ignore[override]
        n_dim, _, _ = Y.shape
        return cls(n_dim, X, Y)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        dists = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
        idx = np.argmin(dists)
        return self.Y[idx]


@dataclass(frozen=True)
class StraightRegressor(Regressor):
    n_wp: int

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray) -> "StraightRegressor":  # type: ignore[override]
        n_dim, n_wp, _ = Y.shape
        return cls(n_dim, n_wp)

    def _predict(self, x: np.ndarray) -> np.ndarray:
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
        n_dim, _, _ = Y.shape
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
        return n_dim, gp, pca

    def _predict(self, x: np.ndarray):
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
        cls: Type[MemmoSolverT],
        config: SQPBasedSolverConfig,
        dataset: List[Tuple[Problem, Trajectory]],
    ) -> MemmoSolverT:
        solver = SQPBasedSolver.init(config)
        regressor_type = cls.get_regressor_type()
        regressor = regressor_type.fit_from_dataset(dataset)
        return cls(solver, regressor, None)

    @classmethod
    def get_result_type(cls) -> Type[SQPBasedSolverResult]:
        return SQPBasedSolverResult

    def setup(self, problem: Problem) -> None:
        self.solver.setup(problem)
        self.problem_vector_description = get_memmo_problem_description(problem)

    def solve(self, init_traj: Optional[Trajectory] = None) -> SQPBasedSolverResult:
        if not init_traj:
            assert self.problem_vector_description is not None
            init_traj = self.regressor.predict(self.problem_vector_description)
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
