from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, TypeVar, Union

import GPy
import numpy as np
from sklearn.decomposition import PCA

from skmp.solver.interface import AbstractDataDrivenSolver, Problem
from skmp.solver.nlp_solver import (
    SQPBasedSolver,
    SQPBasedSolverConfig,
    SQPBasedSolverResult,
)
from skmp.trajectory import Trajectory

RegressorT = TypeVar("RegressorT", bound="Regressor")
MemmoSolverT = TypeVar("MemmoSolverT", bound="AbstractMemmoSolver")


@dataclass(frozen=True)
class Regressor(ABC):
    n_dim: int

    @classmethod
    def fit_from_dataset(
        cls: Type[RegressorT], dataset: List[Tuple[np.ndarray, Trajectory]]
    ) -> RegressorT:
        x_list = []
        y_list = []
        for vec_desc, trajectory in dataset:
            x_list.append(vec_desc)
            y = trajectory.numpy()
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
        n_data, n_wp, n_dim = Y.shape
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
        n_data, n_wp, n_dim = Y.shape
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
        n_data, n_wp, n_dim = Y.shape
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
    config: SQPBasedSolverConfig
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
        dataset: List[Tuple[np.ndarray, Trajectory]],
    ) -> MemmoSolverT:
        solver = SQPBasedSolver.init(config)
        regressor_type = cls.get_regressor_type()
        regressor = regressor_type.fit_from_dataset(dataset)
        return cls(config, solver, regressor, None)

    @classmethod
    def get_result_type(cls) -> Type[SQPBasedSolverResult]:
        return SQPBasedSolverResult

    def _setup(self, problem: Problem) -> None:
        pass

    def _solve(self, replan_info: Optional[np.ndarray] = None) -> SQPBasedSolverResult:
        init_traj = None
        if replan_info is not None:
            init_traj = self.regressor.predict(replan_info)
        result = self.solver._solve(init_traj)
        return result


class NnMemmoSolver(AbstractMemmoSolver):
    @classmethod
    def get_regressor_type(cls) -> Type[NNRegressor]:
        return NNRegressor


class GprMemmoSolver(AbstractMemmoSolver):
    @classmethod
    def get_regressor_type(cls) -> Type[GPRRegressor]:
        return GPRRegressor


class PcaGprMemmoSolver(AbstractMemmoSolver):
    @classmethod
    def get_regressor_type(cls) -> Type[PCAGPRRegressor]:
        return PCAGPRRegressor
