from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Type, TypeVar, Union

import GPy
import numpy as np
from sklearn.decomposition import PCA

RegressorT = TypeVar("RegressorT", bound="Regressor")


class Regressor(ABC):
    @classmethod
    @abstractmethod
    def fit(cls: Type[RegressorT], X: np.ndarray, Y: np.ndarray, **kwargs) -> RegressorT:
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
class GPRRegressor(Regressor):
    gp: Union[GPy.models.GPRegression, GPy.models.SparseGPRegression]
    pca: Optional[PCA] = None

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray, pca_dim: Optional[int] = None) -> "GPRRegressor":  # type: ignore[override]
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
        return cls(gp, pca)

    def predict(self, x: np.ndarray):
        y, cov = self.gp.predict(np.expand_dims(x, axis=0))
        if self.pca is not None:
            y = self.pca.inverse_transform(np.expand_dims(y, axis=0))[0]
        return
