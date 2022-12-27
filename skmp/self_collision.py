from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
from scipy.linalg import block_diag
from selcol.dataset import Header
from selcol.neural import SelcolNet


@dataclass
class SelfCollisionMap:
    f: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    dim: int
    threshold: float = 0.5

    @classmethod
    def from_deep_model(
        cls, model: SelcolNet, header: Header, control_joint_names: List[str]
    ) -> "SelfCollisionMap":
        indices = [header.joint_names.index(jn) for jn in control_joint_names]
        dim = len(indices)
        f, grad = model.as_jit_function(np.array(indices))
        return cls(f, grad, dim)

    def evaluate(self, qs: np.ndarray, with_grad: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        qs is a trajectory (can be 2dim or 1dim(flattend format).
        if with_grad = False, grad will be just an empty matrix:
        """
        if qs.ndim == 1:  # flattened format
            qs = qs.reshape(-1, self.dim)
        assert qs.ndim == 2

        vals = np.array([self.threshold - self.f(q) for q in qs])
        jacobian = np.empty(0)
        if with_grad:
            grads = [-self.grad(q) for q in qs]
            jacobian = block_diag(*grads)
        return vals, jacobian
