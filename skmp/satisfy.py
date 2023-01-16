import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, minimize

from skmp.constraint import (
    AbstractEqConst,
    AbstractIneqConst,
    BoxConst,
    ConfigPointConst,
)


def scipinize(fun: Callable) -> Tuple[Callable, Callable]:
    closure_member = {"jac_cache": None}

    def fun_scipinized(x):
        f, jac = fun(x)
        closure_member["jac_cache"] = jac
        return f

    def fun_scipinized_jac(x):
        return closure_member["jac_cache"]

    return fun_scipinized, fun_scipinized_jac


@dataclass
class SatisfactionConfig:
    ftol: float = 1e-6
    disp: bool = False
    n_max_eval: int = 200
    acceptable_error: float = 1e-6


@dataclass
class SatisfactionResult:
    q: np.ndarray
    elapsed_time: float
    success: bool


def satisfy_by_optimization(
    eq_const: AbstractEqConst,
    box_const: BoxConst,
    ineq_const: Optional[AbstractIneqConst],
    q_seed: Optional[np.ndarray],
    config: Optional[SatisfactionConfig] = None,
) -> SatisfactionResult:
    ts = time.time()
    if config is None:
        config = SatisfactionConfig()

    if isinstance(eq_const, ConfigPointConst):
        return SatisfactionResult(eq_const.desired_angles, 0.0, True)

    def objective_fun(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vals, jac = eq_const.evaluate_single(q, with_jacobian=True)
        f = vals.dot(vals)
        grad = 2 * vals.dot(jac)
        return f, grad

    f, jac = scipinize(objective_fun)

    constraints = []
    if ineq_const is not None:

        def fun_ineq(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            val, jac = ineq_const.evaluate_single(q, with_jacobian=True)  # type: ignore[union-attr]
            margin_numerical = 1e-6
            return val - margin_numerical, jac

        ineq_const_scipy, ineq_const_jac_scipy = scipinize(fun_ineq)
        ineq_dict = {"type": "ineq", "fun": ineq_const_scipy, "jac": ineq_const_jac_scipy}
        constraints.append(ineq_dict)

    bounds = Bounds(box_const.lb, box_const.ub, keep_feasible=True)  # type: ignore

    if q_seed is None:
        q_seed = box_const.sample()

    slsqp_option: Dict = {
        "ftol": config.ftol,
        "disp": config.disp,
        "maxiter": config.n_max_eval - 1,  # somehome scipy iterate +1 more time
    }

    res = minimize(
        f,
        q_seed,
        method="SLSQP",
        jac=jac,
        bounds=bounds,
        constraints=constraints,
        options=slsqp_option,
    )

    # check additional success condition
    is_ik_actually_solved = res.fun < config.acceptable_error  # ensure not in local optima
    if not is_ik_actually_solved:
        res.success = False

    elapsed_time = time.time() - ts

    return SatisfactionResult(res.x, elapsed_time, res.success)


def satisfy_by_optimization_with_budget(
    eq_const: AbstractEqConst,
    box_const: BoxConst,
    ineq_const: Optional[AbstractIneqConst],
    q_seed: Optional[np.ndarray],
    config: Optional[SatisfactionConfig] = None,
    n_trial_budget: int = 20,
) -> SatisfactionResult:
    # a util function
    for i in range(n_trial_budget):
        res = satisfy_by_optimization(eq_const, box_const, ineq_const, q_seed, config=config)
        if res.success:
            return res
        q_seed = (
            None  # if specified seed was not effective, we invalidate it and will use random seed
        )
    assert False, "satisfaction fail"
