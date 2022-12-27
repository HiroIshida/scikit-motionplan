from copy import deepcopy

import numpy as np
from test_solver_interface import standard_problem  # noqa

from skmp.satisfy import satisfy_by_optimization
from skmp.solver.interface import Problem
from skmp.solver.nlp_solver import SQPBasedSolver, SQPBasedSolverConfig, translate
from skmp.solver.trajectory_constraint import TrajectoryConstraint
from skmp.trajectory import Trajectory


def jac_numerical(const: TrajectoryConstraint, qs0: np.ndarray) -> np.ndarray:
    f0, _ = const.evaluate(qs0)
    dim_domain = len(qs0)
    dim_codomain = len(f0)

    jac = np.zeros((dim_codomain, dim_domain))
    eps = 1e-7
    for i in range(dim_domain):
        qs1 = deepcopy(qs0)
        qs1[i] += eps
        f1, _ = const.evaluate(qs1)
        jac[:, i] = (f1 - f0) / eps
    return jac


def test_translate(standard_problem: Problem):  # noqa
    n_wp = 10

    q_goal_cand = np.array([-0.78, 0.055, -1.37, -0.59, -0.494, -0.20, 1.87])
    traj = Trajectory.from_two_points(standard_problem.start, q_goal_cand, n_wp)
    qs0 = traj.numpy().flatten()

    satisfy_by_optimization(
        standard_problem.goal_const,
        standard_problem.box_const,
        standard_problem.global_ineq_const,
        None,
    )
    traj_eq_const, traj_ineq_const = translate(standard_problem, n_wp)
    _, jac_eq_anal = traj_eq_const.evaluate(qs0)
    _, jac_ineq_anal = traj_ineq_const.evaluate(qs0)

    jac_eq_numel = jac_numerical(traj_eq_const, qs0)
    jac_ineq_numel = jac_numerical(traj_ineq_const, qs0)

    np.testing.assert_almost_equal(jac_eq_anal, jac_eq_numel, decimal=4)
    np.testing.assert_almost_equal(jac_ineq_anal, jac_ineq_numel, decimal=4)


def test_sqp_based_solver(standard_problem: Problem):  # noqa
    n_wp = 10
    config = SQPBasedSolverConfig(n_wp)
    solver = SQPBasedSolver.setup(standard_problem, config)

    q_goal_cand = np.array([-0.78, 0.055, -1.37, -0.59, -0.494, -0.20, 1.87])
    init_traj = Trajectory.from_two_points(standard_problem.start, q_goal_cand, n_wp)
    assert not standard_problem.is_satisfied(init_traj)
    result = solver.solve(init_traj)
    assert result.traj is not None
    standard_problem.is_satisfied(result.traj)
