from copy import deepcopy

import numpy as np
from ompl import set_ompl_random_seed
from utils import create_standard_problem

from skmp.satisfy import satisfy_by_optimization
from skmp.solver.nlp_solver.sqp_based_solver import (
    SQPBasedSolver,
    SQPBasedSolverConfig,
    translate,
)
from skmp.solver.nlp_solver.trajectory_constraint import TrajectoryConstraint
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
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


def test_translate():
    problem = create_standard_problem()
    n_wp = 10

    q_goal_cand = np.array([-0.78, 0.055, -1.37, -0.59, -0.494, -0.20, 1.87])
    traj = Trajectory.from_two_points(problem.start, q_goal_cand, n_wp)
    qs0 = traj.numpy().flatten()

    satisfy_by_optimization(
        problem.goal_const,
        problem.box_const,
        problem.global_ineq_const,
        None,
    )
    traj_eq_const, traj_ineq_const = translate(problem, n_wp)
    _, jac_eq_anal = traj_eq_const.evaluate(qs0)
    _, jac_ineq_anal = traj_ineq_const.evaluate(qs0)

    jac_eq_numel = jac_numerical(traj_eq_const, qs0)
    jac_ineq_numel = jac_numerical(traj_ineq_const, qs0)

    np.testing.assert_almost_equal(jac_eq_anal, jac_eq_numel, decimal=4)
    np.testing.assert_almost_equal(jac_ineq_anal, jac_ineq_numel, decimal=4)


def test_sqp_based_solver():
    problem = create_standard_problem()

    np.random.seed(0)
    set_ompl_random_seed(1)
    ompl_solver = OMPLSolver.init(OMPLSolverConfig())
    ompl_solver.setup(problem)
    init_traj = ompl_solver.solve().traj
    assert init_traj is not None

    config1 = SQPBasedSolverConfig(n_wp=30, motion_step_satisfaction="implicit")
    config2 = SQPBasedSolverConfig(n_wp=50, motion_step_satisfaction="explicit")
    for config in [config1, config2]:
        solver = SQPBasedSolver.init(config)
        solver.setup(problem)
        result = solver.solve(init_traj.resample(config.n_wp))
        assert result.traj is not None
        assert problem.is_satisfied(result.traj)

    config3 = SQPBasedSolverConfig(n_wp=15, motion_step_satisfaction="post")
    solver = SQPBasedSolver.init(config3)
    solver.setup(problem)
    result = solver.solve(init_traj.resample(config3.n_wp))
    # internal solver supposed to report "solved"
    # but, the solver will report not solved because n_wp = 15 is too small
    # to satisfy the motion_step constraint
    assert result.osqpsqp_raw_result is not None
    assert result.osqpsqp_raw_result.success
    assert result.traj is None


def test_sqp_based_solver_without_init():
    problem = create_standard_problem(easy=True)
    config = SQPBasedSolverConfig(n_wp=30, motion_step_satisfaction="implicit")
    solver = SQPBasedSolver.init(config)
    solver.setup(problem)
    res = solver.solve()
    assert res.traj is not None


if __name__ == "__main__":
    test_sqp_based_solver()
