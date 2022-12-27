import numpy as np
from test_solver_interface import standard_problem  # noqa

from skmp.solver import OMPLSolver, Problem


def test_ompl_solver(standard_problem: Problem):  # noqa
    solver = OMPLSolver.setup(standard_problem)
    result = solver.solve()
    assert result.traj is not None

    # check goal satsifaction
    vals, _ = standard_problem.goal_const.evaluate_single(result.traj[-1], with_jacobian=False)
    assert vals.dot(vals) < 1e-6

    # check ineq satisfaction
    assert standard_problem.global_ineq_const is not None
    valss, _ = standard_problem.global_ineq_const.evaluate(result.traj.numpy(), with_jacobian=False)
    assert np.all(valss > 0)
