from test_solver_interface import standard_problem  # noqa

from skmp.solver import OMPLSolver, Problem


def test_ompl_solver(standard_problem: Problem):  # noqa
    solver = OMPLSolver.setup(standard_problem)
    result = solver.solve()
    assert result.traj is not None

    assert standard_problem.is_satisfied(result.traj)
