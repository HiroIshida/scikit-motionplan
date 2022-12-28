from utils import create_standard_problem

from skmp.solver import OMPLSolver


def test_ompl_solver():  # noqa
    problem = create_standard_problem()
    solver = OMPLSolver.setup(problem)
    result = solver.solve()
    assert result.traj is not None

    assert problem.is_satisfied(result.traj)


if __name__ == "__main__":
    test_ompl_solver()
