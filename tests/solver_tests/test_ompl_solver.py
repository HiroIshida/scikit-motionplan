from utils import create_standard_problem

from skmp.solver import OMPLSolver, OMPLSolverConfig


def test_ompl_solver():  # noqa
    problem = create_standard_problem()
    solver = OMPLSolver.init(OMPLSolverConfig())
    solver.setup(problem)
    result = solver.solve()
    assert result.traj is not None

    assert problem.is_satisfied(result.traj)


if __name__ == "__main__":
    test_ompl_solver()
