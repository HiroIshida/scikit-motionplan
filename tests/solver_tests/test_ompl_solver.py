from utils import create_standard_problem

from skmp.solver.ompl_solver import LightningSolver, OMPLSolver, OMPLSolverConfig


def test_ompl_solver():  # noqa
    problem = create_standard_problem()
    solver = OMPLSolver.init(OMPLSolverConfig())
    solver.setup(problem)
    result = solver.solve()
    assert result.traj is not None

    assert problem.is_satisfied(result.traj)


def test_lightning_solver():
    problem = create_standard_problem()
    solcon = OMPLSolverConfig()
    solver = OMPLSolver.init(solcon)
    solver.setup(problem)
    result = solver.solve()
    assert result.traj is not None

    lightning = LightningSolver.init(solcon, [result.traj])
    lightning.setup(problem)
    lightning_result = lightning.solve()
    assert lightning_result.traj is not None


if __name__ == "__main__":
    test_lightning_solver()
