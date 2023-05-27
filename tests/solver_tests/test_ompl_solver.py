import numpy as np
from utils import create_standard_problem

from skmp.solver.ompl_solver import OMPLDataDrivenSolver, OMPLSolver, OMPLSolverConfig


def test_ompl_solver():  # noqa
    problem = create_standard_problem()
    solver = OMPLSolver.init(OMPLSolverConfig())
    solver.setup(problem)
    result = solver.solve()
    assert result.traj is not None

    assert problem.is_satisfied(result.traj)
    # test with initial guess

    conf1 = OMPLSolverConfig(expbased_planner_backend="lightning")
    conf2 = OMPLSolverConfig(expbased_planner_backend="ertconnect")

    for conf in [conf1, conf2]:
        solver = OMPLSolver.init(conf)
        solver.setup(problem)
        result_with_guess = solver.solve(result.traj)
        assert result_with_guess.time_elapsed < result.time_elapsed
        assert result_with_guess.n_call < result.n_call


def test_datadriven_solver():
    problem = create_standard_problem()
    solcon = OMPLSolverConfig()
    solver = OMPLSolver.init(solcon)
    solver.setup(problem)
    result = solver.solve()
    assert result.traj is not None

    key = np.hstack([result.traj[0], result.traj[-1]])
    dataset = [(key, result.traj)]

    for exp_backend in ["lightning", "ertconnect"]:
        solcon = OMPLSolverConfig(expbased_planner_backend=exp_backend)  # type: ignore[arg-type]
        ddsolver = OMPLDataDrivenSolver.init(solcon, dataset)
        ddsolver.setup(problem)
        result = ddsolver.solve(key)
        assert result.traj is not None


if __name__ == "__main__":
    # test_lightning_solver()
    test_ompl_solver()
