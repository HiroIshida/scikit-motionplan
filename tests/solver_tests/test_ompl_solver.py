from utils import create_standard_problem

from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig


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
        assert result_with_guess.time_elapsed is not None
        assert result.time_elapsed is not None
        assert result_with_guess.time_elapsed < result.time_elapsed
        assert result_with_guess.n_call < result.n_call


def test_solver_with_guiding_traj():
    problem = create_standard_problem()
    solcon = OMPLSolverConfig()
    solver = OMPLSolver.init(solcon)
    solver.setup(problem)
    result_pre = solver.solve()
    assert result_pre.traj is not None

    for exp_backend in ["lightning", "ertconnect"]:
        solcon = OMPLSolverConfig(expbased_planner_backend=exp_backend)  # type: ignore[arg-type]
        solver = OMPLSolver.init(solcon)
        solver.setup(problem)
        result = solver.solve(result_pre.traj)
        assert result.traj is not None


if __name__ == "__main__":
    # test_lightning_solver()
    test_ompl_solver()
