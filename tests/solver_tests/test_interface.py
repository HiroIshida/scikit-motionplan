import numpy as np
from utils import create_standard_problem

from skmp.solver.interface import NearestNeigborSolver
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig


def test_timeout():  # noqa
    # try to solve infeasible problem with massive computation budget
    # check if interruption by timeout is correctly handled
    problem = create_standard_problem(feasible=False)
    timeout = 1.5
    conf = OMPLSolverConfig(n_max_call=100000, n_max_satisfaction_trial=100000, timeout=timeout)
    solver = OMPLSolver.init(conf)
    solver.setup(problem)
    result = solver.solve()
    assert result.traj is None
    assert result.time_elapsed is not None
    assert np.abs(result.time_elapsed - timeout) < 1e-3


def test_nearest_neighbor_solver():
    problem = create_standard_problem(feasible=True)
    conf = OMPLSolverConfig(n_max_call=100000, n_max_satisfaction_trial=100000, timeout=30)
    solver = OMPLSolver.init(conf)
    solver.setup(problem)
    ret = solver.solve()
    assert ret.traj is not None

    descs = list(np.expand_dims(np.linspace(-0.1, 0.1, 20), axis=1))
    trajs = [ret.traj] * len(descs)
    for _ in range(30):
        desc = np.random.randn()
        if np.abs(desc) > 0.1:
            descs.append(np.array([desc]))  # type: ignore
            trajs.append(None)  # type: ignore

    dataset = list(zip(descs, trajs))
    nn_solver = NearestNeigborSolver.init(OMPLSolver, conf, dataset, 5)  # type: ignore
    nn_solver.setup(problem)
    ret = nn_solver.solve(np.zeros(2))
    # dont care about the result, just check if it runs without error


if __name__ == "__main__":
    test_nearest_neighbor_solver()
