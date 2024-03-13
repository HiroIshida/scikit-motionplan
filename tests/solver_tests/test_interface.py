import numpy as np
from utils import create_standard_problem

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
