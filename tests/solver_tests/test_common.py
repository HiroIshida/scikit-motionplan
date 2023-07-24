import numpy as np
from utils import create_standard_problem

from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig


def test_ompl_solver():  # noqa
    # try to solve infeasible problem with massive computation budget
    # check if interruption by timeout is correctly handled
    problem = create_standard_problem(feasible=False)
    solver = OMPLSolver.init(OMPLSolverConfig(n_max_call=100000, n_max_satisfaction_trial=100000))
    solver.setup(problem)
    result = solver.solve(timeout=1)
    assert result.traj is None
    assert result.time_elapsed is not None
    assert np.abs(result.time_elapsed - 1) < 1e-3
