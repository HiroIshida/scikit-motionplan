import numpy as np
from utils import create_standard_problem

from skmp.solver.datadriven import NearestNeigborSolver
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig


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
    for knn in range(1, 6):
        nn_solver = NearestNeigborSolver.init(OMPLSolver, conf, dataset, knn, conservative=True)  # type: ignore

        if knn == 1:
            nn_solver.infeasibility_threshold == 1

        nn_solver.setup(problem)
        ret = nn_solver.solve(np.zeros((1,)))
        assert ret.traj is not None
        assert nn_solver.previous_est_positive
        assert not nn_solver.previous_false_positive

        # shold detec the problem as infeasible...
        nn_solver.setup(problem)
        ret = nn_solver.solve(np.ones((1,)))
        assert ret.traj is None
        ret = nn_solver.solve(-np.ones((1,)))
        assert ret.traj is None
        assert not nn_solver.previous_est_positive
        assert nn_solver.previous_false_positive is None


if __name__ == "__main__":
    test_nearest_neighbor_solver()
