from typing import Optional, Tuple

import numpy as np
from skrobot.model import RobotModel

from skmp.constraint import AbstractIneqConst
from skmp.solver.nlp_solver.trajectory_constraint import TrajectoryInequalityConstraint


class CircleConstraint(AbstractIneqConst):
    def _evaluate(self, qs: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        r = 1.0
        fss = []
        gradss = []
        for q in qs:
            fs = np.expand_dims(np.sum(q**2) - r**2, axis=0)
            grads = np.expand_dims(2 * q, axis=0)
            fss.append(fs)
            gradss.append(grads)
        return np.array(fss), np.array(gradss)

    def _reflect_skrobot_model(self, robot_model: Optional[RobotModel]) -> None:
        pass


def test_TrajectoryInequalityConstraint():
    const = CircleConstraint()
    const.reflect_skrobot_model(None)

    ineq_const = TrajectoryInequalityConstraint.create_homogeneous(3, 2, const)
    points = np.array([[-1.0, 0.9], [0.0, 0.9], [1.0, 0.9]])
    f, _ = ineq_const.evaluate(points.flatten())
    np.testing.assert_almost_equal((f > 0), np.array([True, False, True]))

    ineq_const = TrajectoryInequalityConstraint.create_homogeneous(4, 2, const)
    points = np.array([[-1.0, 0.9], [-0.5, 0.9], [0.5, 0.9], [1.0, 0.9]])
    f, _ = ineq_const.evaluate(points.flatten())
    np.testing.assert_almost_equal((f > 0), np.array([True, True, True, True]))

    # test with motion_step_box enabled
    y = 1.0 - 1e-4
    ineq_const.motion_step_box = np.ones(2) * 0.1
    points = np.array([[-1.0, y], [-0.5, y], [0.55, y], [1.0, y]])
    f, _ = ineq_const.evaluate(points.flatten())
    n_eval_point = 5 + 11 + 6
    assert len(f) == n_eval_point

    # because [0.0, y] will be evaluated we have
    assert sum(f < 0) == 1
