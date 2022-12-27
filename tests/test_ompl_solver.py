import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Box
from skrobot.models import PR2

from skmp.constraint import CollFreeConst, PoseConstraint
from skmp.solver import OMPLSolver, Problem
from skmp.robot.pr2 import PR2Config


def test_ompl_solver():
    # setup kinematics
    pr2 = PR2()
    pr2.reset_manip_pose()
    config = PR2Config(with_base=False)
    colkin = config.get_collision_kin()
    efkin = config.get_endeffector_kin()
    efkin.reflect_skrobot_model(pr2)
    colkin.reflect_skrobot_model(pr2)

    # box
    box_const = config.get_box_const()

    # goal
    start = np.array([0.564, 0.35, -0.74, -0.7, -0.7, -0.17, -0.63])
    target = Coordinates(pos=[0.8, -0.6, 1.1])
    goal_eq_const = PoseConstraint.from_skrobot_coords([target], efkin)

    # global ineq
    obstacle = Box(extents=[0.7, 0.5, 1.2], with_sdf=True)
    obstacle.translate(np.array([0.85, -0.2, 0.9]))
    assert obstacle.sdf is not None
    global_ienq_const = CollFreeConst(colkin, obstacle.sdf, 3)

    problem = Problem(start, box_const, goal_eq_const, global_ienq_const, None)

    solver = OMPLSolver.setup(problem)
    result = solver.solve()
    assert result.traj is not None

    # check goal satsifaction
    vals, _ = goal_eq_const.evaluate_single(result.traj[-1], with_jacobian=False)
    assert vals.dot(vals) < 1e-6

    # check ineq satisfaction
    valss, _ = global_ienq_const.evaluate(result.traj.numpy(), with_jacobian=False)
    assert np.all(valss > 0)


if __name__ == "__main__":
    test_ompl_solver()
