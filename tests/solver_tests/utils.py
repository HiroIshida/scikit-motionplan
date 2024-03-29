import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Box
from skrobot.models import PR2
from tinyfk import BaseType

from skmp.constraint import CollFreeConst, PoseConstraint
from skmp.robot.pr2 import PR2Config
from skmp.solver.interface import Problem


def create_standard_problem(easy: bool = False, feasible: bool = True) -> Problem:
    # setup kinematics
    pr2 = PR2()
    pr2.reset_manip_pose()
    config = PR2Config(base_type=BaseType.FIXED)
    colkin = config.get_collision_kin()
    efkin = config.get_endeffector_kin()

    # box
    box_const = config.get_box_const()

    # goal
    start = np.array([0.564, 0.35, -0.74, -0.7, -0.7, -0.17, -0.63])
    if feasible:
        target = Coordinates(pos=[0.7, -0.6, 1.0])
    else:
        target = Coordinates(pos=[0.7, -0.6, 3.0])
    goal_eq_const = PoseConstraint.from_skrobot_coords([target], efkin, pr2)

    # global ineq
    if easy:
        obstacle = Box(extents=[0.3, 0.1, 0.3], with_sdf=True)
        obstacle.translate(np.array([2.0, -0.2, 0.9]))
    else:
        obstacle = Box(extents=[0.7, 0.5, 1.2], with_sdf=True)
        obstacle.translate(np.array([0.85, -0.2, 0.9]))
    assert obstacle.sdf is not None
    global_ienq_const = CollFreeConst(colkin, obstacle.sdf, pr2)

    return Problem(start, box_const, goal_eq_const, global_ienq_const, None)
