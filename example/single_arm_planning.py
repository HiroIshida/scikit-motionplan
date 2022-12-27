import time
from typing import Optional

import numpy as np
from skrobot.model.primitives import Axis, Box
from skrobot.models import PR2
from skrobot.viewers import TrimeshSceneViewer

from skmp.constraint import CollFreeConst, ConfigPointConst, PoseConstraint
from skmp.robot.pr2 import PR2Config
from skmp.robot.utils import set_robot_state
from skmp.solver import OMPLSolver, Problem, SQPBasedSolver, SQPBasedSolverConfig

if __name__ == "__main__":
    pr2 = PR2(use_tight_joint_limit=True)
    pr2.reset_manip_pose()
    robot_config = PR2Config(with_base=False)
    colkin = robot_config.get_collision_kin()
    efkin = robot_config.get_endeffector_kin()
    efkin.reflect_skrobot_model(pr2)
    colkin.reflect_skrobot_model(pr2)

    use_pose_constraint = True

    start = np.array([0.564, 0.35, -0.74, -0.7, -0.7, -0.17, -0.63])
    box_const = robot_config.get_box_const()

    target: Optional[Axis]
    if use_pose_constraint:
        target = Axis(axis_radius=0.01, axis_length=0.05)
        target.translate([0.7, -0.6, 1.0])
        goal_eq_const = PoseConstraint.from_skrobot_coords([target], efkin)
    else:
        target = None
        goal = np.array([-0.78, 0.055, -1.37, -0.59, -0.494, -0.20, 1.87])
        goal_eq_const = ConfigPointConst(goal)  # type: ignore[assignment]

    obstacle = Box(extents=[0.7, 0.5, 1.2], with_sdf=True)
    obstacle.translate(np.array([0.85, -0.2, 0.9]))
    assert obstacle.sdf is not None
    global_ienq_const = CollFreeConst(colkin, obstacle.sdf, 3)

    problem = Problem(start, box_const, goal_eq_const, global_ienq_const, None)

    ompl_solver = OMPLSolver.setup(problem)
    result = ompl_solver.solve()
    print(result.time_elapsed)
    assert result.traj is not None

    n_wp = 30
    sqp_config = SQPBasedSolverConfig(n_wp=n_wp)
    nlp_solver = SQPBasedSolver.setup(problem, sqp_config)
    result = nlp_solver.solve(result.traj.resample(n_wp))
    assert result.traj is not None

    viewer = TrimeshSceneViewer(resolution=(640, 480))
    viewer.add(pr2)
    viewer.add(obstacle)
    if target is not None:
        viewer.add(target)

    viewer.show()
    time.sleep(1.0)
    for q in result.traj:
        set_robot_state(pr2, robot_config._get_control_joint_names(), q)
        viewer.redraw()
        time.sleep(0.6)

    print("==> Press [q] to close window")
    while not viewer.has_exit:
        time.sleep(0.1)
        viewer.redraw()
