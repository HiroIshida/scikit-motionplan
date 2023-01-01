import time
from typing import Optional

import numpy as np
from ompl import Algorithm, set_ompl_random_seed
from skrobot.model.primitives import Axis, Box
from skrobot.models import PR2
from skrobot.viewers import TrimeshSceneViewer

from skmp.constraint import (
    CollFreeConst,
    ConfigPointConst,
    IneqCompositeConst,
    PairWiseSelfCollFreeConst,
    PoseConstraint,
)
from skmp.robot.pr2 import PR2Config
from skmp.robot.utils import set_robot_state
from skmp.solver import (
    OMPLSolver,
    OMPLSolverConfig,
    Problem,
    SQPBasedSolver,
    SQPBasedSolverConfig,
)
from skmp.visualization import CollisionSphereVisualizationManager

np.random.seed(0)
set_ompl_random_seed(0)

if __name__ == "__main__":
    pr2 = PR2(use_tight_joint_limit=True)
    pr2.reset_manip_pose()
    pr2.torso_lift_joint.joint_angle(0.1)

    robot_config = PR2Config(with_base=False)
    colkin = robot_config.get_collision_kin()
    efkin = robot_config.get_endeffector_kin()
    efkin.reflect_skrobot_model(pr2)
    colkin.reflect_skrobot_model(pr2)

    use_pose_constraint = True
    neural_selcol = False

    start = np.array([0.564, 0.35, -0.74, -0.7, -0.7, -0.17, -0.63])
    box_const = robot_config.get_box_const()

    # create equality constraint
    target: Optional[Axis]
    if use_pose_constraint:
        target = Axis(axis_radius=0.01, axis_length=0.05)
        target.translate([0.7, -0.6, 1.0])
        goal_eq_const = PoseConstraint.from_skrobot_coords([target], efkin, pr2)
    else:
        target = None
        goal = np.array([-0.78, 0.055, -1.37, -0.59, -0.494, -0.20, 1.87])
        goal_eq_const = ConfigPointConst(goal)  # type: ignore[assignment]

    # create inequality constraint
    obstacle = Box(extents=[0.5, 0.5, 1.2], with_sdf=True)
    obstacle.translate(np.array([0.8, -0.2, 0.9]))
    assert obstacle.sdf is not None
    collfree_const = CollFreeConst(colkin, obstacle.sdf, pr2)

    if neural_selcol:
        selcolfree_const = robot_config.get_neural_selcol_const(pr2)
        selcolfree_const.reflect_skrobot_model(pr2)
    else:
        selcolfree_const = PairWiseSelfCollFreeConst(colkin, pr2)  # type: ignore[assignment]

    global_ineq_const = IneqCompositeConst([collfree_const, selcolfree_const])

    # construct problem
    problem = Problem(start, box_const, goal_eq_const, global_ineq_const, None)

    ompl_config = OMPLSolverConfig(n_max_eval=50000, algorithm=Algorithm.RRT)
    ompl_solver = OMPLSolver.setup(problem, config=ompl_config)
    result = ompl_solver.solve()
    print(result.time_elapsed)
    assert result.traj is not None

    n_wp = 30
    sqp_config = SQPBasedSolverConfig(n_wp=n_wp)
    nlp_solver = SQPBasedSolver.setup(problem, sqp_config)
    result = nlp_solver.solve(result.traj.resample(n_wp))
    print(result.time_elapsed)
    assert result.traj is not None

    viewer = TrimeshSceneViewer(resolution=(640, 480))
    colvis = CollisionSphereVisualizationManager(colkin, viewer)
    viewer.add(pr2)
    viewer.add(obstacle)
    if target is not None:
        viewer.add(target)

    viewer.show()
    time.sleep(1.0)
    for q in result.traj.resample(n_wp):
        set_robot_state(pr2, robot_config._get_control_joint_names(), q)
        colvis.update(pr2, obstacle.sdf)
        viewer.redraw()
        time.sleep(0.6)

    print("==> Press [q] to close window")
    while not viewer.has_exit:
        time.sleep(0.1)
        viewer.redraw()
