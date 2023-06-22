from typing import Optional

import numpy as np
from ompl import Algorithm, set_ompl_random_seed
from skrobot.model.primitives import Axis, Box
from skrobot.models import PR2
from tinyfk import BaseType

from skmp.constraint import (
    CollFreeConst,
    ConfigPointConst,
    IneqCompositeConst,
    PairWiseSelfCollFreeConst,
    PoseConstraint,
)
from skmp.robot.pr2 import PR2Config
from skmp.robot.utils import set_robot_state
from skmp.solver.interface import Problem
from skmp.solver.nlp_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skmp.visualization.solution_visualizer import InteractiveSolutionVisualizer

np.random.seed(0)
set_ompl_random_seed(0)

if __name__ == "__main__":
    base_type = BaseType.FIXED

    pr2 = PR2(use_tight_joint_limit=False)
    pr2.reset_manip_pose()
    pr2.torso_lift_joint.joint_angle(0.1)

    robot_config = PR2Config(base_type=base_type)
    colkin = robot_config.get_collision_kin()
    efkin = robot_config.get_endeffector_kin()
    efkin.reflect_skrobot_model(pr2)
    colkin.reflect_skrobot_model(pr2)

    use_pose_constraint = True
    smooth_by_nlp = False

    start = np.array([0.564, 0.35, -0.74, -0.7, -0.7, -0.17, -0.63])
    if base_type == BaseType.PLANER:
        start = np.hstack([start, np.zeros(3)])
    elif base_type == BaseType.FLOATING:
        start = np.hstack([start, np.zeros(6)])
    box_const = robot_config.get_box_const()

    # keep the third joint angle static
    eps = 1e-3
    box_const.ub[2] = start[2] + eps
    box_const.lb[2] = start[2] - eps

    # create equality constraint
    target: Optional[Axis]
    if use_pose_constraint:
        target = Axis(axis_radius=0.01, axis_length=0.05)
        target.translate([0.7, -0.6, 1.0])
        goal_eq_const = PoseConstraint.from_skrobot_coords([target], efkin, pr2)
    else:
        target = None
        goal = np.array([-0.78, 0.055, -1.37, -0.59, -0.494, -0.20, 1.87])
        if base_type == BaseType.PLANER:
            goal = np.hstack([goal, np.zeros(3)])
        elif base_type == BaseType.FLOATING:
            goal = np.hstack([goal, np.zeros(6)])
        goal_eq_const = ConfigPointConst(goal)  # type: ignore[assignment]

    # create inequality constraint
    obstacle = Box(extents=[0.5, 0.5, 1.2], with_sdf=True)
    obstacle.translate(np.array([0.8, -0.2, 0.9]))
    assert obstacle.sdf is not None
    collfree_const = CollFreeConst(colkin, obstacle.sdf, pr2)
    selcolfree_const = PairWiseSelfCollFreeConst(colkin, pr2, only_closest_feature=True)  # type: ignore[assignment]
    global_ineq_const = IneqCompositeConst([collfree_const, selcolfree_const])

    # construct problem
    problem = Problem(start, box_const, goal_eq_const, global_ineq_const, None)

    ompl_config = OMPLSolverConfig(n_max_call=10000, algorithm=Algorithm.RRT, simplify=True)
    ompl_solver = OMPLSolver.init(ompl_config)
    ompl_solver.setup(problem)
    result = ompl_solver.solve()
    print(result.time_elapsed)
    assert result.traj is not None
    print(result.traj)

    n_wp = 40
    if smooth_by_nlp:
        sqp_config = SQPBasedSolverConfig(n_wp=n_wp)
        nlp_solver = SQPBasedSolver.init(sqp_config)
        nlp_solver.setup(problem)
        result = nlp_solver.solve(result.traj.resample(n_wp))  # type: ignore
        print(result.time_elapsed)
        assert result.traj is not None

    def robot_updator(robot, q) -> None:
        set_robot_state(robot, robot_config._get_control_joint_names(), q, base_type=base_type)

    geometry = [obstacle]
    if target is not None:
        geometry.append(target)
    vis = InteractiveSolutionVisualizer(pr2, geometry, robot_updator=robot_updator)
    assert result.traj is not None
    vis.visualize_trajectory(result.traj.resample(n_wp))
