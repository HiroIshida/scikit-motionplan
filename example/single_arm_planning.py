import numpy as np
from ompl import Algorithm, set_ompl_random_seed
from skrobot.model.primitives import Axis, Box
from skrobot.models import PR2
from tinyfk import BaseType

from skmp.constraint import (
    CollFreeConst,
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
    smooth_by_nlp = False
    base_type = BaseType.FIXED

    # initial setup
    pr2 = PR2()
    pr2.reset_manip_pose()
    pr2.torso_lift_joint.joint_angle(0.1)

    robot_config = PR2Config(base_type=base_type)
    colkin = robot_config.get_collision_kin()
    efkin = robot_config.get_endeffector_kin()
    efkin.reflect_skrobot_model(pr2)
    colkin.reflect_skrobot_model(pr2)

    # define start configuration
    start = np.array([0.564, 0.35, -0.74, -0.7, -0.7, -0.17, -0.63])
    if base_type == BaseType.PLANER:
        start = np.hstack([start, np.zeros(3)])
    elif base_type == BaseType.FLOATING:
        start = np.hstack([start, np.zeros(6)])

    # create goal equality constraint
    target = Axis(axis_radius=0.01, axis_length=0.05)
    target.translate([0.7, -0.6, 1.0])
    goal_eq_const = PoseConstraint.from_skrobot_coords([target], efkin, pr2)

    # create inequality constraint
    obstacle = Box(extents=[0.5, 0.5, 1.2], with_sdf=True)
    obstacle.translate(np.array([0.8, -0.2, 0.9]))
    assert obstacle.sdf is not None
    collfree_const = CollFreeConst(colkin, obstacle.sdf, pr2)
    selcolfree_const = PairWiseSelfCollFreeConst(colkin, pr2, only_closest_feature=True)  # type: ignore[assignment]
    global_ineq_const = IneqCompositeConst([collfree_const, selcolfree_const])

    # construct problem
    box_const = robot_config.get_box_const()
    motion_step_box = robot_config.get_default_motion_step_box()
    problem = Problem(
        start, box_const, goal_eq_const, global_ineq_const, None, motion_step_box_=motion_step_box
    )

    # solve problem by ompl
    ompl_config = OMPLSolverConfig(n_max_call=10000, algorithm=Algorithm.RRT, simplify=True)
    ompl_solver = OMPLSolver.init(ompl_config)
    ompl_solver.setup(problem)
    result = ompl_solver.solve()
    print(result.time_elapsed)
    assert result.traj is not None

    # smooth by trajectory optimization (if enabled)
    if smooth_by_nlp:
        n_wp = 40
        sqp_config = SQPBasedSolverConfig(n_wp=n_wp)
        nlp_solver = SQPBasedSolver.init(sqp_config)
        nlp_solver.setup(problem)
        result = nlp_solver.solve(result.traj.resample(n_wp))  # type: ignore
        print(result.time_elapsed)
        assert result.traj is not None

    # visualization using skrobot
    def robot_updator(robot, q) -> None:
        set_robot_state(robot, robot_config._get_control_joint_names(), q, base_type=base_type)

    geometry = [obstacle]
    if target is not None:
        geometry.append(target)
    vis = InteractiveSolutionVisualizer(pr2, geometry, robot_updator=robot_updator)
    assert result.traj is not None
    vis.visualize_trajectory(result.traj.resample(n_wp))
