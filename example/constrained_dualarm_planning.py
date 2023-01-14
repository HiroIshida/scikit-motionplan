import time

import numpy as np
from ompl import Algorithm, set_ompl_random_seed
from skrobot.model.primitives import Axis, Box
from skrobot.models import PR2
from skrobot.viewers import TrimeshSceneViewer

from skmp.constraint import (
    CollFreeConst,
    ConfigPointConst,
    IneqCompositeConst,
    PoseConstraint,
    RelativePoseConstraint,
)
from skmp.robot.pr2 import PR2Config
from skmp.robot.utils import set_robot_state
from skmp.satisfy import satisfy_by_optimization_with_budget
from skmp.solver import (
    OMPLSolver,
    OMPLSolverConfig,
    Problem,
    SQPBasedSolver,
    SQPBasedSolverConfig,
)

np.random.seed(0)
set_ompl_random_seed(0)


if __name__ == "__main__":
    pr2 = PR2(use_tight_joint_limit=False)
    pr2.reset_manip_pose()
    pr2.torso_lift_joint.joint_angle(0.1)

    with_base = True

    robot_config = PR2Config(control_arm="dual", with_base=with_base)
    colkin = robot_config.get_collision_kin()
    efkin = robot_config.get_endeffector_kin()
    efkin.reflect_skrobot_model(pr2)
    colkin.reflect_skrobot_model(pr2)

    selcol_const = robot_config.get_neural_selcol_const(pr2)

    relative_position = np.array([0, 0, -0.2])
    box_const = robot_config.get_box_const()

    # determine the start pose
    rarm_target = Axis(axis_radius=0.01, axis_length=0.05)
    rarm_target.translate([0.6, -0.1, 0.5])
    rarm_target.rotate(np.pi * 0.5, "x")
    larm_target = Axis(axis_radius=0.01, axis_length=0.05)
    larm_target.newcoords(rarm_target.copy_worldcoords())
    larm_target.translate(relative_position)
    eqconst = PoseConstraint.from_skrobot_coords([rarm_target, larm_target], efkin, pr2)

    res = satisfy_by_optimization_with_budget(eqconst, box_const, selcol_const, None)
    q_start = res.q

    # determine the goal pose
    translation = 0.4
    rarm_target.translate([0, 0, translation], wrt="world")
    larm_target.translate([0, 0, translation], wrt="world")
    eqconst = PoseConstraint.from_skrobot_coords([rarm_target, larm_target], efkin, pr2)

    res = satisfy_by_optimization_with_budget(eqconst, box_const, selcol_const, q_start)
    q_goal = res.q

    # create planning problem instance
    relconst = RelativePoseConstraint(relative_position, efkin, PR2())

    obstacle = Box(extents=[0.15, 3.0, 0.2], with_sdf=True)
    obstacle.translate(np.array([0.6, 0.0, 0.7]))
    assert obstacle.sdf is not None
    collfree_const = CollFreeConst(colkin, obstacle.sdf, pr2)

    goal_eq_const = ConfigPointConst(q_goal)
    global_ineq_const = IneqCompositeConst([collfree_const, selcol_const])
    problem = Problem(q_start, box_const, goal_eq_const, global_ineq_const, relconst)

    # solve by ompl
    ompl_config = OMPLSolverConfig(
        n_max_call=300000, algorithm=Algorithm.KPIECE1, algorithm_range=0.1
    )
    ompl_solver = OMPLSolver.init(ompl_config)
    ompl_solver.setup(problem)
    result = ompl_solver.solve()
    assert result.traj is not None
    print("time to solve by ompl: {}".format(result.time_elapsed))

    # solve by sqp
    n_wp = 20
    sqp_config = SQPBasedSolverConfig(
        n_wp=n_wp, motion_step_satisfaction="debug_ignore", ctol_eq=1e-3
    )
    nlp_solver = SQPBasedSolver.init(sqp_config)
    nlp_solver.setup(problem)
    nlp_result = nlp_solver.solve(result.traj.resample(n_wp))
    assert nlp_result.traj is not None
    print("time to solve by sqp: {}".format(nlp_result.time_elapsed))

    viewer = TrimeshSceneViewer(resolution=(640, 480))
    viewer.add(pr2)
    viewer.add(obstacle)

    viewer.show()

    for q in nlp_result.traj:
        set_robot_state(pr2, robot_config._get_control_joint_names(), q, with_base)
        viewer.redraw()
        time.sleep(1.0)

    print("==> Press [q] to close window")
    while not viewer.has_exit:
        time.sleep(0.1)
        viewer.redraw()
