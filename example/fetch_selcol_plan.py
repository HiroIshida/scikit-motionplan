import argparse
import time

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis
from skrobot.models import Fetch
from skrobot.viewers import PyrenderViewer

from tinyfk import RotationType
from skmp.constraint import CollFreeConst, IneqCompositeConst, PoseConstraint
from skmp.robot.fetch import FetchConfig
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.robot.robot import RobotSurrounding, SurroundingList, EndEffectorList
from skmp.solver.interface import Problem
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skmp.visualization.collision_visualizer import CollisionSphereVisualizationManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", action="store_true", help="parallelize")
    parser.add_argument("--visualize", action="store_true", help="visualize")
    args = parser.parse_args()

    np.random.seed(0)

    end_effector_list = EndEffectorList(
        link_names=["gripper_link"],
        end_effector_names=["gripper_link"],
        positions=[[0.0, 0, 0.0]],
        rpys=[[0, 0, 0]],
    )

    # basic setups
    conf = FetchConfig(end_effector_list)
    fetch = Fetch()
    fetch.reset_pose()
    q_init = get_robot_state(fetch, conf.get_control_joint_names())
    efkin = conf.get_endeffector_kin(rot_type = RotationType.RPY)
    box_const = conf.get_box_const()

    # define surrounding
    surrounding_list = SurroundingList(
        name=["table", "ground"],
        shape=["Box", "Box"],
        size=[[1.0, 2.0, 0.05], [2.0, 2.0, 0.05]],
        position=[[1.0, 0.0, 0.8], [0.0, 0.0, 0.0]],
        rpy=[[0, 0, 0], [0, 0, 0]],
        color=[[120, 120, 120, 120], [120, 120, 120, 120]],
    )
    surrounding = RobotSurrounding(surrounding_list)
    # define (+self) collision free constraint
    # NOTE: for handling self-collision of fetch robot, we handle self bodies are obstacles
    self_body_obstacles = conf.get_self_body_obstacles()
    surrounding.append_sdf(self_body_obstacles)

    sdf = surrounding.get_sdf_list()
    colkin = conf.get_collision_kin()
    coll_free_const = CollFreeConst(colkin, sdf, fetch)

    # define goal constraint
    goal_coords = Coordinates([0.8, 0.0, 0.95])
    goal_const = PoseConstraint.from_skrobot_coords([goal_coords], efkin, fetch)

    # solve IK + path-planning problem
    solver = OMPLSolver.init(
        OMPLSolverConfig(n_max_call=10000, simplify=True, algorithm_range=None)
    )
    if args.parallel:
        # NOTE: for such an easy problem, parallelization is actually rather slow
        solver = solver.as_parallel_solver(4)  # type: ignore

    ineq_const = IneqCompositeConst([coll_free_const])
    problem = Problem(q_init, box_const, goal_const, ineq_const, None, motion_step_box_=0.1)
    solver.setup(problem)

    ts = time.time()
    res = solver.solve()
    assert res.traj is not None
    print("success!, total time: ", time.time() - ts)

    if args.visualize:
        v = PyrenderViewer()
        v.add(fetch)
        object_list = surrounding.get_object_list()
        for obj in object_list:
            v.add(obj)
        target_axis = Axis.from_coords(goal_coords)
        v.add(target_axis)
        colvisman = CollisionSphereVisualizationManager(colkin, v, sdf)  # visualize approx spheres
        v.show()
        time.sleep(1.0)
        for q in res.traj.resample(50):
            set_robot_state(fetch, conf.get_control_joint_names(), q)
            colvisman.update(fetch)
            v.redraw()
            time.sleep(0.4)

        input("Press any key to exit")
