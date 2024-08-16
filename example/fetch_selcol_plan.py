import argparse
import time

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis, Box
from skrobot.models import Fetch
from skrobot.sdf import UnionSDF
from skrobot.viewers import PyrenderViewer

from skmp.constraint import CollFreeConst, IneqCompositeConst, PoseConstraint
from skmp.robot.fetch import FetchConfig
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.solver.interface import Problem
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skmp.visualization.collision_visualizer import CollisionSphereVisualizationManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", action="store_true", help="parallelize")
    parser.add_argument("--visualize", action="store_true", help="visualize")
    args = parser.parse_args()

    np.random.seed(0)

    # basic setups
    conf = FetchConfig()
    fetch = Fetch()
    fetch.reset_pose()
    q_init = get_robot_state(fetch, conf.get_control_joint_names())
    efkin = conf.get_endeffector_kin()
    box_const = conf.get_box_const()

    # define (+self) collision free constraint
    # NOTE: for handling self-collision of fetch robot, we handle self bodies are obstacles
    self_body_obstacles = conf.get_self_body_obstacles()
    table = Box([1.0, 2.0, 0.05], with_sdf=True)
    table.translate([1.0, 0.0, 0.8])
    ground = Box([2.0, 2.0, 0.05], with_sdf=True)
    all_obstacles = [table, ground] + self_body_obstacles

    sdf = UnionSDF([obs.sdf for obs in all_obstacles])
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
        v.add(table)
        v.add(ground)
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

        time.sleep(1000)
