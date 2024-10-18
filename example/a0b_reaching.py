import argparse
import time

import numpy as np
from skrobot.model.primitives import Axis
from skrobot.utils.urdf import mesh_simplify_factor
from skrobot.viewers import TrimeshSceneViewer

from skmp.constraint import CollFreeConst, ConfigPointConst, PoseConstraint
from skmp.robot.a0b import A0BConfig
from skmp.robot.utils import set_robot_state
from skmp.robot.robot import EndEffectorList, SurroundingList, Robot, RobotSurrounding
from skmp.satisfy import satisfy_by_optimization_with_budget
from skmp.solver.interface import Problem
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skmp.visualization.collision_visualizer import CollisionSphereVisualizationManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="visualize")
    parser.add_argument("--colvis", action="store_true", help="colvis")
    args = parser.parse_args()
    with_visualize = args.visualize
    with_colvis = args.colvis

    end_effector_list = EndEffectorList(
        link_names=["RARM_LINK5"],
        end_effector_names=["rarm_end_coords"],
        positions=[[0.25, 0, 0.0]],
        rpys=[[0, 0, 0]],
    )
    urdf_path = "/home/h-ishida/Downloads/a0b/A0B_original.urdf"
    with mesh_simplify_factor(0.3):
        model = Robot(urdf_path, end_effector_list)

    surrounding_list = SurroundingList(
        name=["pole", "table", "obstacle"],
        shape=["Box", "Box", "Box"],
        size=[[0.2, 0.2, 1.5], [2.0, 2.0, 0.1], [0.05, 0.05, 0.4]],
        position=[[-0.2, 0, -0.75], [0.0, 0.0, -0.5], [0.5, 0.15, -0.3]],
        rpy=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        color=[[120, 120, 120, 120], [120, 120, 0, 120], [120, 0, 0, 120]],
    )
    surrounding = RobotSurrounding(surrounding_list)
    # union signed distance function
    sdf = surrounding.get_sdf_list()

    conf = A0BConfig(urdf_path, end_effector_list)
    efkin = conf.get_endeffector_kin()
    colkin = conf.get_collision_kin()
    box_const = conf.get_box_const()

    # inequality constraint
    coll_free_const = CollFreeConst(colkin, sdf, model)

    # define target coords
    co_target = surrounding.table.copy_worldcoords()
    co_target.translate([0.25, 0.3, 0.1])
    co_target.rotate(np.pi * 0.5, "y")
    ax_target = Axis.from_coords(co_target)

    # equatlity constraint
    pose_const = PoseConstraint.from_skrobot_coords([co_target], efkin, model)

    # solve IK
    print("start solving IK")
    result = satisfy_by_optimization_with_budget(
        pose_const, box_const, coll_free_const, None, n_trial_budget=100
    )
    assert result.success
    print("time to solve ik: {}".format(result.elapsed_time))

    # setup path planning problem
    problem = Problem(
        np.zeros(6),
        box_const,
        ConfigPointConst(result.q),
        coll_free_const,
        None,
        motion_step_box_=0.05,
    )

    print("start solving path planning")
    solver = OMPLSolver.init(OMPLSolverConfig(n_max_call=10000, simplify=True))
    solver.setup(problem)
    rrt_result = solver.solve()
    assert rrt_result.traj is not None
    print("time to solve rrt: {}".format(rrt_result.time_elapsed))

    if with_visualize:
        vis = TrimeshSceneViewer()
        if with_colvis:
            colvis = CollisionSphereVisualizationManager(colkin, vis)
            colvis.update(model)

        ax = Axis.from_cascoords(model.rarm_end_coords)
        vis.add(model)
        vis.add(ax_target)
        object_list = surrounding.get_object_list()
        for obj in object_list:
            vis.add(obj)
        vis.show()
        time.sleep(1)

        for q in rrt_result.traj.resample(10):
            set_robot_state(model, conf.get_control_joint_names(), q)
            vis.redraw()
            if with_colvis:
                colvis.update(model)  # type: ignore
            time.sleep(1)

        input('Press any key to stop')
