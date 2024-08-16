import time

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis, Box
from skrobot.sdf import UnionSDF
from skrobot.viewers import PyrenderViewer
from tinyfk import BaseType, RotationType

from skmp.constraint import (
    CollFreeConst,
    ConfigPointConst,
    EqCompositeConst,
    FixedZAxisConstraint,
    IneqCompositeConst,
    PoseConstraint,
    RelativePoseConstraint,
)
from skmp.robot.jaxon import Jaxon, JaxonConfig
from skmp.robot.utils import set_robot_state
from skmp.satisfy import SatisfactionConfig, satisfy_by_optimization_with_budget
from skmp.solver.interface import Problem
from skmp.solver.myrrt_solver import MyRRTConfig, MyRRTConnectSolver
from skmp.solver.nlp_solver import SQPBasedSolver, SQPBasedSolverConfig

np.random.seed(0)

if __name__ == "__main__":
    # define obstacles
    ground = Box([2.0, 2.0, 0.03], with_sdf=True)
    ground.translate([0.0, 0.0, -0.015])

    table = Box([0.6, 1.0, 0.8], with_sdf=True)
    table.rotate(np.pi * 0.5, "z")
    table.translate([0.7, 0.0, 0.4])

    obstacles = [ground, table]

    rarm_target = table.copy_worldcoords()
    rarm_target.translate([0.0, -0.25, 0.66]).rotate(np.pi * 0.5, "z")
    larm_target = table.copy_worldcoords()
    larm_target.translate([0.0, +0.25, 0.66]).rotate(np.pi * 0.5, "z")
    axis_list = [Axis.from_coords(co) for co in [rarm_target, larm_target, table]]

    env_sdf = UnionSDF([obstacle.sdf for obstacle in obstacles])
    obstacles = [obstacle for obstacle in obstacles]

    com_box = Box([0.25, 0.5, 5.0], with_sdf=True)
    com_box.visual_mesh.visual.face_colors = [255, 0, 100, 100]

    box = Box([0.4, 0.4, 0.4], with_sdf=True, face_colors=[0, 0, 255, 230])
    box.translate([0.6, 0.0, 0.2])

    print("loading robot model may take a while...")
    jaxon = Jaxon()
    print("finish loading")
    config = JaxonConfig()

    # determine initial coordinate
    start_coords_list = [
        Coordinates([0.0, -0.2, 0]),
        Coordinates([0.0, +0.2, 0]),
        Coordinates([0.6, -0.25, 0.25]).rotate(+np.pi * 0.5, "z"),
        Coordinates([0.6, +0.25, 0.25]).rotate(+np.pi * 0.5, "z"),
    ]
    com_const_with_force = config.get_com_stability_const(
        jaxon, com_box, ["RARM_LINK7", "LARM_LINK7"], [5.0, 5.0]
    )
    colkin = config.get_collision_kin(rsole=False, lsole=False, rgripper=False, lgripper=False)
    colfree_const = CollFreeConst(colkin, env_sdf, jaxon, only_closest_feature=True)

    ineq_const = IneqCompositeConst([com_const_with_force, colfree_const])
    obstacle_kin = config.get_attached_obstacle_kin(np.array([0.25, 0.0, -0.04]), box)

    # solve ik to determine start state (random)
    print("determining q_init by solving IK")
    efkin = config.get_endeffector_kin()
    dualfoot_efkin = config.get_endeffector_kin(
        rleg=True, lleg=True, rarm=False, larm=False, rot_type=RotationType.RPY
    )
    dualarm_efkin = config.get_endeffector_kin(
        rleg=False, lleg=False, rarm=True, larm=True, rot_type=RotationType.XYZW
    )
    rarm_efkin = config.get_endeffector_kin(
        rleg=False, lleg=False, rarm=True, larm=False, rot_type=RotationType.RPY
    )
    fixaxis_const = FixedZAxisConstraint(rarm_efkin, jaxon)

    eq_const_start = PoseConstraint.from_skrobot_coords(start_coords_list, efkin, jaxon)
    bounds = config.get_box_const()
    ts = time.time()
    res_start = satisfy_by_optimization_with_budget(
        eq_const_start, bounds, ineq_const, None, n_trial_budget=300
    )
    print(time.time() - ts)
    set_robot_state(
        jaxon, config._get_control_joint_names(), res_start.q, base_type=BaseType.FLOATING
    )
    jaxon.rarm_end_coords.assoc(box)

    # solve second IK
    start_coords_list = [
        Coordinates([0.0, -0.2, 0]),
        Coordinates([0.0, +0.2, 0]),
        rarm_target,
        larm_target,
    ]

    eq_const_start = PoseConstraint.from_skrobot_coords(start_coords_list, efkin, jaxon)
    # eq_const_start = PoseConstraint.from_skrobot_coords(start_coords_list, efkin, jaxon)
    bounds = config.get_box_const()
    ts = time.time()
    res_goal = satisfy_by_optimization_with_budget(
        eq_const_start, bounds, ineq_const, res_start.q, n_trial_budget=300
    )
    assert res_goal.success

    solve_plan = True
    if not solve_plan:
        vis = PyrenderViewer()
        vis.add(ground)
        vis.add(table)
        set_robot_state(
            jaxon, config._get_control_joint_names(), res_goal.q, base_type=BaseType.FLOATING
        )
        vis.add(jaxon)
        vis.add(box)
        for ax in axis_list:
            vis.add(ax)
        vis.show()
        time.sleep(100)
    else:
        # solve planning problem
        diff = np.array([0.5, 0.0, 0.0])
        hand_relative_const = RelativePoseConstraint(diff, dualarm_efkin, jaxon)
        foot_contact_const = PoseConstraint.from_skrobot_coords(
            start_coords_list[:2], dualfoot_efkin, jaxon
        )

        box_collfree_const = CollFreeConst(obstacle_kin, env_sdf, jaxon, only_closest_feature=True)
        ineq_const_path_plan = IneqCompositeConst(
            [colfree_const, com_const_with_force, box_collfree_const]
        )
        eq_const_path_plan = EqCompositeConst(
            [hand_relative_const, foot_contact_const, fixaxis_const]
        )
        eq_const_goal = ConfigPointConst(res_goal.q)

        problem = Problem(
            res_start.q,
            bounds,
            eq_const_goal,
            ineq_const_path_plan,
            eq_const_path_plan,
            motion_step_box_=config.get_motion_step_box(),
        )
        rrt_conf = MyRRTConfig(10000, satisfaction_conf=SatisfactionConfig(n_max_eval=50))
        rrt = MyRRTConnectSolver.init(rrt_conf)
        rrt_parallel = rrt.as_parallel_solver(12)
        rrt_parallel.setup(problem)
        print("start solving rrt")
        from pyinstrument import Profiler

        profiler = Profiler()
        profiler.start()
        result = rrt.solve()
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True, show_all=True))
        assert result.traj is not None
        print(f"finish solving. elapsed {result.time_elapsed} sec")

        solver = SQPBasedSolver.init(
            SQPBasedSolverConfig(
                n_wp=50,
                n_max_call=200,
                motion_step_satisfaction="explicit",
                verbose=True,
                ctol_eq=1e-3,
                ctol_ineq=1e-3,
                ineq_tighten_coef=0.0,
            )
        )
        solver.setup(problem)
        from pyinstrument import Profiler

        profiler = Profiler()
        profiler.start()
        smooth_result = solver.solve(result.traj)
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True, show_all=True))
        print(f"solved? {smooth_result.traj is not None}")
        print(f"time to solve {smooth_result.time_elapsed} sec")

        vis = PyrenderViewer()
        # vis.add(ground)
        vis.add(table)
        vis.add(jaxon)
        vis.add(box)

        set_robot_state(
            jaxon, config._get_control_joint_names(), res_start.q, base_type=BaseType.FLOATING
        )

        vis.show()
        time.sleep(4)
        assert smooth_result.traj is not None
        for q in smooth_result.traj.resample(40):
            time.sleep(0.15)
            set_robot_state(
                jaxon, config._get_control_joint_names(), q, base_type=BaseType.FLOATING
            )
            vis.redraw()
        time.sleep(1000)
