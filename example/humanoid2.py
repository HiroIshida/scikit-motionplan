import time

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Box
from skrobot.utils.urdf import mesh_simplify_factor
from skrobot.viewers import TrimeshSceneViewer
from tinyfk import BaseType

from skmp.constraint import (
    CollFreeConst,
    ConfigPointConst,
    IneqCompositeConst,
    PoseConstraint,
)
from skmp.robot.jaxon import Jaxon, JaxonConfig
from skmp.robot.utils import set_robot_state
from skmp.satisfy import satisfy_by_optimization_with_budget
from skmp.solver.interface import Problem
from skmp.solver.myrrt_solver import MyRRTConfig, MyRRTConnectSolver
from skmp.solver.nlp_solver import SQPBasedSolver, SQPBasedSolverConfig

np.random.seed(5)

com_box = Box([0.25, 0.5, 5.0], with_sdf=True)
com_box.visual_mesh.visual.face_colors = [255, 0, 100, 100]

box = Box([1.0, 3.0, 0.1], with_sdf=True)
box.translate([0.8, 0.0, 1.1])
with mesh_simplify_factor(0.3):
    jaxon = Jaxon()
config = JaxonConfig()

# determine initial coordinate
start_coords_list = [
    Coordinates([0.0, -0.2, 0]),
    Coordinates([0.0, +0.2, 0]),
    Coordinates([0.7, -0.2, 1.5], rot=[0, -0.5 * np.pi, 0]),
    Coordinates([0.7, +0.2, 1.5], rot=[0, -0.5 * np.pi, 0]),
]
selcol_const = config.get_neural_selcol_const(jaxon)
colkin = config.get_collision_kin()
col_const = CollFreeConst(colkin, box.sdf, jaxon)
com_const = config.get_com_stability_const(jaxon, lambda x: -com_box.sdf(x))
ineq_const = IneqCompositeConst([com_const, col_const, selcol_const])

efkin = config.get_endeffector_kin()
eq_const_start = PoseConstraint.from_skrobot_coords(start_coords_list, efkin, jaxon)
bounds = config.get_box_const()
res_start = satisfy_by_optimization_with_budget(
    eq_const_start, bounds, ineq_const, None, n_trial_budget=300
)
assert res_start.success
print(res_start.q)

# setup for solve ik
goal_coords_list = [
    Coordinates([0.0, -0.2, 0]),
    Coordinates([0.0, +0.2, 0]),
    Coordinates([0.5, -0.6, 0.8], rot=[0, -0.5 * np.pi, 0]),
]
efkin_goal_ik = config.get_endeffector_kin(rarm=True, larm=False)
eq_const_goal = PoseConstraint.from_skrobot_coords(goal_coords_list, efkin_goal_ik, jaxon)

# setup for solve rrt
efkin_rrt = config.get_endeffector_kin(rarm=False, larm=False)
const_coords_list = [
    Coordinates([0.0, -0.2, 0]),
    Coordinates([0.0, +0.2, 0]),
]
eq_const_path_plan = PoseConstraint.from_skrobot_coords(const_coords_list, efkin_rrt, jaxon)

print("start solving IK")
ts = time.time()
res_goal = satisfy_by_optimization_with_budget(
    eq_const_goal, bounds, ineq_const, None, n_trial_budget=300
)
print(time.time() - ts)
assert res_goal.success

problem = Problem(
    res_start.q,
    bounds,
    ConfigPointConst(res_goal.q),
    ineq_const,
    eq_const_path_plan,
    motion_step_box_=0.1,
)
print("start solving rrt")
rrt = MyRRTConnectSolver.init(MyRRTConfig(10000))
rrt.setup(problem)
result = rrt.solve()
assert result.traj is not None
print(result)
print(time.time() - ts)

print("smooth out the result")
solver = SQPBasedSolver.init(
    SQPBasedSolverConfig(
        n_wp=60,
        n_max_call=200,
        motion_step_satisfaction="explicit",
        verbose=True,
        ctol_eq=1e-3,
        ctol_ineq=1e-3,
        ineq_tighten_coef=0.0,
    )
)
solver.setup(problem)
result = solver.solve(result.traj)
assert result.traj is not None
print(result.time_elapsed)

vis = TrimeshSceneViewer()
vis.add(box)
vis.add(jaxon)
vis.add(com_box)
vis.show()
time.sleep(4)
for q in result.traj.resample(20):
    set_robot_state(jaxon, config._get_control_joint_names(), q, base_type=BaseType.FLOATING)
    time.sleep(0.5)
    vis.redraw()
time.sleep(10)
