import time

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Box
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

# np.random.seed(5)

box = Box([1.0, 3.0, 0.1], with_sdf=True)
box.translate([0.6, 0.0, 1.1])
jaxon = Jaxon()
config = JaxonConfig()

# determine initial coordinate
start_coords_list = [
    Coordinates([0.0, -0.3, 0]),
    Coordinates([0.0, +0.3, 0]),
    Coordinates([0.3, -0.2, 1.3], rot=[0, -0.5 * np.pi, 0]),
    Coordinates([0.3, +0.3, 1.3], rot=[0, -0.5 * np.pi, 0]),
]
selcol_const = config.get_neural_selcol_const(jaxon)
colkin = config.get_collision_kin()
col_const = CollFreeConst(colkin, box.sdf, jaxon)
ineq_const = IneqCompositeConst([col_const, selcol_const])

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
    Coordinates([0.0, -0.3, 0]),
    Coordinates([0.0, +0.3, 0]),
    Coordinates([0.4, -0.1, 0.9], rot=[0, -0.5 * np.pi, 0]),
    Coordinates([0.4, +0.4, 0.9], rot=[0, -0.5 * np.pi, 0]),
]
efkin = config.get_endeffector_kin(rarm=True, larm=True)
eq_const_start = PoseConstraint.from_skrobot_coords(goal_coords_list, efkin, jaxon)
selcol_const = config.get_neural_selcol_const(jaxon)
colkin = config.get_collision_kin()
col_const = CollFreeConst(colkin, box.sdf, jaxon)
ineq_const = IneqCompositeConst([col_const, selcol_const])
restricted_bounds = config.get_close_box_const(
    res_start.q, base_pos_margin=0.5, base_rot_margin=0.8
)

# setup for solve rrt
efkin = config.get_endeffector_kin(rarm=False, larm=False)
const_coords_list = [
    Coordinates([0.0, -0.3, 0]),
    Coordinates([0.0, +0.3, 0]),
]
eq_const_path_plan = PoseConstraint.from_skrobot_coords(const_coords_list, efkin, jaxon)

print("start solving IK")
ts = time.time()
res_goal = satisfy_by_optimization_with_budget(
    eq_const_start, restricted_bounds, ineq_const, None, n_trial_budget=300
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
rrt = MyRRTConnectSolver.init(MyRRTConfig(2000))
rrt.setup(problem)
result = rrt.solve()
assert result.traj is not None
print(result)
print(time.time() - ts)

vis = TrimeshSceneViewer()
vis.add(box)
vis.add(jaxon)
vis.show()
time.sleep(10)
for q in result.traj.resample(30):
    set_robot_state(jaxon, config._get_control_joint_names(), q, base_type=BaseType.FLOATING)
    time.sleep(0.5)
    vis.redraw()
