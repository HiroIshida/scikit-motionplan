import time

import numpy as np
from skrobot.coordinates import Coordinates, rpy_matrix
from skrobot.model.primitives import Box
from skrobot.sdf import UnionSDF
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
from skmp.visualization import CollisionSphereVisualizationManager

np.random.seed(6)

# create obstacle world
boxes = []
for i in range(15):
    box = Box([0.1, 1.0, 0.05], with_sdf=True)
    box.translate([0.12, 0.0, -0.9 + 0.3 * i])
    # box.translate([0.3, 0.0, -0.9 + 0.3 * i])
    boxes.append(box)
sdf = UnionSDF([box.sdf for box in boxes])

with mesh_simplify_factor(1.0):
    config = JaxonConfig()
    colkin_nosole = config.get_collision_kin(rsole=True, lsole=False)
    efkin = config.get_endeffector_kin(lleg=True, rleg=True)
    bounds = config.get_box_const()
    jaxon = Jaxon()

eps = 0.06
angle = -1.54
start_coords_list = [
    Coordinates([0.0, -0.2, eps]),
    Coordinates([0.0, +0.2, 0.0]),
    Coordinates([0.0, -0.3, 1.5], rot=rpy_matrix(0, angle, 0)),
    Coordinates([0.0, +0.3, 1.5], rot=rpy_matrix(0, angle, 0)),
]

goal_coords_list = [
    Coordinates([0.0, -0.2, 0.3 + eps]),
    Coordinates([0.0, +0.2, 0.0]),
    Coordinates([0.0, -0.3, 1.5], rot=rpy_matrix(0, angle, 0)),
    Coordinates([0.0, +0.3, 1.5], rot=rpy_matrix(0, angle, 0)),
]

with mesh_simplify_factor(0.2):
    selcol_const = config.get_neural_selcol_const(jaxon)
    col_const = CollFreeConst(colkin_nosole, sdf, jaxon)
    ineq_const = IneqCompositeConst([col_const, selcol_const])

    eq_const_start = PoseConstraint.from_skrobot_coords(start_coords_list, efkin, jaxon)
res_start = satisfy_by_optimization_with_budget(
    eq_const_start, bounds, ineq_const, None, n_trial_budget=300
)
assert res_start.success
print(res_start)

with mesh_simplify_factor(0.2):
    eq_const_goal = PoseConstraint.from_skrobot_coords(goal_coords_list, efkin, jaxon)
res_goal = satisfy_by_optimization_with_budget(
    eq_const_goal, bounds, ineq_const, res_start.q, n_trial_budget=300
)
assert res_goal.success
print(res_goal)

const_coords_list = [
    Coordinates([0.0, +0.2, 0.0]),
    Coordinates([0.0, -0.3, 1.5], rot=rpy_matrix(0, angle, 0)),
    Coordinates([0.0, +0.3, 1.5], rot=rpy_matrix(0, angle, 0)),
]
with mesh_simplify_factor(0.2):
    efkin_swing = config.get_endeffector_kin(rleg=False)
    eq_const_global = PoseConstraint.from_skrobot_coords(const_coords_list, efkin_swing, jaxon)

problem = Problem(
    res_start.q,
    bounds,
    ConfigPointConst(res_goal.q),
    ineq_const,
    eq_const_global,
    motion_step_box_=0.1,
)

print("start solver rrt")
conf = MyRRTConfig(2000)
solver = MyRRTConnectSolver.init(conf)
solver.setup(problem)
res = solver.solve()
print(res)
assert res.traj is not None
traj = res.traj.resample(15)

vis = TrimeshSceneViewer()
vis.add(jaxon)
colvis = CollisionSphereVisualizationManager(colkin_nosole, vis)
for box in boxes:
    vis.add(box)
vis.show()
time.sleep(15)

for q in traj:
    print(q)
    set_robot_state(jaxon, config._get_control_joint_names(), q, base_type=BaseType.FLOATING)
    colvis.update(jaxon)
    vis.redraw()
    time.sleep(0.5)
time.sleep(20)
