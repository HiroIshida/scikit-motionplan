import time

import numpy as np
from skrobot.coordinates import Coordinates, rpy_matrix
from skrobot.viewers import TrimeshSceneViewer
from tinyfk import BaseType

from skmp.constraint import ConfigPointConst, PoseConstraint
from skmp.robot.jaxon import Jaxon, JaxonConfig
from skmp.robot.utils import set_robot_state
from skmp.satisfy import satisfy_by_optimization_with_budget
from skmp.solver.interface import Problem
from skmp.solver.nlp_solver import SQPBasedSolver, SQPBasedSolverConfig

np.random.seed(0)

config = JaxonConfig()
efkin = config.get_endeffector_kin(lleg=True, rleg=True)
bounds = config.get_box_const()
jaxon = Jaxon()
angle = -2.0
start_coords_list = [
    Coordinates([0, -0.15, 0]),
    Coordinates([0, +0.15, 0]),
    Coordinates([0, -0.15, 1.5], rot=rpy_matrix(0, angle, 0)),
    Coordinates([0, +0.15, 1.5], rot=rpy_matrix(0, angle, 0)),
]

goal_coords_list = [
    Coordinates([0, -0.15, 0.3]),
    Coordinates([0, +0.15, 0]),
    Coordinates([0, -0.15, 1.5], rot=rpy_matrix(0, angle, 0)),
    Coordinates([0, +0.15, 1.5], rot=rpy_matrix(0, angle, 0)),
]

selcol_const = config.get_neural_selcol_const(jaxon)

eq_const_start = PoseConstraint.from_skrobot_coords(start_coords_list, efkin, jaxon)
res_start = satisfy_by_optimization_with_budget(
    eq_const_start, bounds, selcol_const, None, n_trial_budget=100
)
assert res_start.success

eq_const_goal = PoseConstraint.from_skrobot_coords(goal_coords_list, efkin, jaxon)
res_goal = satisfy_by_optimization_with_budget(
    eq_const_goal, bounds, selcol_const, res_start.q, n_trial_budget=100
)
assert res_goal.success

const_coords_list = [
    Coordinates([0, +0.15, 0]),
    Coordinates([0, -0.15, 1.5], rot=rpy_matrix(0, angle, 0)),
    Coordinates([0, +0.15, 1.5], rot=rpy_matrix(0, angle, 0)),
]
efkin_swing = config.get_endeffector_kin(rleg=False)
eq_const_global = PoseConstraint.from_skrobot_coords(const_coords_list, efkin_swing, jaxon)
problem = Problem(
    res_start.q, bounds, ConfigPointConst(res_goal.q), None, eq_const_global, motion_step_box_=0.1
)

print("start solving")
# solver = OMPLSolver.init(
#     OMPLSolverConfig(
#         n_max_call=10000, simplify=True, algorithm=Algorithm.KPIECE1, algorithm_range=0.001
#     )
# )
# solver.setup(problem)
# res = solver.solve()
# assert res.traj is not None

sqp_solver = SQPBasedSolver.init(
    SQPBasedSolverConfig(
        30, n_max_call=100, motion_step_satisfaction="explicit", verbose=True, ctol_eq=1e-3
    )
)
sqp_solver.setup(problem)
sqp_res = sqp_solver.solve()
assert sqp_res.traj is not None

traj = sqp_res.traj.resample(30)

vis = TrimeshSceneViewer()
vis.add(jaxon)
vis.show()
time.sleep(15)

for q in traj:
    print(q)
    set_robot_state(jaxon, config._get_control_joint_names(), q, base_type=BaseType.FLOATING)
    vis.redraw()
    time.sleep(1)
time.sleep(20)
