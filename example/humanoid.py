import time

import numpy as np
from ompl import Algorithm
from skrobot.viewers import TrimeshSceneViewer
from tinyfk import BaseType

from skmp.constraint import ConfigPointConst, PoseConstraint
from skmp.robot.jaxon import Jaxon, JaxonConfig
from skmp.robot.utils import set_robot_state
from skmp.satisfy import satisfy_by_optimization
from skmp.solver.interface import Problem
from skmp.solver.nlp_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig

np.random.seed(0)

config = JaxonConfig()
efkin = config.get_endeffector_kin()
bounds = config.get_box_const()
jaxon = Jaxon()

start_coords_list = [
    np.array([0.0, -0.3, 0.0, 0, 0, 0]),
    np.array([0.0, +0.3, 0.0, 0, 0, 0]),
    np.array([0.6, -0.3, 0.0, 0, 0, 0]),
    np.array([0.6, +0.3, 0.0, 0, 0, 0]),
]

goal_coords_list = [
    np.array([0.0, -0.3, 0.0, 0, 0, 0]),
    np.array([0.0, +0.3, 0.0, 0, 0, 0]),
    np.array([0.9, -0.3, 0.0, 0, 0, 0]),
    np.array([0.6, +0.3, 0.0, 0, 0, 0]),
]

eq_const = PoseConstraint(start_coords_list, efkin, jaxon)
res_start = satisfy_by_optimization(eq_const, bounds, None, None)
assert res_start.success

eq_const = PoseConstraint(goal_coords_list, efkin, jaxon)
res_goal = satisfy_by_optimization(eq_const, bounds, None, None)
assert res_goal.success

const_coords_list = [
    np.array([0.0, -0.3, 0.0, 0, 0, 0]),
    np.array([0.0, +0.3, 0.0, 0, 0, 0]),
    np.array([0.6, +0.3, 0.0, 0, 0, 0]),
]
efkin_swing = config.get_endeffector_kin(rarm=False)
eq_const_global = PoseConstraint(const_coords_list, efkin_swing, jaxon)

print("start solving")
problem = Problem(res_start.q, bounds, ConfigPointConst(res_goal.q), None, eq_const_global)
solver = OMPLSolver.init(
    OMPLSolverConfig(
        n_max_call=10000, simplify=True, algorithm=Algorithm.KPIECE1, algorithm_range=0.1
    )
)
solver.setup(problem)
res = solver.solve()
assert res.traj is not None

sqp_solver = SQPBasedSolver.init(
    SQPBasedSolverConfig(
        30, n_max_call=100, motion_step_satisfaction="explicit", verbose=True, ctol_eq=1e-3
    )
)
sqp_solver.setup(problem)
sqp_res = sqp_solver.solve(res.traj)
assert sqp_res.traj is not None

traj = sqp_res.traj.resample(30)

vis = TrimeshSceneViewer()
vis.add(jaxon)
vis.show()
time.sleep(15)

for q in traj:
    set_robot_state(jaxon, config._get_control_joint_names(), q, base_type=BaseType.FLOATING)
    vis.redraw()
    time.sleep(1)
time.sleep(20)
