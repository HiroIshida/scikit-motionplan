import argparse
import time

import numpy as np
from skrobot.models import Fetch
from skrobot.sdf import UnionSDF
from skrobot.viewers import PyrenderViewer

from skmp.constraint import CollFreeConst, ConfigPointConst
from skmp.robot.fetch import FetchConfig
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.solver.interface import Problem
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig

parser = argparse.ArgumentParser()
parser.add_argument("--approx", action="store_true", help="approximate")
parser.add_argument("--visualize", action="store_true", help="visualize")
args = parser.parse_args()

np.random.seed(0)

conf = FetchConfig()
fetch = Fetch()
fetch.reset_pose()
q_init = get_robot_state(fetch, conf.get_control_joint_names())
fetch.init_pose()
q_final = get_robot_state(fetch, conf.get_control_joint_names())

if args.approx:
    obstacles = conf.get_self_body_obstacles()
    colkin = conf.get_collision_kin()
    sdf = UnionSDF([obs.sdf for obs in obstacles])
    ineq_const = CollFreeConst(colkin, sdf, fetch)
else:
    ineq_const = conf.get_selcol_consts(fetch)

box_const = conf.get_box_const()
solver = OMPLSolver.init(OMPLSolverConfig(n_max_call=10000, simplify=False, algorithm_range=None))
problem = Problem(
    q_final, box_const, ConfigPointConst(q_init), ineq_const, None, motion_step_box_=0.1
)
solver.setup(problem)
res = solver.solve()

assert res.traj is not None
print("success!")

if args.visualize:
    v = PyrenderViewer()
    v.add(fetch)
    v.show()
    time.sleep(1.0)
    for q in res.traj.resample(30):
        set_robot_state(fetch, conf.get_control_joint_names(), q)
        v.redraw()
        time.sleep(0.4)

    time.sleep(1000)
