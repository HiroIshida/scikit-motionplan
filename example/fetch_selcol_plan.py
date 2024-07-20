import argparse
import time
from dataclasses import dataclass
from typing import List

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis, Box
from skrobot.models import Fetch
from skrobot.viewers import PyrenderViewer

from skmp.constraint import (
    CollFreeConst,
    ConfigPointConst,
    IneqCompositeConst,
    PoseConstraint,
)
from skmp.robot.fetch import FetchConfig
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.satisfy import satisfy_by_optimization_with_budget
from skmp.solver.interface import Problem
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skmp.utils import sksdf_to_cppsdf


@dataclass
class UnionSDF:
    sdfs: List

    def __call__(self, pts):
        sd_vals_list = [sdf(pts) for sdf in self.sdfs]
        sd_vals_union = np.min(np.array(sd_vals_list), axis=0)
        return sd_vals_union


parser = argparse.ArgumentParser()
parser.add_argument("--approx", action="store_true", help="approximate")
parser.add_argument("--visualize", action="store_true", help="visualize")
args = parser.parse_args()

np.random.seed(0)

conf = FetchConfig()
fetch = Fetch()
fetch.reset_pose()
q_init = get_robot_state(fetch, conf.get_control_joint_names())
efkin = conf.get_endeffector_kin()
box_const = conf.get_box_const()

# define self collision free constraint
self_body_obstacles = conf.get_self_body_obstacles()
colkin = conf.get_collision_kin()
sdf = UnionSDF([sksdf_to_cppsdf(obs.sdf) for obs in self_body_obstacles])
approx_selcol_free_const = CollFreeConst(colkin, sdf, fetch)
exact_selcol_free_const = conf.get_selcol_consts(fetch)  # gradient is not provided

# define collision free constraint
table = Box([1.0, 2.0, 0.05], with_sdf=True)
table.translate([1.0, 0.0, 0.8])
colkin = conf.get_collision_kin()
col_free_const = CollFreeConst(colkin, sksdf_to_cppsdf(table.sdf), fetch)

# composite ineq const
ineq_const = IneqCompositeConst([approx_selcol_free_const, col_free_const])

# solve collision free IK
goal_coords = Coordinates([0.8, 0.0, 0.95])
goal_const = PoseConstraint.from_skrobot_coords([goal_coords], efkin, fetch)
ik_result = satisfy_by_optimization_with_budget(goal_const, box_const, ineq_const, q_init)
assert ik_result.success
q_final = ik_result.q

# solve reaching problem
solver = OMPLSolver.init(OMPLSolverConfig(n_max_call=10000, simplify=True, algorithm_range=None))
ineq_const = IneqCompositeConst([exact_selcol_free_const, col_free_const])
problem = Problem(
    q_init, box_const, ConfigPointConst(q_final), ineq_const, None, motion_step_box_=0.1
)
solver.setup(problem)
res = solver.solve()

assert res.traj is not None
print("success!")

if args.visualize:
    v = PyrenderViewer()
    v.add(fetch)
    v.add(table)
    target_axis = Axis.from_coords(goal_coords)
    v.add(target_axis)
    v.show()
    time.sleep(1.0)
    for q in res.traj.resample(50):
        set_robot_state(fetch, conf.get_control_joint_names(), q)
        v.redraw()
        time.sleep(0.4)

    time.sleep(1000)
