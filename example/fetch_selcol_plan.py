import time

from skrobot.models import Fetch
from skrobot.viewers import PyrenderViewer

from skmp.constraint import ConfigPointConst
from skmp.robot.fetch import FetchConfig
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.solver.interface import Problem
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig

conf = FetchConfig()
fetch = Fetch()
fetch.reset_pose()
q_init = get_robot_state(fetch, conf.joint_names)
fetch.init_pose()
q_final = get_robot_state(fetch, conf.joint_names)

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

v = PyrenderViewer()
v.add(fetch)
v.show()
time.sleep(1.0)
for q in res.traj.resample(30):
    set_robot_state(fetch, conf.joint_names, q)
    v.redraw()
    time.sleep(0.4)

time.sleep(1000)
