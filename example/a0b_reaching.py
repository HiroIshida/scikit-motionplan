import time

import numpy as np
from skrobot.model.primitives import Axis, Box
from skrobot.sdf import UnionSDF
from skrobot.utils.urdf import mesh_simplify_factor
from skrobot.viewers import TrimeshSceneViewer

from skmp.constraint import CollFreeConst, ConfigPointConst, PoseConstraint
from skmp.robot.a0b import A0B, A0BConfig, A0BSurrounding
from skmp.robot.utils import set_robot_state
from skmp.satisfy import satisfy_by_optimization_with_budget
from skmp.solver.interface import Problem
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skmp.visualization import CollisionSphereVisualizationManager

with_colvis = True

urdf_path = "/home/h-ishida/Downloads/a0b/A0B_original.urdf"
with mesh_simplify_factor(0.3):
    model = A0B(urdf_path)

surrounding = A0BSurrounding()

conf = A0BConfig(urdf_path)
efkin = conf.get_endeffector_kin()
colkin = conf.get_collision_kin()
box_const = conf.get_box_const()

vis = TrimeshSceneViewer()

# define target coords
co_target = surrounding.table.copy_worldcoords()
co_target.translate([0.25, 0.3, 0.1])
co_target.rotate(np.pi * 0.5, "y")
ax_target = Axis.from_coords(co_target)

# define custom obstacle
obstacle = Box([0.05, 0.05, 0.4], with_sdf=True)
obstacle.translate([0.3, 0.15, -0.3])

# union signed distance function
sdf = UnionSDF([surrounding.pole.sdf, surrounding.table.sdf, obstacle.sdf])
# sdf = UnionSDF([surrounding.pole.sdf, surrounding.table.sdf])

# inequality constraint
coll_free_const = CollFreeConst(colkin, sdf, model)

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

if with_colvis:
    colvis = CollisionSphereVisualizationManager(colkin, vis)
    colvis.update(model)

ax = Axis.from_cascoords(model.rarm_end_coords)
vis.add(model)
vis.add(ax_target)
vis.add(surrounding.table)
vis.add(surrounding.pole)
vis.add(obstacle)
vis.show()
time.sleep(3)

for q in rrt_result.traj.resample(10):
    set_robot_state(model, conf._get_control_joint_names(), q)
    vis.redraw()
    colvis.update(model)
    time.sleep(1)

time.sleep(100)
