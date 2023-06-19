import copy
import time
from pathlib import Path
from typing import List

import numpy as np
import rospy
from skrobot.interfaces.ros import PR2ROSRobotInterface
from skrobot.model.primitives import Axis
from skrobot.models.pr2 import PR2
from tinyfk import BaseType
from voxbloxpy import EsdfMap, GridSDF
from voxbloxpy.ros import EsdfNode, EsdfNodeConfig

from skmp.constraint import CollFreeConst, PoseConstraint
from skmp.robot.pr2 import PR2Config
from skmp.robot.utils import get_robot_state
from skmp.solver.interface import Problem
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig

# initialize
model = PR2(use_tight_joint_limit=False)
model.reset_manip_pose()
ri = PR2ROSRobotInterface(model)
ri.angle_vector(model.angle_vector(), time=1.0, time_scale=1.0)
ri.wait_interpolation()

topic = "/kinect_head/depth_registered/throttled/points"
world_frame = "/base_link"
config = EsdfNodeConfig(point_cloud_topic=topic, world_frame=world_frame, voxel_size=0.02)

grid_sdf_list: List[GridSDF] = []


def hook(esdf_map: EsdfMap):
    # this function might be too heavy.
    # I just do the following to make an animation. but do not do that
    # in the realtime application!
    ts = time.time()
    info = esdf_map.get_voxel_info()
    measure_grid = info.get_boundary_grid(grid_size=0.2)
    grid_sdf = esdf_map.get_grid_sdf(measure_grid, create_itp_lazy=True, fill_value=1.0)
    grid_sdf_list.append(grid_sdf)
    te = time.time()
    rospy.loginfo("elapsed time for getting gridsdf {} sec".format(te - ts))


node = EsdfNode(config, hook=hook)

time.sleep(5)  # stop node after a while
node.callback_running = False

fig_path = Path(__file__).resolve().parent / "figs"
fig_path.mkdir(exist_ok=True)
file_name = fig_path / "final_sdf.html"
fig = grid_sdf_list[-1].render_volume(isomin=-0.2, isomax=0.2, show=False)
fig.write_html(file_name)

# start planning
base_type = BaseType.FIXED
robot_config = PR2Config(base_type=base_type)
colkin = robot_config.get_collision_kin()
efkin = robot_config.get_endeffector_kin()
efkin.reflect_skrobot_model(model)
colkin.reflect_skrobot_model(model)
box_const = robot_config.get_box_const()

target = Axis(axis_radius=0.01, axis_length=0.05)
target.translate([0.9, -0.3, 0.85])
goal_eq_const = PoseConstraint.from_skrobot_coords([target], efkin, model)
collfree_const = CollFreeConst(colkin, grid_sdf_list[-1], model)

ompl_config = OMPLSolverConfig(n_max_call=10000, simplify=True)
ompl_solver = OMPLSolver.init(ompl_config)

q_start = get_robot_state(model, robot_config._get_control_joint_names(), base_type)
problem = Problem(q_start, box_const, goal_eq_const, collfree_const, None, motion_step_box_=0.05)
ompl_solver.setup(problem)
result = ompl_solver.solve()

q_init = model.angle_vector()
name_idx_table = {name: i for i, name in enumerate(model.joint_names)}
control_indices = np.array(
    [name_idx_table[name] for name in robot_config._get_control_joint_names()]
)


if result.traj is not None:
    print("planning success")
    print("trajectory shape: {}".format(result.traj.numpy().shape))

    for q_partial in result.traj.numpy():
        q_full = copy.deepcopy(q_init)
        q_full[control_indices] = q_partial
        ri.angle_vector(q_full, time_scale=1.0, time=1.0)
        ri.wait_interpolation()
