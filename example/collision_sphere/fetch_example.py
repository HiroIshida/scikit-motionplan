import argparse
import time

from skrobot.models import Fetch
from skrobot.utils.urdf import mesh_simplify_factor
from skrobot.viewers import PyrenderViewer

from skmp.robot.fetch import FetchConfig
from skmp.visualization.collision_visualizer import CollisionSphereVisualizationManager

parser = argparse.ArgumentParser()
parser.add_argument("--init", action="store_true", help="init pose")
args = parser.parse_args()
init_pose: bool = args.init

with mesh_simplify_factor(0.3):
    conf = FetchConfig()
    colkin = conf.get_collision_kin()
    vis = PyrenderViewer()

    colvis = CollisionSphereVisualizationManager(colkin, vis)
    fetch = Fetch()
if init_pose:
    fetch.init_pose()
else:
    fetch.reset_pose()
fetch.r_gripper_finger_joint.joint_angle(0.1)
fetch.l_gripper_finger_joint.joint_angle(0.1)
colkin.reflect_skrobot_model(fetch)
colvis.update(fetch)

self_body_obstacles = conf.get_self_body_obstacles()

vis.add(fetch)
for obs in self_body_obstacles:
    vis.add(obs)
vis.show()

time.sleep(1000)
