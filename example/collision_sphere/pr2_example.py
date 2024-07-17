import time

from skrobot.models import PR2
from skrobot.viewers import TrimeshSceneViewer
from tinyfk import BaseType

from skmp.robot.pr2 import PR2Config
from skmp.visualization.collision_visualizer import CollisionSphereVisualizationManager

conf = PR2Config(control_arm="dual", base_type=BaseType.PLANER)
colkin = conf.get_collision_kin()
vis = TrimeshSceneViewer()

colvis = CollisionSphereVisualizationManager(colkin, vis)
pr2 = PR2()
pr2.reset_manip_pose()
colkin.reflect_skrobot_model(pr2)
colvis.update(pr2)
vis.add(pr2)
vis.show()

time.sleep(1000)
