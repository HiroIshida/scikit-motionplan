import time

import numpy as np
from skrobot.model.primitives import Axis
from skrobot.viewers import TrimeshSceneViewer
from tinyfk import BaseType

from skmp.constraint import PoseConstraint
from skmp.robot.jaxon import Jaxon, JaxonConfig
from skmp.robot.utils import set_robot_state
from skmp.satisfy import satisfy_by_optimization

np.random.seed(0)

config = JaxonConfig()
efkin = config.get_endeffector_kin()
bounds = config.get_box_const()
jaxon = Jaxon()

coords_list = [
    np.array([0.0, -0.3, 0.0, 0, 0, 0]),
    np.array([0.0, +0.3, 0.0, 0, 0, 0]),
    np.array([0.5, -0.3, 0.0, 0, 0, 0]),
    np.array([0.5, +0.3, 0.0, 0, 0, 0]),
]

efkin = PoseConstraint(coords_list, efkin, jaxon)
res = satisfy_by_optimization(efkin, bounds, None, None)

set_robot_state(jaxon, config._get_control_joint_names(), res.q, base_type=BaseType.FLOATING)

ax1 = Axis.from_coords(jaxon.rarm_end_coords.copy_worldcoords())
ax2 = Axis.from_coords(jaxon.larm_end_coords.copy_worldcoords())
ax3 = Axis.from_coords(jaxon.rleg_end_coords.copy_worldcoords())
ax4 = Axis.from_coords(jaxon.lleg_end_coords.copy_worldcoords())

vis = TrimeshSceneViewer()
vis.add(jaxon)
vis.add(ax1)
vis.add(ax2)
vis.add(ax3)
vis.add(ax4)
vis.show()
time.sleep(100)
