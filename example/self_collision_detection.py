import time

import numpy as np
from selcol.traintime.dataset import get_random_robot_state
from skrobot.models import PR2
from skrobot.viewers import TrimeshSceneViewer

from skmp.constraint import PairWiseSelfCollFreeConst
from skmp.robot.pr2 import PR2Config

robot = PR2()

viewer = TrimeshSceneViewer()
viewer.add(robot)
viewer.show()

conf = PR2Config(with_base=False)
colkin = conf.get_collision_kin()

const = PairWiseSelfCollFreeConst(colkin, robot)

names = robot.joint_names
indices = [names.index(jn) for jn in colkin.control_joint_names]


while True:
    print("checking...")

    q = get_random_robot_state(robot)
    qp = q[indices]

    for name, angle in zip(colkin.control_joint_names, qp):
        robot.__dict__[name].joint_angle(angle)

    ts = time.time()
    val, _ = const.evaluate_single(qp, True)
    print(time.time() - ts)
    if np.any(val < 0):
        viewer.redraw()
        print("collide")
        time.sleep(1)
