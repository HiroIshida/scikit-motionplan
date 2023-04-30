import time
from typing import Optional

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Box
from skrobot.models import PR2
from skrobot.viewers import TrimeshSceneViewer
from tinyfk import BaseType

from skmp.constraint import (
    CollFreeConst,
    ConfigPointConst,
    PoseConstraint,
    RelativePoseConstraint,
)
from skmp.robot.pr2 import PR2Config
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.satisfy import SatisfactionResult, satisfy_by_optimization


def test_satisfy_by_optimization_pose_const():
    pr2 = PR2()
    pr2.reset_manip_pose()
    config = PR2Config(base_type=BaseType.FIXED)
    colkin = config.get_collision_kin()
    efkin = config.get_endeffector_kin()

    box = Box(extents=[0.7, 0.5, 1.2], with_sdf=True)
    box.translate(np.array([0.85, -0.2, 0.9]))
    assert box.sdf is not None
    ineq_const = CollFreeConst(colkin, box.sdf, pr2)
    box_const = config.get_box_const()

    eq_const1 = ConfigPointConst(np.array([-0.78, 0.055, -1.37, -0.59, -0.494, -0.20, 1.87]))
    eq_const1.reflect_skrobot_model(pr2)

    co_list = [Coordinates(pos=[0.7, -0.6, 1.0])]
    eq_const2 = PoseConstraint.from_skrobot_coords(co_list, efkin, pr2)
    eq_const2.reflect_skrobot_model(pr2)

    for eq_const in [eq_const1, eq_const2]:
        result: Optional[SatisfactionResult] = None
        for _ in range(30):
            result = satisfy_by_optimization(eq_const, box_const, ineq_const, None)
            if result.success:
                break
        assert result is not None
        eq_val, _ = eq_const.evaluate_single(result.q, False)
        ineq_val, _ = ineq_const.evaluate_single(result.q, False)
        assert eq_val.dot(eq_val) < 1e-6
        assert np.all(ineq_val > 0)


def test_satisfy_by_optimization_relative_pose_const(debug: bool = False):
    pr2 = PR2(use_tight_joint_limit=False)
    pr2.reset_manip_pose()
    config = PR2Config(base_type=BaseType.FIXED, control_arm="dual")
    efkin = config.get_endeffector_kin()
    box_const = config.get_box_const()

    relconst = RelativePoseConstraint(np.array([0.0, 0.0, -0.2]), efkin, pr2)
    result: Optional[SatisfactionResult] = None
    for _ in range(30):
        result = satisfy_by_optimization(relconst, box_const, None, None)
        if result.success:
            break
    assert result is not None
    assert result.success

    if debug:
        set_robot_state(pr2, efkin.control_joint_names, result.q)
        viewer = TrimeshSceneViewer()
        viewer.add(pr2)
        viewer.show()
        time.sleep(10)


def test_satisfy_by_optimization_without_eq_const():
    pr2 = PR2()
    pr2.reset_manip_pose()
    config = PR2Config(base_type=BaseType.FIXED)
    colkin = config.get_collision_kin()
    config.get_endeffector_kin()

    box = Box(extents=[0.7, 0.5, 1.2], with_sdf=True)
    box.translate(np.array([0.68, -0.3, 0.9]))
    assert box.sdf is not None
    ineq_const = CollFreeConst(colkin, box.sdf, pr2)
    box_const = config.get_box_const()

    q_now = get_robot_state(pr2, config._get_control_joint_names())

    # ensure that robot collides with the obstacle at initial state
    assert not ineq_const.is_valid(q_now)

    result = satisfy_by_optimization(None, box_const, ineq_const, q_now)
    assert result.success

    assert ineq_const.is_valid(result.q)


if __name__ == "__main__":
    test_satisfy_by_optimization_without_eq_const()
