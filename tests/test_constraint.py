from copy import deepcopy

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Box
from skrobot.models import PR2

from skmp.constraint import (
    SELCOL_FOUND,
    AbstractConst,
    CollFreeConst,
    ConfigPointConst,
    IneqCompositeConst,
    PairWiseSelfCollFreeConst,
    PoseConstraint,
)
from skmp.robot.pr2 import PR2Config


def jac_numerical(const: AbstractConst, q0: np.ndarray, eps: float) -> np.ndarray:
    f0, _ = const.evaluate_single(q0, with_jacobian=False)
    dim_domain = len(q0)
    dim_codomain = len(f0)

    jac = np.zeros((dim_codomain, dim_domain))
    for i in range(dim_domain):
        q1 = deepcopy(q0)
        q1[i] += eps
        f1, _ = const.evaluate_single(q1, with_jacobian=False)
        jac[:, i] = (f1 - f0) / eps
    return jac


def check_jacobian(const: AbstractConst, eps: float = 1e-7, decimal: int = 4):
    # check single jacobian
    for _ in range(10):
        q_test = np.random.randn(7)
        _, jac_anal = const.evaluate_single(q_test, with_jacobian=True)
        jac_numel = jac_numerical(const, q_test, eps)
        np.testing.assert_almost_equal(jac_anal, jac_numel, decimal=decimal)

    # check traj jacobian
    for _ in range(10):
        qs_test = np.random.randn(10, 7)
        _, jac_anal = const.evaluate(qs_test, with_jacobian=True)
        jac_numel = np.array([jac_numerical(const, q, eps) for q in qs_test])
        np.testing.assert_almost_equal(jac_anal, jac_numel, decimal=decimal)


def test_box_const():
    config = PR2Config(with_base=False)
    box_const = config.get_box_const()
    check_jacobian(box_const)


def test_collfree_const():
    config = PR2Config(with_base=False)
    colkin = config.get_collision_kin()
    box = Box(extents=[0.7, 0.5, 1.2], with_sdf=True)
    box.translate(np.array([0.85, -0.2, 0.9]))
    assert box.sdf is not None
    collfree_const = CollFreeConst(colkin, box.sdf, 3)
    check_jacobian(collfree_const)


def test_neural_collfree_const():
    if not SELCOL_FOUND:
        return

    config = PR2Config(with_base=False)
    selcol_const = config.get_neural_selcol_const()
    pr2 = PR2()
    pr2.reset_manip_pose()
    selcol_const.reflect_skrobot_model(pr2)

    # NOTE: selcol model uses float32. So, larger eps is required
    check_jacobian(selcol_const, eps=1e-4, decimal=2)


def test_configpoint_const():
    const = ConfigPointConst(np.zeros(7))
    check_jacobian(const)


def test_pose_const():
    config = PR2Config(with_base=False)
    efkin = config.get_endeffector_kin()

    target = Coordinates(pos=[0.8, -0.6, 1.1])
    const = PoseConstraint.from_skrobot_coords([target], efkin)

    check_jacobian(const)


def test_pair_wise_selfcollfree_cost():
    config = PR2Config(with_base=False)
    colkin = config.get_collision_kin()
    const = PairWiseSelfCollFreeConst.from_colkin(colkin)
    check_jacobian(const)

    q_init = np.zeros(7)
    values, _ = const.evaluate_single(q_init, with_jacobian=False)
    assert np.all(values > 0)


def test_composite_constraint():
    config = PR2Config(with_base=False)

    colkin = config.get_collision_kin()
    box = Box(extents=[0.7, 0.5, 1.2], with_sdf=True)
    box.translate(np.array([0.85, -0.2, 0.9]))
    assert box.sdf is not None
    collfree_const = CollFreeConst(colkin, box.sdf, 3)

    selcol_const = config.get_neural_selcol_const()
    pr2 = PR2()
    pr2.reset_manip_pose()
    selcol_const.reflect_skrobot_model(pr2)

    IneqCompositeConst.composite([collfree_const, selcol_const])
    # NOTE: selcol model uses float32. So, larger eps is required
    check_jacobian(selcol_const, eps=1e-4, decimal=2)


if __name__ == "__main__":
    test_box_const()
    test_collfree_const()
    test_neural_collfree_const()
    test_configpoint_const()
    test_pose_const()
    test_composite_constraint()
