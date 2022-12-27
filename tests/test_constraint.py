from copy import deepcopy

import numpy as np
from skrobot.model.primitives import Box

from skmp.constraint import AbstractConst, CollFreeConst, ConfigPointConst
from skmp.robot.pr2 import PR2Config


def jac_numerical(const: AbstractConst, q0: np.ndarray) -> np.ndarray:
    f0, _ = const.evaluate_single(q0, with_jacobian=False)
    dim_domain = len(q0)
    dim_codomain = len(f0)

    jac = np.zeros((dim_codomain, dim_domain))
    eps = 1e-7
    for i in range(dim_domain):
        q1 = deepcopy(q0)
        q1[i] += eps
        f1, _ = const.evaluate_single(q1, with_jacobian=False)
        jac[:, i] = (f1 - f0) / eps
    return jac


def check_jacobian(const: AbstractConst):
    # check single jacobian
    for _ in range(10):
        q_test = np.random.randn(7)
        _, jac_anal = const.evaluate_single(q_test, with_jacobian=True)
        jac_numel = jac_numerical(const, q_test)
        np.testing.assert_almost_equal(jac_anal, jac_numel, decimal=4)

    # check traj jacobian
    for _ in range(10):
        qs_test = np.random.randn(10, 7)
        _, jac_anal = const.evaluate(qs_test, with_jacobian=True)
        jac_numel = np.array([jac_numerical(const, q) for q in qs_test])
        np.testing.assert_almost_equal(jac_anal, jac_numel, decimal=4)


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


def test_configpoint_const():
    const = ConfigPointConst(np.zeros(7))
    check_jacobian(const)


if __name__ == "__main__":
    test_box_const()
    test_collfree_const()
    test_configpoint_const()
