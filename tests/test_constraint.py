from copy import deepcopy

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Box
from skrobot.models import PR2
from skrobot.utils.urdf import mesh_simplify_factor
from tinyfk import BaseType, RotationType

from skmp.constraint import (
    AbstractConst,
    CollFreeConst,
    ConfigPointConst,
    IneqCompositeConst,
    PairWiseSelfCollFreeConst,
    PoseConstraint,
    ReducedCollisionFreeConst,
    RelativePoseConstraint,
)
from skmp.robot.jaxon import Jaxon, JaxonConfig
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


def check_jacobian(
    const: AbstractConst, dim: int, eps: float = 1e-7, decimal: int = 4, std: float = 1.0
):
    # check single jacobian
    for _ in range(10):
        q_test = np.random.randn(dim) * std
        _, jac_anal = const.evaluate_single(q_test, with_jacobian=True)
        jac_numel = jac_numerical(const, q_test, eps)
        np.testing.assert_almost_equal(jac_anal, jac_numel, decimal=decimal)

    # check traj jacobian
    for _ in range(10):
        qs_test = np.random.randn(10, dim) * std
        _, jac_anal = const.evaluate(qs_test, with_jacobian=True)
        jac_numel = np.array([jac_numerical(const, q, eps) for q in qs_test])
        np.testing.assert_almost_equal(jac_anal, jac_numel, decimal=decimal)


def test_box_const():
    config = PR2Config(base_type=BaseType.FIXED)
    box_const = config.get_box_const()
    check_jacobian(box_const, 7)

    # check if id_value is assigned
    assert isinstance(box_const.id_value, str)


def test_collfree_const():
    config = PR2Config(base_type=BaseType.FIXED)
    colkin = config.get_collision_kin()
    box = Box(extents=[0.7, 0.5, 1.2], with_sdf=True)
    box.translate(np.array([0.85, -0.2, 0.9]))
    assert box.sdf is not None
    collfree_const = CollFreeConst(colkin, box.sdf, PR2(), only_closest_feature=False)
    collfree_const.only_closest_feature = False
    check_jacobian(collfree_const, 7)

    collfree_const_oc = CollFreeConst(colkin, box.sdf, PR2(), only_closest_feature=True)
    check_jacobian(collfree_const_oc, 7)

    for _ in range(20):
        q = np.random.randn(7)
        values = collfree_const.evaluate_single(q, False)[0]
        closest_value = collfree_const_oc.evaluate_single(q, False)[0][0]
        assert np.min(values) == closest_value


def test_reduced_collfree_const():
    config = PR2Config(base_type=BaseType.FIXED)
    colkin = config.get_collision_kin()
    box = Box(extents=[0.7, 0.5, 1.2], with_sdf=True)
    box.translate(np.array([0.85, -0.2, 0.9]))
    assert box.sdf is not None
    collfree_const = ReducedCollisionFreeConst(colkin, box.sdf, PR2())
    check_jacobian(collfree_const, 7)

    # check if id_value is assigned
    assert isinstance(collfree_const.id_value, str)


def test_neural_collfree_const():
    pr2 = PR2()
    pr2.reset_manip_pose()

    config = PR2Config(base_type=BaseType.FIXED)
    selcol_const = config.get_neural_selcol_const(pr2)

    # NOTE: selcol model uses float32. So, larger eps is required
    check_jacobian(selcol_const, 7, eps=1e-4, decimal=2)

    # test with base
    config = PR2Config(base_type=BaseType.PLANER)
    selcol_const = config.get_neural_selcol_const(pr2)
    check_jacobian(selcol_const, 10, eps=1e-4, decimal=2)

    # check if id_value is assigned
    assert isinstance(selcol_const.id_value, str)


def test_configpoint_const():
    const = ConfigPointConst(np.zeros(7))
    check_jacobian(const, 7)

    # check if id_value is assigned
    assert isinstance(const.id_value, str)


def test_pose_const():
    config = PR2Config(base_type=BaseType.FIXED)
    efkin = config.get_endeffector_kin()

    rot_types = [RotationType.IGNORE, RotationType.RPY, RotationType.XYZW]
    dofs = [3, 6, 7]

    for rot_type, dof in zip(rot_types, dofs):
        efkin.update_rotation_type(rot_type)

        target = Coordinates(pos=[0.8, -0.6, 1.1])
        const = PoseConstraint.from_skrobot_coords([target], efkin, PR2())

        check_jacobian(const, 7, decimal=3)

        q = np.random.randn(len(config._get_control_joint_names()))
        val, jac = const.evaluate_single(q, True)
        assert val.shape == (dof,)
        assert jac.shape == (dof, len(q))

        # check if id_value is assigned
        assert isinstance(const.id_value, str)


def test_realtive_pose_const():
    config = PR2Config(base_type=BaseType.FIXED, control_arm="dual")
    efkin = config.get_endeffector_kin()
    relconst = RelativePoseConstraint(np.ones(3) * 0.1, efkin, PR2())

    check_jacobian(relconst, 14)

    # inside relconst constructor, efkin is copied and
    # modified to add a new feature point.
    # this test that, efkin is properly copied and the
    # original one does not change
    assert efkin.n_feature == 2
    assert len(efkin.tinyfk_feature_ids) == 2

    # check if id_value is assigned
    assert isinstance(relconst.id_value, str)


def test_pair_wise_selfcollfree_const():
    config = PR2Config(base_type=BaseType.FIXED)
    colkin = config.get_collision_kin()

    const = PairWiseSelfCollFreeConst(colkin, PR2(), only_closest_feature=False)
    check_jacobian(const, 7)
    q_init = np.zeros(7)
    values, _ = const.evaluate_single(q_init, with_jacobian=False)
    assert np.all(values > 0)
    assert isinstance(const.id_value, str)

    const_only_closest = PairWiseSelfCollFreeConst(colkin, PR2(), only_closest_feature=True)
    check_jacobian(const_only_closest, 7)
    q_init = np.zeros(7)
    values, _ = const_only_closest.evaluate_single(q_init, with_jacobian=False)
    assert np.all(values > 0)
    assert isinstance(const_only_closest.id_value, str)

    # check consistency
    for _ in range(10):
        q = np.random.randn(7)
        values, _ = const.evaluate_single(q, with_jacobian=False)
        closest_value, _ = const_only_closest.evaluate_single(q, with_jacobian=False)
        np.testing.assert_almost_equal(np.min(values), closest_value[0])


def test_com_stability_const():
    config = JaxonConfig()
    with mesh_simplify_factor(0.2):
        jaxon = Jaxon()

    com_box = Box([0.2, 0.6, 5.0], with_sdf=True)
    const = config.get_com_stability_const(jaxon, com_box)
    # NOTE: currently base jacobian computation is unstable when
    # rpy angle is big due to singularity
    # thus we set std = 0.1
    check_jacobian(const, len(config._get_control_joint_names()) + 6, std=0.1)

    # check if id_value is assigned
    assert isinstance(const.id_value, str)


def test_composite_constraint():
    pr2 = PR2()
    pr2.reset_manip_pose()

    config = PR2Config(base_type=BaseType.FIXED)

    colkin = config.get_collision_kin()
    box = Box(extents=[0.7, 0.5, 1.2], with_sdf=True)
    box.translate(np.array([0.85, -0.2, 0.9]))
    assert box.sdf is not None
    collfree_const = CollFreeConst(colkin, box.sdf, pr2)
    selcol_const = PairWiseSelfCollFreeConst(colkin, PR2())

    composite_const = IneqCompositeConst([collfree_const, selcol_const])
    # NOTE: selcol model uses float32. So, larger eps is required
    check_jacobian(composite_const, 7, eps=1e-4, decimal=2)

    # check if id_value is assigned
    assert isinstance(composite_const.id_value, str)

    # check if composite const is properly reduced
    composite_const = IneqCompositeConst([collfree_const, selcol_const, collfree_const])
    assert len(composite_const.const_list) == 2
    assert composite_const.const_list[0].id_value == collfree_const.id_value
    assert composite_const.const_list[1].id_value == selcol_const.id_value


if __name__ == "__main__":
    # test_box_const()
    # test_collfree_const()
    # test_neural_collfree_const()
    # test_configpoint_const()
    # test_pose_const()
    # test_composite_constraint()
    # test_realtive_pose_const()
    test_com_stability_const()
