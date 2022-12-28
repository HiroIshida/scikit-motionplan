import numpy as np

from skmp.solver.motion_step_box import interpolate_fractions


def test_interpolation_fraction():
    box = np.array([0.1, 0.2])
    fractions = np.array(interpolate_fractions(box, np.zeros(2), np.array([0.0, 1.0]), False))
    fractions_gt = np.linspace(0, 1.0, 6)[1:]
    np.testing.assert_almost_equal(fractions, fractions_gt)

    fractions = np.array(interpolate_fractions(box, np.zeros(2), np.array([0.0, 1.0]), True))
    fractions_gt = np.linspace(0, 1.0, 6)
    np.testing.assert_almost_equal(fractions, fractions_gt)

    fractions = np.array(interpolate_fractions(box, np.zeros(2), np.ones(2), False))
    fractions_gt = np.linspace(0, 1, 11)[1:]
    np.testing.assert_almost_equal(fractions, fractions_gt)

    # test non edge case
    fractions = np.array(interpolate_fractions(box, np.zeros(2), np.ones(2) * 1.1, False))
    fractions_gt = np.hstack((np.linspace(0, 1, 11)[1:], 1.1)) / 1.1
    np.testing.assert_almost_equal(fractions, fractions_gt)
