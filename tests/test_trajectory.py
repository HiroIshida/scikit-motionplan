import numpy as np
import tqdm
from skrobot.coordinates import Coordinates
from skrobot.models.pr2 import PR2
from tinyfk import BaseType, RotationType

from skmp.constraint import PoseConstraint
from skmp.robot.pr2 import PR2Config
from skmp.satisfy import satisfy_by_optimization
from skmp.trajectory import EndEffectorDistanceMetric, Trajectory


def test_trajectory():
    n = 1000
    angles = np.linspace(0, 2 * np.pi, n)
    xs = np.cos(angles)
    ys = np.sin(angles)
    X = list(np.vstack((xs, ys)).T)
    traj = Trajectory(X)
    np.testing.assert_almost_equal(traj.get_length(), 2 * np.pi, decimal=2)

    traj_coarse = traj.resample(100)
    np.testing.assert_almost_equal(traj_coarse.get_length(), 2 * np.pi, decimal=2)


def test_trajectory_with_custom_metric():
    config = PR2Config(base_type=BaseType.FLOATING)  # to solve ik easily
    efkin = config.get_endeffector_kin(RotationType.IGNORE)
    box_const = config.get_box_const()

    def draw_an_arc():
        n = 100
        angles = np.linspace(0, 2 * np.pi, n)
        xs = np.cos(angles)
        ys = np.sin(angles)
        zs = np.ones(n)
        X = np.vstack((xs, ys, zs)).T
        pr2 = PR2()
        q_seed = None
        q_list = []
        for x in tqdm.tqdm(X):
            co = Coordinates(pos=x)
            const = PoseConstraint.from_skrobot_coords([co], efkin, pr2)
            ret = satisfy_by_optimization(const, box_const, None, q_seed)
            q_list.append(ret.q)
            if not ret.success:
                return None
            q_seed = ret.q
        traj = Trajectory(q_list)
        return traj

    for _ in range(10):
        traj = draw_an_arc()  # this may fail
        if traj is not None:
            break
    assert isinstance(traj, Trajectory)
    efmetric = EndEffectorDistanceMetric(efkin)
    L = traj.get_length(efmetric)
    np.testing.assert_almost_equal(L, 2 * np.pi, decimal=2)

    traj_coarse = traj.resample(30, efmetric)
    np.testing.assert_almost_equal(traj_coarse.get_length(efmetric), 2 * np.pi, decimal=1)


if __name__ == "__main__":
    test_trajectory_with_custom_metric()
