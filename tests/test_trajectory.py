import pickle

import numpy as np
import tqdm
from skrobot.coordinates import Coordinates
from skrobot.models.pr2 import PR2
from tinyfk import BaseType, RotationType

from skmp.constraint import PoseConstraint
from skmp.robot.pr2 import PR2Config
from skmp.satisfy import satisfy_by_optimization
from skmp.trajectory import EndEffectorDistanceMetric, Trajectory


def test_trajectory_from_two_points():
    start = np.zeros(2)
    goal = np.ones(2)
    traj = Trajectory.from_two_points(start, goal, 10)
    np.testing.assert_almost_equal(start, traj[0])
    np.testing.assert_almost_equal(goal, traj[-1])


def test_trajectory1():
    start = np.zeros(2)
    goal = np.ones(2)
    traj = Trajectory.from_two_points(start, goal, 10)
    np.testing.assert_almost_equal(traj.get_length(), np.sqrt(2))

    np.testing.assert_almost_equal(traj.sample_point(0.0), traj[0])
    np.testing.assert_almost_equal(traj.sample_point(0.1), 0.1 * np.ones(2) / np.sqrt(2))
    np.testing.assert_almost_equal(traj.sample_point(0.8), 0.8 * np.ones(2) / np.sqrt(2))


def test_trajectory2():
    n = 1000
    angles = np.linspace(0, 2 * np.pi, n)
    xs = np.cos(angles)
    ys = np.sin(angles)
    X = list(np.vstack((xs, ys)).T)
    traj = Trajectory(X)
    np.testing.assert_almost_equal(traj.get_length(), 2 * np.pi, decimal=2)

    traj_coarse = traj.resample(100)
    np.testing.assert_almost_equal(traj_coarse.get_length(), 2 * np.pi, decimal=2)


def test_trajectory_serialization():
    points = list(np.random.randn(100, 2))
    traj = Trajectory(points)
    traj_again = Trajectory.loads(traj.dumps())
    assert pickle.dumps(traj) == pickle.dumps(traj_again)


def test_trajectory_with_custom_metric():
    config = PR2Config(base_type=BaseType.FLOATING)  # to solve ik easily
    efkin = config.get_endeffector_kin(RotationType.IGNORE)
    box_const = config.get_box_const()

    def draw_an_arc():
        n = 200
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
    traj_new = traj.get_metric_changed(efmetric)
    np.testing.assert_almost_equal(traj_new.get_length(), 2 * np.pi, decimal=2)
    traj_resampled = traj_new.resample(50)

    np.testing.assert_almost_equal(traj_resampled.get_length(), 2 * np.pi, decimal=1)

    # check if resampled trajecoty has almost regular interval wrt the custom metric
    for i in range(len(traj_resampled) - 1):
        q1 = traj_resampled[i]
        q2 = traj_resampled[i + 1]
        d = efmetric(q1, q2)
        np.testing.assert_almost_equal(d, 2 * np.pi / len(traj_resampled), decimal=1)


if __name__ == "__main__":
    test_trajectory_with_custom_metric()
