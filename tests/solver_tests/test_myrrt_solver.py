from typing import Optional

import numpy as np

from skmp.solver._manifold_rrt_solver import (
    ManifoldRRT,
    ManifoldRRTConfig,
    ManifoldRRTConnect,
)


def example_project(q: np.ndarray, coll_aware: bool) -> Optional[np.ndarray]:
    return q / np.linalg.norm(q)


def example_is_valid(q: np.ndarray) -> bool:
    if abs(q[0]) > 0.2:
        return True
    if abs(q[1]) < 0.2:
        return True
    return False


def test_manifold_rrt():
    start = np.array([-1, 0, 0])
    b_min = -np.ones(3) * 1.5
    b_max = +np.ones(3) * 1.5
    motion_step_box = np.ones(3) * 0.2
    conf = ManifoldRRTConfig(5000)

    def f_goal_project(q):
        goal = np.array([+1, 0, 0])
        return goal

    bitree = ManifoldRRT(
        start,
        f_goal_project,
        b_min,
        b_max,
        motion_step_box,
        example_project,
        example_is_valid,
        config=conf,
    )
    is_success = bitree.solve()
    assert is_success
    traj = bitree.get_solution()

    np.testing.assert_almost_equal(traj[0], start)
    np.testing.assert_almost_equal(traj[-1], np.array([+1, 0, 0]))

    for q in traj:
        assert example_is_valid(q)


def test_manifold_rrt_connect():
    start = np.array([-1, 0, 0])
    goal = np.array([+1, 0, 0])
    b_min = -np.ones(3) * 1.5
    b_max = +np.ones(3) * 1.5
    motion_step_box = np.ones(3) * 0.2
    conf = ManifoldRRTConfig(1000)
    bitree = ManifoldRRTConnect(
        start, goal, b_min, b_max, motion_step_box, example_project, example_is_valid, config=conf
    )
    is_success = bitree.solve()
    assert is_success
    traj = bitree.get_solution()

    np.testing.assert_almost_equal(traj[0], start)
    np.testing.assert_almost_equal(traj[-1], goal)

    for q in traj:
        assert example_is_valid(q)
