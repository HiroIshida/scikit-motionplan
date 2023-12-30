import copy
from dataclasses import dataclass
from typing import Callable, List, Tuple, overload

import numpy as np

from skmp.kinematics import ArticulatedEndEffectorKinematicsMap


class InvalidSamplePointError(Exception):
    pass


class EuclideanMetric:
    def __call__(self, p0: np.ndarray, p1: np.ndarray) -> float:
        return float(np.linalg.norm(p0 - p1))


@dataclass
class EndEffectorDistanceMetric:
    efkin: ArticulatedEndEffectorKinematicsMap

    def __post_init__(self):
        assert self.efkin.n_feature == 1

    def __call__(self, q0: np.ndarray, q1: np.ndarray) -> float:
        P, _ = self.efkin.map(np.array([q0, q1]))
        x0 = P[0, 0]
        x1 = P[1, 0]
        return float(np.linalg.norm(x0 - x1))


class Trajectory:
    _points: List[np.ndarray]

    def __init__(self, points: List[np.ndarray]):
        self._points = points

    def get_length(
        self, metric: Callable[[np.ndarray, np.ndarray], float] = EuclideanMetric()
    ) -> float:
        n_point = len(self._points)
        total = 0.0
        for i in range(n_point - 1):
            p0 = self._points[i]
            p1 = self._points[i + 1]
            total += metric(p0, p1)
        return total

    def sample_point(
        self,
        dist_from_start: float,
        metric: Callable[[np.ndarray, np.ndarray], float] = EuclideanMetric(),
    ) -> np.ndarray:

        L = self.get_length(metric)
        if dist_from_start > L + 1e-6:
            raise InvalidSamplePointError("exceed total length")

        dist_from_start = min(dist_from_start, L)
        edge_dist_sum = 0.0
        for i in range(len(self) - 1):
            edge_dist_sum += metric(self._points[i + 1], self._points[i])
            if dist_from_start <= edge_dist_sum:
                diff = edge_dist_sum - dist_from_start
                vec_to_prev = self._points[i] - self._points[i + 1]
                vec_to_prev_unit = vec_to_prev / metric(self._points[i], self._points[i + 1])
                point_new = self._points[i + 1] + vec_to_prev_unit * diff
                return point_new
        raise InvalidSamplePointError()

    def resample(
        self, n_waypoint: int, metric: Callable[[np.ndarray, np.ndarray], float] = EuclideanMetric()
    ) -> "Trajectory":

        # yeah, it's inefficient. n^2 instead of n ...
        L = self.get_length(metric)
        point_new_list = []
        partial_length = L / (n_waypoint - 1)
        for i in range(n_waypoint):
            dist_from_start = partial_length * i
            point_new = self.sample_point(dist_from_start, metric)
            point_new_list.append(point_new)
        return Trajectory(point_new_list)

    def get_duplicate_removed(self) -> "Trajectory":
        point_new_list: List[np.ndarray] = []
        eps = 1e-4
        for point in self._points:
            if len(point_new_list) == 0:
                point_new_list.append(point)
            else:
                diff_from_prev = self._points[-1] - point
                if np.all(np.abs(diff_from_prev) > eps):
                    point_new_list.append(point)
        return Trajectory(point_new_list)

    def numpy(self):
        return np.array(self._points)

    def visualize(self, fax: Tuple, *args, **kwargs) -> None:
        fig, ax = fax
        arr = self.numpy()
        ax.plot(arr[:, 0], arr[:, 1], *args, **kwargs)

    @classmethod
    def from_two_points(cls, start: np.ndarray, goal: np.ndarray, n_waypoint) -> "Trajectory":
        diff = goal - start
        points = [start + diff / (n_waypoint - 1) * i for i in range(n_waypoint)]
        return cls(points)

    @overload
    def __getitem__(self, indices: List[int]) -> List[np.ndarray]:
        pass

    @overload
    def __getitem__(self, indices: slice) -> List[np.ndarray]:
        pass

    @overload
    def __getitem__(self, index: int) -> np.ndarray:
        pass

    def __getitem__(self, indices_like):
        points = self._points
        return points[indices_like]  # type: ignore

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self):
        return self._points.__iter__()

    def adjusted(self, goal: np.ndarray) -> "Trajectory":
        dists = np.sum((self._points - goal) ** 2, axis=1)
        idx_min = np.argmin(dists).item()
        points_new = self._points[: idx_min + 1] + [goal]
        traj_new = Trajectory(points_new)
        return traj_new.resample(len(traj_new) - 1)

    def append(self, point: np.ndarray) -> None:
        self._points.append(point)

    def __add__(self, other: "Trajectory") -> "Trajectory":
        diff_contact = np.linalg.norm(self._points[-1] - other._points[0])
        assert diff_contact < 1e-6
        points = copy.deepcopy(self._points) + copy.deepcopy(other._points[1:])
        return Trajectory(points)
