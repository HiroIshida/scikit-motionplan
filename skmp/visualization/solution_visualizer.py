import collections
import copy
import tempfile
import time
from abc import abstractmethod
from pathlib import Path
from typing import (
    Callable,
    Collection,
    Dict,
    Generic,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
)

import imageio
import numpy as np
import trimesh
from skrobot.model import CascadedLink, Link, RobotModel
from skrobot.viewers import TrimeshSceneViewer

from skmp.kinematics import ArticulatedCollisionKinematicsMap
from skmp.trajectory import Trajectory
from skmp.visualization.collision_visualizer import CollisionSphereVisualizationManager


class SceneWrapper(trimesh.Scene):
    """
    This class is almost copied from skrobot.viewers.TrimeshSceneViewer
    But slightly differs to save figures
    """

    _links: Dict[str, Link]

    def __init__(self):
        super(SceneWrapper, self).__init__()
        self._links = collections.OrderedDict()

    def show(self):
        pass

    def redraw(self):
        # apply latest angle-vector
        for link_id, link in self._links.items():
            link.update(force=True)
            transform = link.worldcoords().T()
            self.graph.update(link_id, matrix=transform)

    def update_scene_graph(self):
        # apply latest angle-vector
        for link_id, link in self._links.items():
            link.update(force=True)
            transform = link.worldcoords().T()
            self.graph.update(link_id, matrix=transform)

    @staticmethod
    def convert_geometry_to_links(geometry):
        if isinstance(geometry, Link):
            links = [geometry]
        elif isinstance(geometry, CascadedLink):
            links = geometry.link_list
        else:
            raise TypeError("geometry must be Link or CascadedLink")
        return links

    def add(self, link):
        links = self.convert_geometry_to_links(link)

        for link in links:
            link_id = str(id(link))
            if link_id in self._links:
                return
            transform = link.worldcoords().T()
            mesh = link.visual_mesh
            # TODO(someone) fix this at trimesh's scene.
            if (isinstance(mesh, list) or isinstance(mesh, tuple)) and len(mesh) > 0:
                mesh = trimesh.util.concatenate(mesh)
            self.add_geometry(
                geometry=mesh,
                node_name=link_id,
                geom_name=link_id,
                transform=transform,
            )
            self._links[link_id] = link


ViewerT = TypeVar("ViewerT", bound=Union[TrimeshSceneViewer, SceneWrapper])


class VisualizableProtocol(Protocol[ViewerT]):  # type: ignore
    def visualize(self, viewer: ViewerT) -> None:
        ...


Geometry = Union[Link, Collection[Link]]
Visualizable = Union[VisualizableProtocol[ViewerT], Collection[VisualizableProtocol[ViewerT]]]


class SolutionVisualizerBase(Generic[ViewerT]):
    viewer: ViewerT
    robot_model: RobotModel
    _robot_updator: Optional[Callable[[RobotModel, np.ndarray], None]]
    _show_called: bool
    _flags: Dict
    _colvis: Optional[CollisionSphereVisualizationManager]

    @classmethod
    @abstractmethod
    def viewer_type(cls) -> Type[ViewerT]:
        ...

    def __init__(
        self,
        robot: RobotModel,
        geometry: Optional[Geometry] = None,
        visualizable: Optional[Visualizable] = None,
        robot_updator: Optional[Callable[[RobotModel, np.ndarray], None]] = None,
        show_wireframe: bool = False,
        enable_colvis: bool = False,
        colkin: Optional[ArticulatedCollisionKinematicsMap] = None,
        sdf: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):

        viewer = self.viewer_type()()

        if visualizable is not None:
            if not isinstance(visualizable, Collection):
                visualizable = [visualizable]  # type: ignore
            for vable in visualizable:
                vable.visualize(viewer)

        if geometry is not None:
            if not isinstance(geometry, Collection):
                geometry = [geometry]  # type: ignore
            for geo in geometry:
                viewer.add(geo)

        viewer.add(robot)

        flags = {}
        if show_wireframe:
            flags["wireframe"] = True

        if enable_colvis:
            assert colkin is not None
            assert sdf is not None
            self._colvis = CollisionSphereVisualizationManager(colkin, viewer, sdf)
        else:
            self._colvis = None

        self.robot_model = robot
        self.viewer = viewer
        self._robot_updator = robot_updator
        self._show_called = False
        self.flags = flags

    def update_robot_state(self, q: np.ndarray) -> None:
        # please overwrite this
        assert self._robot_updator is not None
        self._robot_updator(self.robot_model, q)
        if self._colvis is not None:
            self._colvis.update(self.robot_model)


class InteractiveSolutionVisualizer(SolutionVisualizerBase[TrimeshSceneViewer]):
    def show(self) -> None:
        if not self._show_called:
            # NOTE: wire frame is not supporeted yet. Just ignore
            self.viewer.show()
            time.sleep(1.0)
            self._show_called = True

    def visualize_trajectory(self, trajectory: Trajectory, t_interval: float = 0.6) -> None:
        self.show()

        q_end = trajectory.numpy()[-1]
        self.update_robot_state(q_end)

        for q in trajectory.numpy():
            self.update_robot_state(q)
            self.viewer.redraw()
            time.sleep(t_interval)

        print("==> Press [q] to close window")
        while not self.viewer.has_exit:
            time.sleep(0.1)
            self.viewer.redraw()

    @classmethod
    def viewer_type(cls) -> Type[TrimeshSceneViewer]:
        return TrimeshSceneViewer


class StaticSolutionVisualizer(SolutionVisualizerBase[SceneWrapper]):
    @classmethod
    def viewer_type(cls) -> Type[SceneWrapper]:
        return SceneWrapper

    def save_image(self, path: Union[Path, str]) -> None:
        if isinstance(path, str):
            path = Path(path)

        png = self.viewer.save_image(resolution=[640, 480], visible=True, flags=self.flags)
        with path.open(mode="wb") as f:
            f.write(png)

    def save_trajectory_gif(self, trajectory: Trajectory, path: Union[Path, str]) -> None:

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            file_path_list = []

            for i, q in enumerate(trajectory.numpy()):
                self.update_robot_state(q)
                self.viewer.redraw()
                time.sleep(0.5)
                file_path = td_path / "{}.png".format(i)
                file_path_list.append(file_path)
                self.save_image(file_path)

            images = []
            for file_path in file_path_list:
                images.append(imageio.imread(file_path))
            for _ in range(10):
                images.append(imageio.imread(file_path_list[-1]))
            imageio.mimsave(path, images)

    def save_trajectory_image(self, trajectory: Trajectory, path: Union[Path, str]) -> None:

        for q in trajectory.numpy():
            robot_model_copied = copy.deepcopy(self.robot_model)
            self.update_robot_state(q)
            self.set_robot_alpha(robot_model_copied, 30)
            self.viewer.add(robot_model_copied)

        robot_model_copied = copy.deepcopy(self.robot_model)
        self.update_robot_state(trajectory.numpy()[-1])
        self.viewer.add(robot_model_copied)

        if isinstance(path, str):
            path = Path(path)
        png = self.viewer.save_image(resolution=[640, 480], visible=True)
        with path.open(mode="wb") as f:
            f.write(png)

    @staticmethod
    def set_robot_alpha(robot: RobotModel, alpha: int):
        assert alpha < 256
        for link in robot.link_list:
            visual_mesh = link.visual_mesh
            if isinstance(visual_mesh, list):
                for mesh in visual_mesh:
                    mesh.visual.face_colors[:, 3] = alpha
            else:
                visual_mesh.visual.face_colors[:, 3] = alpha
