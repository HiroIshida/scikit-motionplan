from typing import Callable, List, Optional

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model import RobotModel
from skrobot.model.primitives import Sphere
from skrobot.viewers import TrimeshSceneViewer

from skmp.kinematics import ArticulatedCollisionKinematicsMap


class SphereColor:
    NORMAL = (250, 250, 10, 200)
    COLLISION = (255, 0, 0, 200)


class CollisionSphereVisualizationManager:
    kinmap: ArticulatedCollisionKinematicsMap
    viewer: TrimeshSceneViewer
    sphere_list: List[Sphere]
    sdf: Optional[Callable[[np.ndarray], np.ndarray]]

    def __init__(
        self,
        kinmap: ArticulatedCollisionKinematicsMap,
        viewer: TrimeshSceneViewer,
        sdf: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        sphere_list = []
        for i in range(kinmap.n_feature):
            kinmap.sphere_name_list[i]
            r = kinmap.radius_list[i]
            center = kinmap.sphere_center_list[i]
            sphere = Sphere(radius=r, pos=center, color=SphereColor.NORMAL)
            sphere_list.append(sphere)

        for sphere in sphere_list:
            viewer.add(sphere)

        self.kinmap = kinmap
        self.viewer = viewer
        self.sphere_list = sphere_list
        self.sdf = sdf

    def update(
        self, robot: RobotModel, sdf_latest: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ) -> None:

        if sdf_latest is not None:
            self.sdf = sdf_latest

        self.kinmap.reflect_skrobot_model(robot)
        tmp, _ = self.kinmap.map_skrobot_model(robot)
        points_tspace = tmp[0]
        assert len(points_tspace) == len(self.sphere_list)

        for point, sphere in zip(points_tspace, self.sphere_list):
            co = Coordinates(point)
            sphere.newcoords(co)

        if self.sdf is not None:
            # warn collision state by chaning the color
            for sphere in self.sphere_list:
                point = sphere.worldpos()
                radius = sphere.visual_mesh.metadata["radius"]
                val = self.sdf(np.expand_dims(point, axis=0)).item() - radius
                if val < 0.0:
                    n_facet = len(sphere._visual_mesh.visual.face_colors)
                    sphere._visual_mesh.visual.face_colors = np.array(
                        [SphereColor.COLLISION] * n_facet
                    )
