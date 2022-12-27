import uuid
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.decomposition import PCA
from trimesh import Trimesh


@dataclass
class SphereCollection:
    center_list: List[np.ndarray]
    radius_list: List[float]
    name_list: List[str]

    def __len__(self) -> int:
        return len(self.center_list)


@dataclass
class SphereCreatorConfig:
    tol: float = 0.1
    radius_scale: float = 1.0


def create_sphere_collection(
    mesh: Trimesh, config: Optional[SphereCreatorConfig] = None
) -> SphereCollection:
    if config is None:
        config = SphereCreatorConfig()

    verts = mesh.vertices
    pca = PCA(n_components=3)
    pca.fit(verts)

    principle_axis = 0
    plane_axes = [1, 2]

    # then compute the bounding-circle for vertices projected
    # to the plane.
    def determine_radius(verts_2d_projected):
        X, Y = verts_2d_projected.T
        radius_vec = np.sqrt(X**2 + Y**2)
        radius = np.max(radius_vec)
        return radius

    margin_factor = 1.01
    verts_mapped = pca.transform(mesh.vertices)
    radius = determine_radius(verts_mapped[:, plane_axes]) * margin_factor

    # compute the maximum and minimum heights (h_center_max, h_center_min)
    # of the sphere centers. Here, hight is defined in the principle direction.
    squared_radius_arr = np.sum(verts_mapped[:, plane_axes] ** 2, axis=1)
    h_center_arr = verts_mapped[:, principle_axis]

    h_vert_max = np.max(verts_mapped[:, principle_axis])
    h_vert_min = np.min(verts_mapped[:, principle_axis])

    def get_h_center_max():
        def cond_all_inside_positive(h_center_max):
            sphere_heights = h_center_max + np.sqrt(radius**2 - squared_radius_arr)
            return np.all(sphere_heights > h_center_arr)

        # get first index that satisfies the condition
        h_cand_list = np.linspace(0, h_vert_max, 30)
        idx = np.where([cond_all_inside_positive(h) for h in h_cand_list])[0][0]
        h_center_max = h_cand_list[idx]
        return h_center_max

    def get_h_center_min():
        def cond_all_inside_negative(h_center_min):
            sphere_heights = h_center_min - np.sqrt(radius**2 - squared_radius_arr)
            return np.all(h_center_arr > sphere_heights)

        # get first index that satisfies the condition
        h_cand_list = np.linspace(0, h_vert_min, 30)
        idx = np.where([cond_all_inside_negative(h) for h in h_cand_list])[0][0]
        h_center_min = h_cand_list[idx]
        return h_center_min

    h_center_max = get_h_center_max()
    h_center_min = get_h_center_min()

    # using h_center_min and h_center_max, generate center points in
    # the mapped space.
    def compute_center_pts_mapped_space(n_sphere):
        h_centers = np.linspace(h_center_min, h_center_max, n_sphere)
        centers = np.zeros((n_sphere, 3))
        centers[:, principle_axis] = h_centers
        return centers

    n_sphere = 1
    while True:  # iterate until the approximation satisfies tolerance
        centers_pts_mapped_space = compute_center_pts_mapped_space(n_sphere)
        dists_foreach_sphere = np.array(
            [
                np.sqrt(np.sum((verts_mapped - c[None, :]) ** 2, axis=1))
                for c in centers_pts_mapped_space
            ]
        )
        # verts distance to the approximating spheres
        # if this distance is positive value, the vertex is jutting-out
        # from the swept-sphere.
        jut_dists = np.min(dists_foreach_sphere, axis=0) - radius
        max_jut = np.max(jut_dists)
        err_ratio = max_jut / radius
        if err_ratio < config.tol:
            break
        n_sphere += 1

    # map all centers to the original space
    centers_original_space = pca.inverse_transform(centers_pts_mapped_space)
    radius_list = [radius * config.radius_scale for _ in range(n_sphere)]
    name_list = [str(uuid.uuid4()) for _ in range(n_sphere)]
    return SphereCollection(list(centers_original_space), radius_list, name_list)
