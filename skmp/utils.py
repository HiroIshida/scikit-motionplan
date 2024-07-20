import copy
from pathlib import Path
from typing import Dict

import psdf
from skrobot.sdf.signed_distance_function import (
    BoxSDF,
    CylinderSDF,
    SignedDistanceFunction,
)
from skrobot.utils.checksum import checksum_md5
from skrobot.utils.urdf import URDF

_loaded_urdf_models_with_geometry: Dict[str, URDF] = {}
_loaded_urdf_models: Dict[str, URDF] = {}


def load_urdf_model_using_cache(file_path: Path, with_geometry: bool = False):
    file_path = file_path.expanduser()
    assert file_path.exists()

    hashvalue = checksum_md5(str(file_path))

    if hashvalue not in _loaded_urdf_models_with_geometry:
        urdf = URDF.load(str(file_path))
        _loaded_urdf_models_with_geometry[hashvalue] = urdf

        # create urdf without mesh, which is easy to copy
        urdf_without_geometry: URDF = copy.deepcopy(urdf)
        for link in urdf_without_geometry._links:
            link.visuals = None
            link.collisions = None
        _loaded_urdf_models[hashvalue] = urdf_without_geometry

    if with_geometry:
        return copy.deepcopy(_loaded_urdf_models_with_geometry[hashvalue])
    else:
        return copy.deepcopy(_loaded_urdf_models[hashvalue])


def sksdf_to_cppsdf(sksdf: SignedDistanceFunction):
    pose = psdf.Pose(sksdf.worldpos(), sksdf.worldrot())
    if isinstance(sksdf, BoxSDF):
        sdf = psdf.BoxSDF(sksdf._width, pose)

    elif isinstance(sksdf, CylinderSDF):
        sdf = psdf.CylinderSDF(sksdf._radius, sksdf._height, pose)
    else:
        raise ValueError("Unsupported SDF type")
    return lambda pts: sdf.evaluate(pts.T)
