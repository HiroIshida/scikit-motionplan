import copy
from pathlib import Path
from typing import Dict

from skrobot.utils.checksum import checksum_md5
from skrobot.utils.urdf import URDF

_loaded_urdf_models: Dict[str, URDF] = {}


def load_urdf_model_using_cache(file_path: Path):
    file_path = file_path.expanduser()
    assert file_path.exists()

    hashvalue = checksum_md5(str(file_path))

    if hashvalue not in _loaded_urdf_models:
        urdf = URDF.load(str(file_path))
        _loaded_urdf_models[hashvalue] = urdf
    return copy.deepcopy(_loaded_urdf_models[hashvalue])
