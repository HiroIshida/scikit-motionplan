import time
from pathlib import Path
import copy
from tempfile import TemporaryDirectory
import numpy as np
from dataclasses import dataclass
from typing import Protocol, List, Dict, Any, Tuple

import xml.etree.ElementTree as ET
from robot_descriptions.jaxon_description import URDF_PATH
from robot_descriptions.loaders.pinocchio import load_robot_description
from skrobot.models.pr2 import PR2
from skrobot.models.urdf import RobotModelFromURDF
from skrobot.model.primitives import Axis
from skrobot.coordinates import Coordinates
from skrobot.viewers import TrimeshSceneViewer
import tinyfk
import skrobot
from skrobot.model import Link
import pinocchio as pin


class KinematicsSolverInterface(Protocol):

    def get_joint_ids(self, joint_names):
        ...

    def get_link_ids(self, link_names):
        ...

    def set_joint_angles(self, angles):
        ...


@dataclass
class PinocchioWrapper:
    frame_idx_table: Dict[str, int]
    joint_idx_table: Dict[str, int]
    model: pin.RobotWrapper
    joint_angles: np.ndarray

    @staticmethod
    def find_root_link_name(root: ET) -> str:
        # find root link with observation that
        # only the root link cannot be a child link of anu joints
        links = root.findall('link')
        link_names = set([l.get("name") for l in links])

        link_child_names = []
        joints = root.findall('joint')
        for joint in joints:
            child = joint.find('child')
            link_child_names.append(child.get('link'))
        link_child_names = set(link_child_names)
        a = link_names.difference(link_child_names)
        return list(a)[0]

    @classmethod
    def from_urdf_path(cls, urdf_path: str) -> "PinocchioWrapper":
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        root_link_name = cls.find_root_link_name(root)
        new_link = ET.Element("link", {"name": "floating_base"})
        root.append(new_link)

        new_joint = ET.Element("joint", {"name": "floating_joint", "type": "floating"})
        origin = ET.Element("origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        new_joint.append(origin)

        parent = ET.Element("parent", {"link": "floating_base"})
        new_joint.append(parent)

        child = ET.Element("child", {"link": root_link_name})
        new_joint.append(child)

        axis = ET.Element("axis", {"xyz": "0 0 1"})
        new_joint.append(axis)

        root.append(new_joint)

        # replace continous joint with revolute joint
        continuous_joints = root.findall(".//joint[@type='continuous']")
        for joint in continuous_joints:
            joint.set("type", "revolute")
            limit = joint.find("limit")
            if limit is not None:
                    limit.set("lower", "-10.0")
                    limit.set("upper", "10.0")
                    limit.set("effort", "10.0")
                    limit.set("velocity", "10.0")
            else:
                new_limit = ET.Element(
                    "limit",
                    {"lower": "-10.0",
                     "upper": "10.0",
                     "effort": "10.0",
                     "velocity": "10.0"})
                joint.append(new_limit)

        # remove 
        with TemporaryDirectory() as td:
            td_path = Path(td)
            temp_urdf_path = td_path / "temp.urdf"
            tree.write(str(temp_urdf_path))
            tree.write("/tmp/tmp.urdf")

            model = pin.RobotWrapper.BuildFromURDF(
                filename=str(temp_urdf_path),
                package_dirs=None,
                root_joint=None,
            )
          
            frame_idx_table = {f.name: i for i, f in enumerate(model.model.frames)}
            joint_idx_table = {name: i for i, name in enumerate(model.model.names)}
            del joint_idx_table["universe"]
            del joint_idx_table["floating_joint"]

            joint_angles = np.zeros(model.nq)
            return cls(frame_idx_table, joint_idx_table, model, joint_angles)

    def get_joint_ids(self, joint_names: List[str]) -> np.ndarray:
        return np.array([self.joint_idx_table[jn] for jn in joint_names])

    def get_link_ids(self, link_names: List[str]) -> np.ndarray:
        return np.array([self.frame_idx_table[ln] for ln in link_names])

    def set_joint_angles(self, joint_ids: np.ndarray, angles: np.ndarray, with_base: bool = False) -> None:
        self.joint_angles[joint_ids - 2 + 7] = angles  # -1 because universe is deleted

    def solve_forward_kinematics(
        self,
        joint_angles_sequence,
        elink_ids,
        joint_ids,
        with_rot=False,
        with_base=False,
        with_jacobian=False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert not with_base
        points = []
        jacs = []
        for av_partial in joint_angles_sequence:
            av = copy.deepcopy(self.joint_angles)
            av[joint_ids - 2 + 7] = av_partial
            self.model.forwardKinematics(av)
            self.model.computeJointJacobians(av)

            for elink_id in elink_ids:
                point = self.model.framePlacement(av, int(elink_id), update_kinematics=False)
                jac = self.model.getFrameJacobian(int(elink_id), rf_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                if with_rot:
                    assert False
                else:
                    if with_base:
                        assert False
                    else:
                        jac = jac[3:, joint_ids - 2 + 7]
                    point = point.translation
                points.append(point)
                jacs.append(jac)
        return np.array(points), np.vstack(jacs)


if __name__ == "__main__":
    pr2 = PR2()
    pr2.reset_manip_pose()
    joint_angle_table = {jn: pr2.__dict__[jn].joint_angle() for jn in pr2.joint_names}

    joint_names = ["r_shoulder_pan_joint", "r_shoulder_lift_joint", "r_upper_arm_roll_joint", "r_elbow_flex_joint", "r_forearm_roll_joint", "r_wrist_flex_joint", "r_wrist_roll_joint"]
    link_names = ["r_shoulder_pan_link", "r_shoulder_lift_link", "r_upper_arm_roll_link", "r_elbow_flex_link", "r_forearm_roll_link", "r_wrist_flex_link", "r_wrist_roll_link", "base_link", "r_upper_arm_link"]

    urdf_model_path = tinyfk.pr2_urdfpath()

    # compute using tinyfk
    kin_solver = tinyfk.RobotModel(urdf_model_path)
    tfk_av = np.array([joint_angle_table[jn] for jn in joint_names])
    joint_ids = kin_solver.get_joint_ids(joint_names)
    elink_ids = kin_solver.get_link_ids(link_names)

    all_joint_ids = kin_solver.get_joint_ids(list(joint_angle_table.keys()))
    kin_solver.set_joint_angles(all_joint_ids, list(joint_angle_table.values()))
    ts = time.time()
    for _ in range(1000):
        P_tiny, J_tiny = kin_solver.solve_forward_kinematics([tfk_av], elink_ids=elink_ids, joint_ids=joint_ids, with_jacobian=True)
    print(time.time() - ts)

    # compute using pinocchio
    pinwrapper = PinocchioWrapper.from_urdf_path(urdf_model_path)
    all_joint_ids = pinwrapper.get_joint_ids(list(joint_angle_table.keys()))
    pinwrapper.set_joint_angles(all_joint_ids, np.array(list(joint_angle_table.values())))
    # pin_av = np.array([joint_angle_table[name] for name, idx in pinwrapper.joint_idx_table.items()])
    pin_av = np.array([joint_angle_table[name] for name in joint_names])

    joint_ids = pinwrapper.get_joint_ids(joint_names)
    elink_ids = pinwrapper.get_link_ids(link_names)

    print("hoge")
    from pyinstrument import Profiler
    profiler = Profiler()
    profiler.start()
    ts = time.time()
    for _ in range(1000):
        P_pin, J_pin = pinwrapper.solve_forward_kinematics([pin_av], elink_ids=elink_ids, joint_ids=joint_ids)
    print(time.time() - ts)
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True, show_all=True))

    np.testing.assert_almost_equal(P_pin, P_tiny, decimal=3)
    
    # print(P_tiny - P_pin)
    # print(J_tiny - J_pin)
