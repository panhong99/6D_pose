"""M1013 kinematics and top-down grasp frames (pure numpy/scipy: no Isaac, no ROS).

Kinematics: forward/inverse kinematics from the Doosan URDF.  Checked against the real robot: all joints 0 gives a
flange at (0.14, 34.8, 1452.5) mm in this model vs (0.02, 34.5, 1452.5) mm read from the controller.
cube_yaw_deg / grasp_frames: top-down pregrasp flange pose above a cube (cube symmetry: yaw taken modulo 90 deg).
All transforms are 4x4 in metres; T_a_b maps b coordinates into a.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calib_utils import make_T

DEFAULT_URDF = Path('/home/pan/pan/doosan_ws/install/dsr_description2/share/dsr_description2/urdf/m1013.urdf')


class Kinematics:
    def __init__(self, urdf):
        self.joints = []
        for j in ET.parse(urdf).getroot().findall('joint'):
            if j.get('type') != 'revolute':
                continue
            o = j.find('origin')
            xyz = np.fromstring(o.get('xyz', '0 0 0'), sep=' ')
            rpy = np.fromstring(o.get('rpy', '0 0 0'), sep=' ')
            axis = np.fromstring(j.find('axis').get('xyz'), sep=' ')
            lim = j.find('limit')
            self.joints.append((j.find('child').get('link'), make_T(Rotation.from_euler('xyz', rpy).as_matrix(), xyz), axis,
                                float(lim.get('lower')), float(lim.get('upper'))))
        assert len(self.joints) == 6

    def frames(self, q):
        t = np.eye(4)
        result = {'base_link': t.copy()}
        for angle, (name, origin, axis, _, _) in zip(q, self.joints):
            t = t @ origin @ make_T(Rotation.from_rotvec(axis * angle).as_matrix(), [0, 0, 0])
            result[name] = t.copy()
        return result

    def fk(self, q):
        return self.frames(q)['link_6']

    def ik(self, target, seed):
        def residual(q):
            actual = self.fk(q)
            return np.r_[actual[:3, 3] - target[:3, 3],
                         Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec() * .25]
        fit = least_squares(residual, seed, bounds=([j[3] for j in self.joints], [j[4] for j in self.joints]), max_nfev=250)
        if np.linalg.norm(residual(fit.x)) > 1e-4:
            raise RuntimeError('Unreachable IK target')
        return fit.x


TOP_DOWN = np.diag([1.0, -1.0, -1.0])


def cube_yaw_deg(base_obj):
    """Yaw of the cube's side faces about base Z, modulo 90 deg, in [-45, 45).

    Uses the object axis that is most horizontal-aligned after dropping the
    most vertical one, so it works whichever face FoundationPose calls 'up'.
    """
    rot = base_obj[:3, :3]
    up = int(np.argmax(np.abs(rot[2])))
    side = rot[:, (up + 1) % 3]
    yaw = np.degrees(np.arctan2(side[1], side[0]))
    return float((yaw + 45.0) % 90.0 - 45.0), float(np.degrees(np.arccos(min(1.0, abs(rot[2, up])))))


def grasp_frames(kin, cube_centre, yaw_deg, tool_length_m, pregrasp_m):
    """Top-down wrist (j4 ~ 0, j5 points the tool down) with j6 on a cube face.

    Returns (yaw, pregrasp flange T, grasp flange T, q_pregrasp, q_grasp) for the
    face-aligned yaw (cube repeats every 90 deg) needing the least j6 twist.
    The TCP sits tool_length_m along flange +z; grasp puts the TCP at the cube
    centre, pregrasp raises it by pregrasp_m in base Z.
    """
    seed = np.radians([np.degrees(np.arctan2(cube_centre[1], cube_centre[0])), 0, 90, 0, 90, 0])
    best = None
    for k in range(4):
        yaw = yaw_deg + 90.0 * k
        rotation = Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ TOP_DOWN
        flange_up = np.array([0.0, 0.0, tool_length_m])  # tool points down, so flange is above TCP
        pre = make_T(rotation, np.asarray(cube_centre) + [0, 0, pregrasp_m] + flange_up)
        grasp = make_T(rotation, np.asarray(cube_centre) + flange_up)
        try:
            q_pre = kin.ik(pre, seed)
            q_grasp = kin.ik(grasp, q_pre)
        except RuntimeError:
            continue
        twist = abs((np.degrees(q_pre[5]) + 180.0) % 360.0 - 180.0)
        if best is None or twist < best[0]:
            best = (twist, ((yaw + 180.0) % 360.0 - 180.0, pre, grasp, q_pre, q_grasp))
    if best is None:
        raise RuntimeError('No reachable top-down grasp yaw for this cube')
    return best[1]
