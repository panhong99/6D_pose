"""Build m1013_2f85.urdf: Doosan M1013 + Robotiq 2F-85 for Isaac Lab (numpy only, run with any python).

Geometry/inertia come from DeepMind mujoco_menagerie robotiq_2f85 (assets/robotiq_2f85/2f85.xml, open pose).
Simplification (physics-friendly): the 4-bar linkage is NOT simulated.  Two prismatic finger links (follower + pad
meshes, pad box collisions) slide toward each other; driver/coupler/spring links are static decoration.
Joints: right_finger_joint / left_finger_joint, 0 = open (85 mm), 0.0425 = closed.  Gripper origin is COUPLING_M
(AGC-DOO-KIT-85 thickness, 20 mm confirmed by Human 2026-10-02) along link_6 +z.
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GRIPPER = HERE.parent / 'assets' / 'robotiq_2f85'
M1013_SHARE = Path('/home/pan/pan/doosan_ws/install/dsr_description2/share/dsr_description2')
STROKE = 0.0425


def quat_R(q):
    w, x, y, z = np.array(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mk(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def rpy(R):
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-9:
        return np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], sy), np.arctan2(R[1, 0], R[0, 0])
    return np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(-R[2, 0], sy), 0.0


def origin(T):
    r, p, y = rpy(T[:3, :3])
    return f'<origin xyz="{T[0, 3]:.6f} {T[1, 3]:.6f} {T[2, 3]:.6f}" rpy="{r:.6f} {p:.6f} {y:.6f}"/>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coupling-m', type=float, default=0.02)
    ap.add_argument('--out', type=Path, default=HERE / 'assets' / 'm1013_2f85.urdf')
    args = ap.parse_args()

    root = ET.parse(GRIPPER / '2f85.xml').getroot()
    meshfile = {m.get('file')[:-4]: m.get('file') for m in root.iter('mesh') if m.get('file')}
    bodies = {}                                              # name -> (T in gripper-origin frame, body element)

    def walk(body, T_parent):
        T = T_parent @ mk(quat_R([float(v) for v in body.get('quat', '1 0 0 0').split()]),
                          [float(v) for v in body.get('pos', '0 0 0').split()])
        bodies[body.get('name')] = (T, body)
        for c in body.findall('body'):
            walk(c, T)
    walk(root.find('worldbody').find('body'), np.eye(4))

    def mesh_xml(tag, T_link, mesh, rgba=None):
        f = (GRIPPER / 'assets' / meshfile[mesh]).as_posix()
        mat = f'<material name="m_{mesh}_{abs(hash((tag, str(T_link.round(4))))) % 10**6}"><color rgba="{rgba}"/></material>' if rgba else ''
        return (f'<{tag}>{origin(T_link)}<geometry><mesh filename="{f}" scale="0.001 0.001 0.001"/></geometry>'
                f'{mat if tag == "visual" else ""}</{tag}>')

    black, grey = '0.15 0.15 0.15 1', '0.46 0.46 0.46 1'
    # --- gripper_base: base_mount + base (visual+collision), static decoration for the linkage ---
    parts = ['<link name="gripper_base"><inertial><origin xyz="0 0 0.04"/><mass value="0.777"/>'
             '<inertia ixx="2.6e-4" ixy="0" ixz="0" iyy="2.3e-4" iyz="0" izz="1.5e-4"/></inertial>']
    for name, mesh, rgba, coll in (('base_mount', 'base_mount', black, True), ('base', 'base', black, True),
                                   ('right_driver', 'driver', grey, False), ('right_coupler', 'coupler', black, False),
                                   ('right_spring_link', 'spring_link', black, False),
                                   ('left_driver', 'driver', grey, False), ('left_coupler', 'coupler', black, False),
                                   ('left_spring_link', 'spring_link', black, False)):
        T = bodies[name][0]
        parts.append(mesh_xml('visual', T, mesh, rgba))
        if coll:
            parts.append(mesh_xml('collision', T, mesh))
    parts.append('</link>')
    links = [''.join(parts)]
    joints = []
    # --- finger links ---
    for side in ('right', 'left'):
        T_pad, pad = bodies[f'{side}_pad']
        T_fol = bodies[f'{side}_follower'][0]
        center = np.array([0.0, 0.0, T_pad[2, 3]])
        d_world = center - T_pad[:3, 3]
        d_world[2] = 0.0
        d_world /= np.linalg.norm(d_world)                      # closing direction in gripper frame
        axis = T_pad[:3, :3].T @ d_world                        # in the pad (joint) frame
        T_fol_in_pad = np.linalg.inv(T_pad) @ T_fol
        T_sil = T_pad                                           # silicone pad body has no offset in the pad body
        s = [f'<link name="{side}_finger"><inertial><origin xyz="0 0 0.02"/><mass value="0.05"/>'
             '<inertia ixx="2e-5" ixy="0" ixz="0" iyy="2e-5" iyz="0" izz="1e-5"/></inertial>',
             mesh_xml('visual', T_fol_in_pad, 'follower', black),
             mesh_xml('visual', np.eye(4), 'pad', grey),
             mesh_xml('visual', np.eye(4), 'silicone_pad', black)]
        for g in pad.findall('geom'):
            if g.get('class') in ('pad_box1', 'pad_box2'):
                d = next(x for x in root.iter('default') if x.get('class') == g.get('class')).find('geom')   # pos/size live in the class default
                pos = [float(v) for v in d.get('pos').split()]
                size = [2 * float(v) for v in d.get('size').split()]
                s.append(f'<collision><origin xyz="{pos[0]} {pos[1]} {pos[2]}" rpy="0 0 0"/><geometry><box size="{size[0]} {size[1]} {size[2]}"/></geometry></collision>')
        s.append('</link>')
        links.append(''.join(s))
        joints.append(f'<joint name="{side}_finger_joint" type="prismatic"><parent link="gripper_base"/><child link="{side}_finger"/>'
                      f'{origin(T_pad)}<axis xyz="{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f}"/>'
                      f'<limit lower="0" upper="{STROKE}" effort="235" velocity="0.15"/><dynamics damping="1.0" friction="0.0"/></joint>')
    # --- TCP frame (pinch site, 0.145 along the 'base' z) ---
    T_tcp = bodies['base'][0] @ mk(np.eye(3), [0, 0, 0.145])
    links.append('<link name="tcp"/>')
    joints.append(f'<joint name="tcp_joint" type="fixed"><parent link="gripper_base"/><child link="tcp"/>{origin(T_tcp)}</joint>')
    joints.append(f'<joint name="gripper_mount" type="fixed"><parent link="link_6"/><child link="gripper_base"/>'
                  f'<origin xyz="0 0 {args.coupling_m}" rpy="0 0 0"/></joint>')

    base = (M1013_SHARE / 'urdf' / 'm1013.urdf').read_text().replace('package://dsr_description2', M1013_SHARE.as_posix())
    assert '</robot>' in base
    out = base.replace('</robot>', '\n  ' + '\n  '.join(links + joints) + '\n</robot>')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out)
    print(f'wrote {args.out}  (coupling {args.coupling_m} m, TCP {args.coupling_m + T_tcp[2, 3]:.4f} m from link_6)')
    for side in ('right', 'left'):
        print(side, 'pad pos in gripper frame', np.round(bodies[f"{side}_pad"][0][:3, 3], 4).tolist())


main()
