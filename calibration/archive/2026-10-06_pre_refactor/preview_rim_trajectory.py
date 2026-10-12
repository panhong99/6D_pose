"""Generate a simulation-only preview: home -> approach -> 90-degree gripper roll.

No ROS imports or robot commands. Coordinates are in the robot base frame (metres).
The endpoint is deliberately a standoff pose, not a physical grasp or a real-robot plan.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from kinematics import DEFAULT_URDF, HOME_JOINTS_DEG, Kinematics

HERE = Path(__file__).resolve().parent
TOOL_LENGTH = 0.1758
FPS = 30


def tcp_target(position, rotation):
    target = np.eye(4)
    target[:3, :3] = rotation
    target[:3, 3] = np.asarray(position) - TOOL_LENGTH * rotation[:, 2]
    return target


def add_segment(frames, phases, start, end, seconds, phase):
    count = round(seconds * FPS)
    for t in np.linspace(0.0, 1.0, count + 1)[1:]:
        u = t * t * (3.0 - 2.0 * t)
        frames.append(start + u * (end - start))
        phases.append(phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE / 'data' / 'rim_gripper_roll_preview_20261006.npz')
    args = parser.parse_args()
    kin = Kinematics(DEFAULT_URDF)
    home = np.radians(HOME_JOINTS_DEG)

    # Preserve the exact home gripper orientation throughout the approach.
    # At the endpoint, only joint 6 turns; its axis is the flange/tool z axis.
    home_rotation = kin.fk(home)[:3, :3]
    p_far = np.array([0.72, -0.08, 0.33])
    p_turn = np.array([0.94, -0.08, 0.12])

    q_far = kin.ik(tcp_target(p_far, home_rotation), home)
    approach_points = [q_far]
    for t in np.linspace(0.1, 1.0, 10):
        point = p_far + t * (p_turn - p_far)
        approach_points.append(kin.ik(tcp_target(point, home_rotation), approach_points[-1]))
    q_turn = approach_points[-1]
    q_rolled = q_turn.copy()
    q_rolled[5] += np.pi / 2

    # Rotating joint 6 should leave the TCP position and tool pointing axis unchanged.
    before, after = kin.fk(q_turn), kin.fk(q_rolled)
    tcp_before = before[:3, 3] + TOOL_LENGTH * before[:3, 2]
    tcp_after = after[:3, 3] + TOOL_LENGTH * after[:3, 2]
    if np.linalg.norm(tcp_after - tcp_before) > 1e-6 or np.linalg.norm(after[:3, 2] - before[:3, 2]) > 1e-6:
        raise RuntimeError('joint 6 is not a pure roll about the tool axis in this model')
    p_lower = p_turn + np.array([0.0, 0.0, -0.02])
    q_lower = kin.ik(tcp_target(p_lower, after[:3, :3]), q_rolled)

    # Align the gripper's pinch-centre/tool ray with the robot-facing plate rim.
    # The 2F-85 mount in 2f85.xml rotates its closing axis into flange X; the
    # pinch centre itself lies on flange +Z at TOOL_LENGTH from the flange.
    plate_centre_xy = np.array([1.10, 0.0])
    rim_radius = 0.128
    rim_z = -0.02 + 0.0535 + 0.016
    ray = after[:3, 2]
    radial_start = p_lower[:2] - plate_centre_xy
    roots = np.roots([np.dot(ray[:2], ray[:2]), 2 * np.dot(radial_start, ray[:2]),
                      np.dot(radial_start, radial_start) - rim_radius**2])
    forward_roots = [float(v.real) for v in roots if abs(v.imag) < 1e-9 and v.real > 0]
    if not forward_roots:
        raise RuntimeError('gripper centreline does not intersect the plate rim')
    rim_distance_on_ray = min(forward_roots)
    p_rim = np.r_[p_lower[:2] + rim_distance_on_ray * ray[:2], rim_z]
    p_aligned = p_lower.copy()
    p_aligned[2] = rim_z - rim_distance_on_ray * ray[2]
    q_aligned = kin.ik(tcp_target(p_aligned, after[:3, :3]), q_lower)

    frames, phases = [home.copy()], ['home']
    add_segment(frames, phases, home, home, 2.0, 'home')
    add_segment(frames, phases, home, q_far, 6.0, 'approach')
    for a, b in zip(approach_points[:-1], approach_points[1:]):
        add_segment(frames, phases, a, b, 6.0 / (len(approach_points) - 1), 'approach')
    add_segment(frames, phases, q_turn, q_rolled, 5.0, 'gripper_roll_90_deg')
    add_segment(frames, phases, q_rolled, q_lower, 2.0, 'lower_20_mm')
    add_segment(frames, phases, q_lower, q_aligned, 2.5, 'align_rim_axis')
    add_segment(frames, phases, q_aligned, q_aligned, 6.0, 'ready_standoff')
    q = np.asarray(frames)

    # The RL voxel shield checks steel-frame and table distance at sampled poses.
    # Its current gripper box is approximate; support-box and plate clearance still
    # need visual inspection and physical measurements before any real use.
    import sys
    sys.path.insert(0, str(HERE.parent / 'rl'))
    from rl_utils import SafetyShield          # rl/ refactor 2026-10-06 (was rl/safety_shield.py)
    shield = SafetyShield('cpu')
    clearance = []
    for start in range(0, len(q), 32):
        with torch.no_grad():
            frame, table = shield.clearance(torch.tensor(q[start:start + 32], dtype=torch.float32))
        clearance.append(np.column_stack([frame.numpy(), table.numpy()]))
    clearance = np.vstack(clearance)
    min_frame, min_table = clearance.min(axis=0)
    if min_frame < 0.05 or min_table < 0.02:
        raise RuntimeError(f'preview fails shield: frame {min_frame:.3f} m, table {min_table:.3f} m')

    metadata = {
        'simulation_only': True,
        'fps': FPS,
        'home_joints_deg': list(HOME_JOINTS_DEG),
        'phase_sequence': ['home', 'approach', 'gripper_roll_90_deg', 'lower_20_mm', 'align_rim_axis', 'ready_standoff'],
        'plate_xy_m': [1.10, 0.0],
        'target_xy_m': [1.10, -0.45],
        'turn_tcp_m': p_turn.tolist(),
        'ready_tcp_m': p_aligned.tolist(),
        'lower_after_roll_m': 0.02,
        'additional_axis_alignment_lower_m': float(p_lower[2] - p_aligned[2]),
        'aligned_rim_point_m': p_rim.tolist(),
        'rim_distance_along_tool_axis_m': rim_distance_on_ray,
        'roll_joint': 6,
        'roll_degrees': 90,
        'ready_tool_z': after[:3, 2].tolist(),
        'ready_finger_closing_axis': after[:3, 0].tolist(),
        'frame_clearance_min_m': float(min_frame),
        'table_clearance_min_m': float(min_table),
        'note': 'Open gripper; joint 6 rolls, TCP lowers until its pinch-centre ray meets the model rim. Ends before rim contact. Plate, support box, and real gripper geometry need verification.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, joints_rad=q, phases=np.asarray(phases), fps=FPS)
    args.out.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'file': str(args.out), 'frames': len(q), **metadata}, indent=2))


if __name__ == '__main__':
    main()
