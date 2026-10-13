"""Plate task plans that need no Isaac at run time (pipeline.py --task plate): the per-radius sim plan library
(any direction by turning joint 1) and the taught basket-slot paths (teach / replay).  Conda foundationpose Python."""
import json
import os
import socket
import subprocess
from pathlib import Path
import time

import numpy as np

from calib_core import DATA, cpath, load_config
from robot_model import HOME_JOINTS_DEG, Kinematics, polyline_path

C = load_config()
ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def split_long_moves(waypoints, max_deg):
    # Keep offline planning independent of pipeline's ROS/UDP initialization.
    out=[np.asarray(waypoints[0],dtype=float)]
    for b in waypoints[1:]:
        a,b=out[-1],np.asarray(b,dtype=float)
        n=max(1,int(np.ceil(np.degrees(np.abs(b-a).max())/max_deg-1e-9)))
        out += [a+(b-a)*k/n for k in range(1,n+1)]
    return out


def fresh_joints(msg, now=None):
    from pipeline import fresh_joints as fresh
    return fresh(msg, now)


# ================================================================ plan library (rl/eval.py library)

def library_target(p, q):
    """Plate root (FP) -> (radius m, azimuth rad) of the rollout point: FP + offset, the offset turned with the plate
    azimuth (it was measured with the plate straight ahead; the whole motion turns with joint 1)."""
    az = np.arctan2(p[1], p[0])
    c, s = np.cos(az), np.sin(az)
    ox, oy = q['offset_m']
    t = np.array([p[0] + c * ox - s * oy, p[1] + s * ox + c * oy])
    return float(np.hypot(*t)), float(np.arctan2(t[1], t[0]))


_CHECKS = {}


def library_checkers(Tbc, q):
    """FK clearance like the sim shield (rl/config.yaml shield): frame + D455 body >= frame margin, table >= table
    margin, with the 2F-85 as a tool cylinder.  Built once per process."""
    from robot_model import CollisionChecker, D455_BOX, frame_layout, measured_cell
    import yaml
    sh = yaml.safe_load((ROOT / 'rl' / 'config.yaml').read_text())['shield']
    cell=measured_cell()
    key=json.dumps(dict(camera=np.asarray(Tbc).tolist(),cell=cell,
                        tool_length=q.get('tool_length_m',.165),tool_radius=q.get('tool_radius_m',.06),
                        tool_model=q.get('tool_model','cylinder'),
                        frame_margin=sh['frame_margin'],table_margin=sh['table_margin']),sort_keys=True)
    if key not in _CHECKS:
        kin, boxes = Kinematics(), frame_layout(cell)
        frame = [b for b in boxes if b[0] != 'table'] + [('d455_body', np.asarray(Tbc)[:3, 3].tolist(), list(D455_BOX))]
        out = {}
        for name, bx, m in (('frame', frame, sh['frame_margin']), ('table', [b for b in boxes if b[0] == 'table'], sh['table_margin'])):
            c = CollisionChecker(kin, bx, margin=m)
            c.set_tool(q.get('tool_length_m', 0.165), q.get('tool_radius_m', 0.06))
            if q.get('tool_model')=='sim_open_gripper':
                import sys
                sys.path.insert(0,str(ROOT/'rl'))
                from rl_utils import gripper_surface_points, load_config as rl_config
                c.tool=gripper_surface_points(rl_config())
            elif q.get('tool_model','cylinder')!='cylinder':
                raise ValueError('Unknown library tool model')
            out[name] = (c, m)
        _CHECKS[key] = out
    return _CHECKS[key]


def library_ok_radii(q):
    lib = np.load(cpath(q['library']))
    r = lib['radius_m'][lib['ok']]
    return float(r.min()), float(r.max())


def height_library_config(q, plate_z):
    """Opt-in height index. Never extrapolate an untested height or silently fall back to dz=0."""
    if not q.get('library_height_index'):
        return q
    if plate_z is None or not np.isfinite(plate_z):
        raise RuntimeError('Height-axis library needs a finite pre-approach plate root z')
    index=json.loads(cpath(q['library_height_index']).read_text())
    entries=[e for e in index['entries'] if e['n_ok']>0]
    if not entries:
        raise RuntimeError('Height-axis index contains no successful candidate')
    entry=min(entries,key=lambda e:abs(e['plate_root_z_m']-plate_z))
    error=abs(entry['plate_root_z_m']-plate_z)
    if error>index.get('height_tolerance_m',.0025):
        raise RuntimeError(f'Plate z {plate_z:.4f} m outside tested library heights (nearest error {error*1000:.1f} mm)')
    return dict(q,library=entry['library'],selected_plate_z_m=entry['plate_root_z_m'],height_error_m=error)


def nearest_direction_candidates(lib,r,tolerance=.004):
    """Match the nearest successful radius separately for each phi/pitch, then rank directions."""
    good=np.flatnonzero(lib['ok'])
    good=good[np.abs(lib['radius_m'][good]-r)<=tolerance]
    keys=lib.files if hasattr(lib,'files') else lib
    phi=lib['phi_deg'] if 'phi_deg' in keys else np.zeros(len(lib['ok']))
    pitch=lib['pitch_deg'] if 'pitch_deg' in keys else np.zeros(len(lib['ok']))
    selected=[]
    for p,t in sorted(set(zip(phi[good],pitch[good]))):
        ids=good[(phi[good]==p)&(pitch[good]==t)]
        selected.append(ids[np.argmin(abs(lib['radius_m'][ids]-r))])
    return np.asarray(selected,dtype=int)


def _library_plan_candidate(lib,i,r,az,q,checks,plate_z):
    c, end = int(lib['close_step'][i]), int(lib['success_step'][i])
    Q = lib['q_deg'][:end + 1, i].astype(float).copy()
    Q[:, 0] += np.degrees(az)
    home = np.array(HOME_JOINTS_DEG, dtype=float)

    def thin(path, step):
        out = [path[0]]
        for w in path[1:]:
            if np.abs(w - out[-1]).max() >= step:
                out.append(w)
        if np.abs(path[-1] - out[-1]).max() > 1e-3:
            out.append(path[-1])
        return np.array(out)

    direct = np.degrees(split_long_moves([np.radians(home), np.radians(Q[0])], 20.0))   # ONE joint move, split for the mover
    grasp = thin(Q[:c + 1], 1.5)
    lift = thin(Q[c:], 1.5)
    legs = dict(to_grasp=np.concatenate([direct, grasp[1:]]), lift=lift, lower=lift[::-1],
                from_grasp=np.concatenate([grasp[::-1], direct[::-1][1:]]))
    plan = dict(dry_run=True, source='plan_library', library=str(cpath(q['library'])), library_radius_m=float(lib['radius_m'][i]),
                radius_m=r, azimuth_deg=float(np.degrees(az)), checkpoint=q['checkpoint'],
                created=time.strftime('%Y-%m-%d %H:%M:%S'), margin_m=0.0, gripper_force=64,
                cell_margins_m={k: m for k, (_, m) in checks.items()},
                tool_model=q.get('tool_model','cylinder'),
                measured_plate_z_m=plate_z, selected_plate_z_m=q.get('selected_plate_z_m'),
                height_error_m=q.get('height_error_m'),
                approach_phi_deg=float(lib['phi_deg'][i]) if 'phi_deg' in lib.files else 0.0,
                approach_pitch_deg=float(lib['pitch_deg'][i]) if 'pitch_deg' in lib.files else 0.0,
                gripper_close_after_leg='to_grasp', gripper_open_after_leg='lower', legs={}, reasons=[])
    for name, wp in legs.items():
        dense = polyline_path(np.radians(wp), 0.25)
        clear = {k: float(c.path_clearance(dense)) for k, (c, _) in checks.items()}
        # the mover compares min_clearance_straight_m with margin_m: store the clearance BEYOND each margin
        plan['legs'][name] = dict(waypoints_deg=np.round(wp, 3).tolist(), clearance_m=clear, planner='plan library',
                                  min_clearance_straight_m=min(clear[k] - m for k, (_, m) in checks.items()))
        for k, (_, m) in checks.items():
            if clear[k] < m:
                plan['reasons'].append(f'{name}: {clear[k] * 1000:.0f} mm from the {k} (< {m * 1000:.0f} mm)')
        print(f'   [library] {name:10s}: {len(wp):3d} waypoints, frame {clear["frame"] * 1000:4.0f} mm, '
              f'table {clear["table"] * 1000:4.0f} mm', flush=True)
    plan['accepted'] = not plan['reasons']
    return plan


def plan_from_library(r,az,q,Tbc,plate_z=None):
    """Nearest validated radius/height; choose the surviving phi/pitch with most cell clearance.
    One legacy radius path remains supported. No hardware commands."""
    import contextlib
    import io
    q=height_library_config(q,plate_z)
    lib=np.load(cpath(q['library']),allow_pickle=False)
    candidates=nearest_direction_candidates(lib,r,q.get('library_tol_m',.004))
    if not len(candidates):
        raise RuntimeError(f'radius {r:.3f} m has no successful library path within tolerance')
    checks=library_checkers(Tbc,q)
    plans=[]
    for i in candidates:
        with contextlib.redirect_stdout(io.StringIO()):
            plans.append(_library_plan_candidate(lib,int(i),r,az,q,checks,plate_z))
    valid=[p for p in plans if p['accepted']]
    def score(p):
        clearance=min(leg['min_clearance_straight_m'] for leg in p['legs'].values())
        return (clearance,-round(abs(p['library_radius_m']-r),9),
                -abs(p['approach_phi_deg']),-abs(p['approach_pitch_deg']))
    plan=max(valid or plans,key=score)
    plan['evaluated_candidates']=len(plans)
    for name,leg in plan['legs'].items():
        clear=leg['clearance_m']
        print(f'   [library] {name:10s}: {len(leg["waypoints_deg"]):3d} waypoints, '
              f'frame {clear["frame"]*1000:4.0f} mm, table {clear["table"]*1000:4.0f} mm',flush=True)
    out = DATA / 'rl_plans' / (time.strftime('%Y%m%d_%H%M%S') + f'_{int(time.time() * 1000) % 1000:03d}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'plan.json').write_text(json.dumps(plan, indent=1))
    if not plan['accepted']:
        raise RuntimeError('library plan too close to the cell: ' + '; '.join(plan['reasons']))
    return out / 'plan.json'



# ================================================================ basket slots: teaching + replay (2026-10-08)

TEACH_DIR = DATA / 'teach'
TEACH_POSES = (('carry', '운반 자세: 접시를 든 채 바구니 쪽으로 옮기기 좋은 높은 자세'),
               ('above', '슬롯 바로 위: 접시를 슬롯 방향으로 맞추고 핀보다 3~5 cm 위'),
               ('insert', '삽입 완료: 접시 바닥이 슬롯 바닥에 닿기 직전(핀 사이)'),
               ('retreat', '후퇴: 그리퍼를 연 뒤 접시를 건드리지 않고 빠져나온 자세'))


def next_slot():
    f = TEACH_DIR / 'next_slot.json'
    return json.loads(f.read_text())['slot'] if f.is_file() else 1


def set_next_slot(k):
    TEACH_DIR.mkdir(parents=True, exist_ok=True)
    (TEACH_DIR / 'next_slot.json').write_text(json.dumps(dict(slot=k, updated=time.strftime('%Y-%m-%d %H:%M:%S'))))


def teach_slot(k, rec, choose_gripper, lift_end_deg):
    """Human moves the robot with the pendant (jog / direct teaching); Enter stores the live joints + flange posx.
    The pipeline only READS joints here.  The gripper opens (Human choice) between `insert` and `retreat`."""
    print(f'\n=== 슬롯 {k} 티칭 ===\n펜던트로 로봇을 움직이세요 (이 동안 PC는 관절값만 읽습니다).\n'
          f'각 자세에 도착하면 Enter. 다시 찍으려면 r Enter, 그만두려면 q Enter.', flush=True)
    poses = {}
    for name, what in TEACH_POSES:
        if name == 'retreat':
            choose_gripper(f'slot{k}_release')
        while True:
            ans = input(f'  [{name}] {what} > ').strip().lower()
            if ans == 'q':
                raise RuntimeError('teaching cancelled')
            if ans == 'r':
                continue
            msg = rec.last
            if not fresh_joints(msg):
                print('  관절값이 오래됐습니다 (bringup 연결 확인). 다시 Enter.', flush=True)
                continue
            poses[name] = dict(posj=[float(v) for v in msg['posj']], posx=[float(v) for v in msg['posx']])
            rec.event('teach_pose', slot=k, pose=name, **poses[name])
            print(f'     저장: posj {np.round(msg["posj"], 1).tolist()}', flush=True)
            break
    TEACH_DIR.mkdir(parents=True, exist_ok=True)
    path = TEACH_DIR / f'slot_{k:02d}.json'
    if path.exists():
        path.replace(path.with_name(f'slot_{k:02d}_{time.strftime("%Y%m%d_%H%M%S")}.json'))   # keep older teachings
    path.write_text(json.dumps(dict(slot=k, created=time.strftime('%Y-%m-%d %H:%M:%S'), poses=poses,
                                    lift_end_deg=[float(v) for v in lift_end_deg], gripper_force=64), indent=1))
    print(f'슬롯 {k} 저장: {path}\n로봇은 펜던트로 홈에 돌려놓고, 자동 모드로 바꾼 뒤 run.sh connect 를 다시 실행하세요.', flush=True)
    return path


def taught_plan(points, Tbc, q, source):
    """Plan.json from taught joint keyposes: points = {leg: [q_deg, ...]}.  Each leg is densified on the straight
    joint lines between its keyposes (>= 2 deg steps, <= ~90 waypoints for the mover spline) so the smooth spline does
    not swing away from the taught path; FK frame clearance is checked (the basket is NOT in the cell model)."""
    def dense(pts, name):
        pts = [np.asarray(p, dtype=float) for p in pts]
        travel = sum(np.abs(b - a).max() for a, b in zip(pts[:-1], pts[1:]))
        # (10/13) 2-deg steps made the spline ~2x slower than the grasp leg (20-deg steps) at the same set speed: only the
        # plate's way down keeps them; every other leg is split like the grasp leg
        step = 2.0 if ('descend' in name or name in ('shift', 'to_policy_start')) else q.get('transport_step_deg', 20.0)
        # >= 3 moves per leg: a 1-waypoint movesj is accepted by the controller but does not move (10/12 real)
        return np.degrees(split_long_moves([np.radians(p) for p in pts], min(max(step, travel / 85), max(travel / 3, 1e-3))))

    checks = library_checkers(Tbc, q)
    plan = dict(dry_run=True, source=source, created=time.strftime('%Y-%m-%d %H:%M:%S'), margin_m=0.0, legs={}, reasons=[])
    for name, pts in points.items():
        wp = dense(pts, name)
        clear = float(checks['frame'][0].path_clearance(polyline_path(np.radians(wp), 0.25)))
        m = checks['frame'][1]
        plan['legs'][name] = dict(waypoints_deg=np.round(wp, 3).tolist(), clearance_m=dict(frame=clear),
                                  min_clearance_straight_m=clear - m, planner='taught')
        if clear < m:
            plan['reasons'].append(f'{name}: {clear * 1000:.0f} mm from the frame (< {m * 1000:.0f} mm)')
        print(f'   [{source}] {name}: {len(wp)} waypoints, frame {clear * 1000:.0f} mm', flush=True)
    plan['accepted'] = not plan['reasons']
    out = DATA / 'rl_plans' / (time.strftime('%Y%m%d_%H%M%S') + f'_{int(time.time() * 1000) % 1000:03d}_taught')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'plan.json').write_text(json.dumps(plan, indent=1))
    if not plan['accepted']:
        raise RuntimeError('taught plan refused: ' + '; '.join(plan['reasons']))
    return out / 'plan.json'


def slot_plan(k, lift_end_deg, Tbc, q):
    """Slot k after an automatic grasp: lift end -> carry -> above -> insert | insert -> retreat -> home."""
    t = json.loads((TEACH_DIR / f'slot_{k:02d}.json').read_text())['poses']
    P = {n: t[n]['posj'] for n in t}
    return taught_plan(dict(to_slot=[lift_end_deg, P['carry'], P['above'], P['insert']],
                            slot_out=[P['insert'], P['retreat'], list(HOME_JOINTS_DEG)]), Tbc, q, f'teach slot {k}')


# ---------------------------------------------------------------- whole task taught (pick at a fixed spot + slot)

FULL_POSES = (('pre_grasp', '잡기 전: 그리퍼 열린 채 접시 테두리 앞 5~10 cm'),
              ('grasp', '잡는 자세: 손가락 사이에 테두리가 들어온 위치'),
              ('lift', '들어 올린 자세: 접시가 받침에서 약 10 cm 위'),
              ('carry', '운반 자세: 바구니 쪽으로 옮기기 좋은 높은 자세'),
              ('above', '슬롯 바로 위: 접시를 슬롯 방향으로 세우고 핀보다 3~5 cm 위'),
              ('insert', '삽입: 바닥에서 약 1 cm 띄운 위치(핀 사이)'),
              ('retreat', '후퇴: 그리퍼를 연 뒤 접시를 건드리지 않고 빠져나온 자세'))
FULL_GRIPPER = {'lift': 'close', 'retreat': 'open'}      # gripper action BEFORE recording this pose


def next_full():
    f = TEACH_DIR / 'next_full.json'
    return json.loads(f.read_text())['slot'] if f.is_file() else 1


def set_next_full(k):
    TEACH_DIR.mkdir(parents=True, exist_ok=True)
    (TEACH_DIR / 'next_full.json').write_text(json.dumps(dict(slot=k, updated=time.strftime('%Y-%m-%d %H:%M:%S'))))


def teach_full(k, last_joints, choose_gripper, event):
    """Human jogs through FULL_POSES with the pendant; Enter stores the live joints (read only).  The plate must lie
    at the same marked spot whenever this teaching is replayed."""
    print(f'\n=== 전체 티칭 {k} (접시는 표시한 고정 위치) ===\n펜던트로 로봇을 움직이세요. PC는 관절값만 읽습니다.\n'
          f'각 자세에서 Enter, 다시 찍기 r, 그만두기 q.', flush=True)
    poses = {}
    for name, what in FULL_POSES:
        if name in FULL_GRIPPER:
            print(f'  그리퍼를 {"닫으세요 (c)" if FULL_GRIPPER[name] == "close" else "여세요 (o)"}', flush=True)
            choose_gripper(f'teach{k}_{FULL_GRIPPER[name]}')
        while True:
            ans = input(f'  [{name}] {what} > ').strip().lower()
            if ans == 'q':
                raise RuntimeError('teaching cancelled')
            if ans == 'r':
                continue
            msg = last_joints()
            if not fresh_joints(msg):
                print('  관절값이 오래됐습니다 (bringup 연결 확인). 다시 Enter.', flush=True)
                continue
            poses[name] = dict(posj=[float(v) for v in msg['posj']], posx=[float(v) for v in msg['posx']])
            event('teach_pose', slot=k, pose=name, **poses[name])
            print(f'     저장: posj {np.round(msg["posj"], 1).tolist()}', flush=True)
            break
    TEACH_DIR.mkdir(parents=True, exist_ok=True)
    path = TEACH_DIR / f'full_{k:02d}.json'
    if path.exists():
        path.replace(path.with_name(f'full_{k:02d}_{time.strftime("%Y%m%d_%H%M%S")}.json'))
    path.write_text(json.dumps(dict(slot=k, created=time.strftime('%Y-%m-%d %H:%M:%S'), poses=poses, gripper_force=64),
                               indent=1))
    print(f'저장: {path}\n펜던트로 홈 복귀 → 자동 모드 → run.sh connect 재실행 후 replay 하세요.', flush=True)
    return path


def full_plan(k, Tbc, q):
    t = json.loads((TEACH_DIR / f'full_{k:02d}.json').read_text())['poses']
    P = {n: t[n]['posj'] for n in t}
    home = list(HOME_JOINTS_DEG)
    return taught_plan(dict(to_grasp=[home, P['pre_grasp'], P['grasp']],
                            to_slot=[P['grasp'], P['lift'], P['carry'], P['above'], P['insert']],
                            slot_out=[P['insert'], P['retreat'], home]), Tbc, q, f'full teach {k}')


# ---------------------------------------------------------------- FP grasp + golden (taught) basket path

TCP_M = 0.165                                               # flange -> 2F-85 pinch point (rl/env_setup.py)


def _tcp(kin, q_deg):
    T = kin.fk(np.radians(q_deg)).copy()
    return T @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, TCP_M], [0, 0, 0, 1.0]])


def _plate_in_hand(T_tcp, rim_to_centre):
    """Gripper -> plate transform at the grasp: the plate lies flat (z up), its centre rim_to_centre ahead of the
    pinch point along the horizontal tool direction.  Same rule for the taught and the new grasp, so the residual
    model error mostly cancels in the correction."""
    d = T_tcp[:3, 2].copy()
    d[2] = 0.0
    d /= np.linalg.norm(d)
    P = np.eye(4)
    P[:3, 0], P[:3, 1], P[:3, 2] = d, np.cross([0, 0, 1.0], d), [0, 0, 1.0]
    P[:3, 3] = T_tcp[:3, 3] + d * rim_to_centre
    return np.linalg.inv(T_tcp) @ P


def golden_slot_plan(golden_path, grasp_deg, lift_end_deg, Tbc, q, basket_delta=None, plate_xy=None):
    """Plate grasped wherever FoundationPose saw it (plan to_grasp end = grasp_deg), then the taught basket path of
    the golden teaching: carry / above / insert / retreat re-solved so that the PLATE gets the taught poses although
    the new grasp (approach phi / pitch) differs:  T_tcp_new = D . T_tcp_taught . G_taught . G_new^-1  (IK seeded with
    the taught joints, refused if IK fails or a joint jumps).  D = basket_delta = T_basket_now . T_basket_at_teach^-1
    (markers, --place basket): the taught slot path moves with the basket; None = basket where it was taught."""
    lifts = q.get('basket_transit_lift_m', 0.0)
    if isinstance(lifts, (list, tuple)):    # (10/12 Human: carry higher) highest above-lift that the arm still reaches
        for i, lift in enumerate(lifts):
            try:
                out = golden_slot_plan(golden_path, grasp_deg, lift_end_deg, Tbc, q | {'basket_transit_lift_m': lift},
                                       basket_delta, plate_xy)
                print(f'   [golden] above lifted +{lift * 100:.0f} cm', flush=True)
                return out
            except RuntimeError as e:
                if i == len(lifts) - 1 or 'unreachable' not in str(e):
                    raise
    import yaml
    g = yaml.safe_load((ROOT / 'rl' / 'config.yaml').read_text())['geometry']
    rim_to_centre = g['rim_r'] - g['grasp_inset']
    t = json.loads(cpath(golden_path).read_text())['poses']
    kin = Kinematics()
    G_t = _plate_in_hand(_tcp(kin, t['grasp']['posj']), rim_to_centre)
    T_gn = _tcp(kin, grasp_deg)
    if plate_xy is None:
        G_n = _plate_in_hand(T_gn, rim_to_centre)
    else:   # (10/12 real) FoundationPose plate centre at the grasp: the library grasp is not radial (35 deg off the
        #     tool direction, 78 mm instead of 113 mm) -> the radial model put the plate 5 cm +y and 4.5 cm high
        d = np.r_[np.asarray(plate_xy, float) - T_gn[:2, 3], 0.0]
        d /= np.linalg.norm(d)
        F = np.eye(4)
        F[:3, 0], F[:3, 1], F[:3, 2] = d, np.cross([0, 0, 1.0], d), [0, 0, 1.0]
        F[:3, 3] = [plate_xy[0], plate_xy[1], T_gn[2, 3]]
        G_n = np.linalg.inv(T_gn) @ F
    # round plate: its turn about its own normal is free -> pick the one that keeps the tool closest to the taught tool
    # (else a non-radial grasp asks the wrist to turn ~35 deg for nothing)
    def turn(psi):
        R = np.eye(4)
        R[:2, :2] = [[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]
        return R
    cands = [G_t @ turn(psi) @ np.linalg.inv(G_n) for psi in np.radians(np.arange(-180, 180, 1.0))]
    corr = min(cands, key=lambda C_: np.arccos(np.clip((np.trace(C_[:3, :3]) - 1) / 2, -1, 1)))
    flange_from_tcp = np.linalg.inv(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, TCP_M], [0, 0, 0, 1.0]]))
    D = np.eye(4) if basket_delta is None else np.asarray(basket_delta, float)
    if basket_delta is None:    # (10/12) golden: the basket is where it was found today (basket_shift_m), not measured
        D[:3, 3] = q.get('basket_shift_m', [0.0, 0.0, 0.0])
    dyaw = float(np.degrees(np.arctan2(D[1, 0], D[0, 0])))
    # (10/12 real run) Human: carry / above 5-10 cm higher (the plate ticked on the basket going over), whole slot path
    # nudged in base (Human: a bit toward -y); both from config plate_ik_bc so they can be tuned without code
    lift_m = q.get('basket_transit_lift_m', 0.0)
    nudge = np.asarray(q.get('basket_nudge_m', [0.0, 0.0, 0.0]), float)
    if q.get('basket_tilt_deg', 0.0):     # (10/12 Human) tilted 15 deg the plate went in 2 pins (4 cm) toward the robot
        nudge = nudge + np.asarray(D[:3, 0]) * q.get('basket_tilt_x_comp_m', 0.0)   # along the basket x (pin rows)
    # extra turn of the whole slot path about the vertical through the (moved) taught plate centre at the insert
    a = np.radians(q.get('basket_yaw_deg', 0.0))
    c = (D @ _tcp(kin, t['insert']['posj']) @ G_t)[:3, 3] + nudge
    Rz = np.eye(4)
    Rz[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    Rz[:2, 3] = c[:2] - Rz[:2, :2] @ c[:2]
    # (10/12 Human) tilt the plate about the horizontal axis across the pin rows (basket y) through its lowest rim point
    # at the insert, so it slides in: + = the plate top toward +x (away from the robot)
    b = np.radians(q.get('basket_tilt_deg', 0.0))
    Pi = Rz @ (D @ _tcp(kin, t['insert']['posj']) @ G_t)
    Pi[:3, 3] += nudge + [0.0, 0.0, q.get('basket_insert_up_m', 0.0)]
    n_ = Pi[:3, 2]
    down = -(np.array([0, 0, 1.0]) - n_[2] * n_)
    low = Pi[:3, 3] + g['rim_r'] * down / np.linalg.norm(down)
    ax = D[:3, 1] / np.linalg.norm(D[:3, 1])
    K_ = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    Ry = np.eye(4)
    Ry[:3, :3] = np.eye(3) + np.sin(b) * K_ + (1 - np.cos(b)) * K_ @ K_
    Ry[:3, 3] = low - Ry[:3, :3] @ low
    new = {}
    for name in ('carry', 'above', 'insert', 'retreat'):
        if name == 'retreat' and q.get('basket_tilt_deg', 0.0) and q.get('basket_retreat_up_m'):
            # (10/12) tilted: the taught retreat (12 cm up) is out of reach -> straight up from the insert, same tool turn
            sol = None              # highest straight-up retreat the arm reaches (10/12: 8 cm failed with a closer grasp)
            for up in (q['basket_retreat_up_m'], 0.06, 0.04, 0.03):
                target = kin.fk(np.radians(new['insert'])).copy()
                target[2, 3] += up
                try:
                    sol = np.degrees(kin.ik(target, np.radians(new['insert'])))
                    break
                except RuntimeError:
                    continue
            if sol is None:
                raise RuntimeError('golden retreat: cannot even go 3 cm straight up from the insert (out of reach)')
            new[name] = sol
            print(f'   [golden] retreat  straight up {up * 100:.0f} cm from the insert', flush=True)
            continue
        seed = np.radians(t[name]['posj'])
        seed[0] += np.radians(dyaw + q.get('basket_yaw_deg', 0.0))   # a turned basket turns the taught pose about the base too
        target = D @ _tcp(kin, t[name]['posj']) @ corr
        target[:3, 3] += nudge + [0.0, 0.0, {'carry': q.get('basket_carry_lift_m', lift_m), 'above': lift_m}.get(name, 0.0)]
        if name == 'insert':    # (10/12 real, e-stop) Human: the taught depth forces the plate in -> stop higher
            target[2, 3] += q.get('basket_insert_up_m', 0.0)
        target = Rz @ target
        if name == 'insert':    # tilt only at the insert: the plate turns while it descends (above stays high and upright)
            target = Ry @ target
        if name == 'carry':     # flat plate high over the wall: keep the taught tool turn (corrected position only);
            target[:3, :3] = (D @ _tcp(kin, t[name]['posj']))[:3, :3]   # the wrist sits near j5 = 0 here and would swing
        target = target @ flange_from_tcp
        # seeds: the taught pose (j1 turned toward the moved target), then the previous solved pose (10/12: a basket
        # moved 5 cm left the single taught seed in a local minimum -> 'Unreachable IK target')
        tp = _tcp(kin, t[name]['posj'])[:2, 3]
        s1 = seed.copy()
        s1[0] += np.arctan2(target[1, 3], target[0, 3]) - np.arctan2(tp[1], tp[0]) - np.radians(dyaw + q.get('basket_yaw_deg', 0.0))
        sol = None
        for sd in [seed, s1] + ([np.radians(new[prev])] if (prev := {'above': 'carry', 'insert': 'above', 'retreat': 'insert'}.get(name)) in new else []):
            try:
                sol = np.degrees(kin.ik(target, sd))
                break
            except RuntimeError:
                continue
        if sol is None:         # (10/12) many random seeds before calling it unreachable; keep the one nearest the seed
            rng, lim = np.random.default_rng(0), np.radians(np.array(kin.limits_deg()))
            sols = []
            for _ in range(60):
                try:
                    sols.append(np.degrees(kin.ik(target, rng.uniform(np.maximum(lim[:, 0], -np.pi), np.minimum(lim[:, 1], np.pi)))))
                except RuntimeError:
                    pass
            if sols:
                sol = min(sols, key=lambda v: np.abs(v - np.degrees(seed)).max())
                print(f'   [golden] {name}: {len(sols)}/60 random seeds solve it (nearest kept)', flush=True)
        if sol is None:
            raise RuntimeError(f'golden {name}: unreachable (target TCP {np.round((target @ np.linalg.inv(flange_from_tcp))[:3, 3], 3).tolist()}, tool z {np.round(target[:3, 2], 2).tolist()}, flange {np.round(target[:3, 3], 3).tolist()}) - 바구니가 로봇에서 너무 멂: 로봇 쪽(-x)으로 옮기세요')
        err = np.linalg.norm(kin.fk(np.radians(sol))[:3, 3] - target[:3, 3])
        if err > 0.002:
            raise RuntimeError(f'golden {name}: IK misses the target by {err * 1000:.0f} mm (out of reach?)')
        if np.abs(sol - np.degrees(seed)).max() > 60:
            raise RuntimeError(f'golden {name}: corrected IK jumps {np.abs(sol - np.degrees(seed)).max():.0f} deg')
        new[name] = sol
        print(f'   [golden] {name:8s} corrected by {np.abs(sol - np.degrees(seed)).max():5.1f} deg (max joint)', flush=True)
    # (10/12 Human) four stages, each its own leg so the run can stop between them: 1 to_above, 2 descend_pre + descend
    # (+ open), 3 ascend, 4 to_home; abort_home = from above straight home (plate still held), pre_up = back to above
    # descend is split at 'pre' (TCP pre_up_m above the insert, plate bottom ~2.5 cm over the pin tops) for a look;
    # pre lies on the joint line above -> insert, which is straight in TCP within ~2 mm (10/12 check)
    pre_up = q.get('basket_pre_up_m', 0.075) - q.get('basket_insert_up_m', 0.0)   # pre stays where it was
    z_ins = _tcp(kin, new['insert'])[2, 3]
    lo, hi = 0.0, 1.0                                       # fraction from insert toward above
    for _ in range(40):
        mid = (lo + hi) / 2
        z = _tcp(kin, np.asarray(new['insert']) + mid * (np.asarray(new['above']) - new['insert']))[2, 3]
        lo, hi = (mid, hi) if z - z_ins < pre_up else (lo, mid)
    pre = list(np.asarray(new['insert']) + lo * (np.asarray(new['above']) - new['insert']))
    home = list(HOME_JOINTS_DEG)
    out = taught_plan(dict(to_above=[lift_end_deg, new['carry'], new['above']], descend_pre=[new['above'], pre],
                           descend=[pre, new['insert']], ascend=[new['insert'], new['retreat']],
                           to_home=[new['retreat'], home], abort_home=[new['above'], home],
                           pre_up=[pre, new['above']]), Tbc, q,
                      f'golden {cpath(golden_path).name}' + ('' if basket_delta is None else ' (basket moved)'))
    # for the pre-stop wrist measurement: plate centre y at the slot and the plate's lowest rim height at pre
    Ti = D @ _tcp(kin, t['insert']['posj']) @ G_t
    z_low = Ti[2, 3] - (g['rim_r'] - 0.0) * np.sqrt(max(0.0, 1 - Ti[2, 2] ** 2))
    plan = json.loads(out.read_text())
    plan['basket_slot'] = dict(plate_centre_y=float(c[1]),
                               # the plate is placed like the taught one, so its rim sits pre_up_m (+ z nudge) higher
                               plate_bottom_at_pre_z=float(z_low + nudge[2] + q.get('basket_pre_up_m', 0.075)))
    out.write_text(json.dumps(plan, indent=1))
    return out


def shifted_plan(plan_path, dx, Tbc, q):
    """(10/12 real) pre-stop correction: the same slot legs from 'pre' on, with the TCP moved dx (m) along base x
    (wrist photo: plate edge vs pin gap).  Legs: shift (pre -> pre'), descend, ascend, to_home, pre_up, abort_home."""
    kin = Kinematics()
    L = json.loads(Path(plan_path).read_text())['legs']
    pre, ins, ret = L['descend']['waypoints_deg'][0], L['descend']['waypoints_deg'][-1], L['ascend']['waypoints_deg'][-1]
    above = L['pre_up']['waypoints_deg'][-1]

    def moved(qd):
        T = kin.fk(np.radians(qd)).copy()
        T[0, 3] += dx
        sol = np.degrees(kin.ik(T, np.radians(qd)))
        if np.linalg.norm(kin.fk(np.radians(sol))[:3, 3] - T[:3, 3]) > 0.001 or np.abs(sol - qd).max() > 10:
            raise RuntimeError(f'shift {dx * 1000:+.1f} mm: IK failed near {np.round(qd, 1).tolist()}')
        return list(sol)
    p2, i2, r2 = moved(pre), moved(ins), moved(ret)
    home = list(HOME_JOINTS_DEG)
    return taught_plan(dict(shift=[pre, p2], descend=[p2, i2], ascend=[i2, r2], to_home=[r2, home],
                            pre_up=[p2, above], abort_home=[above, home]), Tbc, q,
                       f'{Path(plan_path).parent.name} shifted x {dx * 1000:+.1f} mm')


# ---------------------------------------------------------------- Isaac rollout export (moved from pipeline.py 10/12)
def isaac_cmd(q, mode):
    """rl/eval.py <mode> command line for the plate policy in q (export: one-shot, serve: warm server)."""
    cmd = [q['isaaclab'], '-p', str(ROOT / 'rl/eval.py'), mode, '--headless', '--checkpoint', q['checkpoint'],
           '--num_envs', str(q.get('export_envs', 16))]
    if q.get('residual_base'):                              # pin the BC base of this checkpoint (rl/config.yaml may change)
        cmd += ['--residual_base', q['residual_base']]
    else:
        cmd += ['--residual', 'off']
    if q.get('approach') == 'ik':                           # straight TCP line to the rim, BC from handoff_m on
        cmd += ['--approach', 'ik', '--handoff_m', str(q.get('handoff_m', 0.04)),
                '--approach_fast', str(q.get('approach_fast_sim', 1.0))]
    return cmd


def isaac_env():
    return {k: v for k, v in os.environ.items() if k not in ('CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'PYTHONPATH', 'PYTHONHOME')}


def export_from_server(x, y, yaw, q):
    """Ask a warm `pipeline.py --task plate --export_server` (rl/eval.py serve) first: seconds instead of ~1 min of Isaac
    start-up.  None when no server runs or it serves another policy."""
    req = dict(plate_x=x, plate_y=y, plate_yaw_deg=yaw, checkpoint=str(q['checkpoint']),
               approach=q.get('approach', 'policy'), residual_base=q.get('residual_base') or None)
    try:
        with socket.create_connection(('127.0.0.1', C['udp'].get('export_server', 5012)), timeout=2.0) as conn:
            conn.settimeout(300)
            conn.sendall((json.dumps(req) + '\n').encode())
            reply = json.loads(conn.makefile().readline())
    except (OSError, ValueError):
        return None
    if reply.get('mismatch'):
        print(f'   export server runs another policy ({reply["mismatch"]["server"]}); fresh Isaac instead', flush=True)
        return None
    for line in reply.get('lines', []):
        print('   ' + line, flush=True)
    if not reply.get('ok') or not reply.get('accepted'):
        raise RuntimeError(f'RL plan not created / not accepted (export server): {reply.get("error", reply.get("lines"))}')
    return Path(reply['plan'])


def export_plan(x, y, yaw, log, q):
    """Isaac rollout of the plate policy at the measured plate pose -> calibration/data/rl_plans/<time>/plan.json."""
    plan = export_from_server(x, y, yaw, q)
    if plan is not None:
        log.write_text(f'export server: {plan}\n')
        return plan
    cmd = isaac_cmd(q, 'export') + ['--plate_x', f'{x:.4f}', '--plate_y', f'{y:.4f}', '--plate_yaw_deg', f'{yaw:.1f}']
    for attempt in (1, 2):                                  # Isaac sometimes segfaults while starting (X11 / kit)
        with log.open('wb') as stream:
            subprocess.run(cmd, cwd=ROOT, env=isaac_env(), stdout=stream, stderr=subprocess.STDOUT, timeout=900)
        text = log.read_text(errors='replace')
        if 'EXPORT_PLAN ' in text or not ('Segmentation fault' in text or 'Crash detected' in text) or attempt == 2:
            break
        log.replace(log.with_name('export_crash.log'))
        print('   Isaac 시작 중 비정상 종료 -> 한 번 다시 실행합니다', flush=True)
    for line in text.splitlines():
        if line.startswith('[export]'):
            print('   ' + line, flush=True)
    found = [l for l in text.splitlines() if l.startswith('EXPORT_PLAN ')]
    if not found or not found[-1].endswith('accepted=True'):
        raise RuntimeError(f'RL plan not created / not accepted; see {log}')
    return Path(found[-1].split()[1])


def basket_range_note(Tbb, q):
    """(10/13) where the measured basket is vs the RL training range (config basket_policy_range)."""
    r, x, y = q.get('basket_policy_range', {}), float(Tbb[0][3]), float(Tbb[1][3])
    yaw = float(np.degrees(np.arctan2(Tbb[1][0], Tbb[0][0])))
    ok = lambda k, v: r.get(k) is None or r[k][0] <= v <= r[k][1]               # noqa: E731
    where, base = f'바구니 x {x:.3f} m, y {y:+.3f} m, 각도 {yaw:+.1f} deg', ok('x', x) and ok('yaw_deg', yaw)
    print(f'   [범위] {where}: 학습 범위 안' if base and ok('y', y) else
          f'   [범위 주의] {where}: 학습 범위 밖, sim 확인만 된 범위 -> sim 판정을 보고 진행' if base and ok('y_sim_checked', y) else
          f'   [범위 밖] {where}: 학습 범위 x {r.get("x")} y {r.get("y")} 각도 {r.get("yaw_deg")} 밖 -> sim 판정에서 떨어지면 '
          f'티칭 경로 (이 위치는 티칭 경로도 미검증)', flush=True)


def thin_waypoints(w, tol_deg=0.05):
    """(10/13) drop waypoints within tol_deg of the previous one: a spline with repeated points (the sim pause after the
    release) is accepted by the controller and not executed."""
    out = [list(w[0])]
    for p in w[1:]:
        if np.abs(np.subtract(p, out[-1])).max() > tol_deg:
            out.append(list(p))
    return out


def policy_plan(job, out, Tbc, q, wait_s=240):
    """(10/13) wait for the sim policy rollout (rl/basket/expert.py realplan), accept it only if the exact case inserted
    and >= half of the in-hand-noise variants did; -> plan.json (FK frame clearance checked) or None (golden fallback)."""
    t0 = time.time()
    while job.poll() is None and time.time() - t0 < wait_s:
        time.sleep(1)
    res_path = out / 'realplan.json'
    if job.poll() is None or not res_path.exists():
        print(f'   정책 경로 계산이 끝나지 않음 -> 티칭 경로로 넣기 ({out / "realplan.log"})', flush=True)
        return None
    r = json.loads(res_path.read_text())
    if r.get('reason'):
        print(f"   정책 경로: {r['reason']}", flush=True)
    print(f"   정책 sim 결과: 정확 {r['sim_success_exact']}, 변형 {r['sim_success_variants']:.2f}, 틈 mm {r['final_gap_mm'][:3]}", flush=True)
    if not r.get('legs') or not r['sim_success_exact'] or r['sim_success_variants'] < 0.5:
        print('   정책 경로 기준 미달 -> 티칭 경로로 넣기', flush=True)
        return None
    L = r['legs']
    start, down, up = L['to_policy_start']['waypoints_deg'], L['policy_descend']['waypoints_deg'], L['policy_ascend']['waypoints_deg']
    extra = q.get('basket_policy_extra_down_m', 0.0)      # (10/13 보스) seat a little deeper before the release: straight down
    if extra > 0:
        T = Kinematics().fk(np.radians(down[-1]))
        T[2, 3] -= extra
        down = down + [np.degrees(Kinematics().ik(T, np.radians(down[-1]))).round(3).tolist()]
    thin = thin_waypoints                                  # repeated waypoints: the controller does not move
    start, down, up = thin(start), thin(down), thin([down[-1]] + up)
    lo, hi = np.array(__import__('ros2_nodes').joint_limits_deg()).T          # (10/13) controller limits (J5 +-135, the sim rise went past):
    up = up[:next((i for i, w in enumerate(up) if not ((np.array(w) > lo + 3) & (np.array(w) < hi - 3)).all()), len(up))]   # rise while inside
    if len(up) < 3:
        print('   정책 상승 경로가 관절 한계에 걸림 -> 티칭 경로로 넣기', flush=True)
        return None
    last = up[-1]
    try:
        return taught_plan(dict(to_policy_start=start, policy_descend=down, policy_ascend=up,
                                to_home=[last, list(HOME_JOINTS_DEG)], policy_abort=[start[-1], start[0], list(HOME_JOINTS_DEG)]),
                           Tbc, q, 'learned policy (BC + residual RL)')
    except RuntimeError as e:
        print(f'   정책 경로 거부({e}) -> 티칭 경로로 넣기', flush=True)
        return None


def merged_plan(parts, name, max_wp=100):
    """(10/13 보스) one spline over consecutive legs of (possibly different) plan files with no stop between them:
    parts = [(plan_path, leg), ...] -> plan.json with the single leg `name`, or None (> max_wp waypoints for movesj)."""
    wp, slack = [], []
    for path, leg in parts:
        pl = json.loads(Path(path).read_text())
        if not pl.get('accepted') or leg not in pl['legs']:
            return None
        wp += pl['legs'][leg]['waypoints_deg']
        slack.append(pl['legs'][leg]['min_clearance_straight_m'] - pl.get('margin_m', 0.0))
    wp = thin_waypoints(wp)
    if len(wp) > max_wp or len(wp) < 3:
        return None
    plan = dict(dry_run=True, accepted=True, source='merged ' + ' + '.join(f'{lg}' for _, lg in parts), reasons=[], margin_m=0.0,
                created=time.strftime('%Y-%m-%d %H:%M:%S'), legs={name: dict(waypoints_deg=wp, min_clearance_straight_m=min(slack))})
    out = DATA / 'rl_plans' / (time.strftime('%Y%m%d_%H%M%S') + f'_{int(time.time() * 1000) % 1000:03d}_merged')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'plan.json').write_text(json.dumps(plan, indent=1))
    print(f'   [merged] {name}: {len(wp)} waypoints ({plan["source"]})', flush=True)
    return out / 'plan.json'
