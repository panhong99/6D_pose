"""(10/20 보스: "cuRobo가 경로 생성하는 과정을 GUI로") browser view of one cuRobo carry: the scene, the graph-planner
seeds, every optimizer iteration of every attempt, and the final path.  Offline: plans only, no robot.

    .envs/curobo-bm-20261020/bin/python experiments/curobo_bm/view.py <plan.json> [--standing Q4:6] [--port 8080]
    -> open http://localhost:8080 (Ctrl+C ends)

Same start/goal as the saved carry (lift end -> top of the straight-down insert), same boxes / plate spheres / floor cost
as calibration/curobo_carry.py.  store_debug records the iterations, which turns CUDA graphs off: planning takes seconds
here instead of ~0.2 s.  The scene is rebuilt in the foundationpose env (our checkers) by this file's --export mode."""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONDA_PY = Path.home() / 'miniconda3/envs/foundationpose/bin/python'


def export(plan, standing, out):
    """foundationpose env: the request curobo_carry.carry would send for this plan's carry."""
    sys.path[:0] = [str(ROOT / 'calibration'), str(Path(__file__).parent)]
    import curobo_carry as cc
    import scene
    from robot_model import DEFAULT_URDF
    sc = scene.build(plan, standing)
    q = sc['q']
    req = dict(start=sc['carry'][0].tolist(), goal=sc['carry'][-1].tolist(),
               cuboids=cc.cuboids(sc['held'], (q.get('basket_curobo', {}) or {}).get('grow', (0.5, 1.0, 1.0))),
               plate=cc.plate_spheres(sc['H'], sc['tool']),
               floor=dict(T_basket_inv=np.linalg.inv(sc['basket']).tolist(), half=sc['floor'].half.tolist(), z=sc['z_floor']),
               spheres=cc.robot_spheres(q), urdf=str(DEFAULT_URDF), saved=[c.tolist() for c in sc['carry']],
               T_basket=sc['basket'].tolist(), quadrant=sc['quadrant'])
    Path(out).write_text(json.dumps(req))


def plan_with_trace(req, attempts=5):
    """cuRobo as the pipeline's worker, plus a record of the graph seeds and each optimizer iteration."""
    import torch
    sys.path.insert(0, str(ROOT / 'calibration'))
    import curobo_worker as cw
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.scene import Cuboid, Scene
    from curobo.types import JointState
    cw.patch_floor_cost()
    folder = Path(tempfile.mkdtemp(prefix='curobo_view_'))
    planner = MotionPlanner(MotionPlannerCfg.create(
        robot=cw.robot_cfg(req['urdf'], req['spheres'], folder), collision_cache={'cuboid': 64},
        self_collision_check=False, optimizer_collision_activation_distance=0.05, store_debug=True))
    dev = planner.default_joint_state.position.device
    f32 = dict(device=dev, dtype=torch.float32)
    kp = planner.kinematics.config.kinematics_config
    Ti = torch.tensor(req['floor']['T_basket_inv'], **f32)
    cw.F.update(idx=kp.get_sphere_index_from_link_name('attached_object').to(dev), R=Ti[:3, :3].clone(),
                t=Ti[:3, 3].clone(), half=torch.tensor(req['floor']['half'], **f32),
                z=torch.tensor(float(req['floor']['z']), **f32), w=torch.tensor(cw.FLOOR_WEIGHT, **f32))
    planner.update_world(Scene(cuboid=[Cuboid(name=c['name'], dims=c['dims'], pose=c['pose']) for c in req['cuboids']]))
    start = JointState.from_position(torch.tensor([req['start']], **f32), joint_names=cw.JOINTS)
    goal = JointState.from_position(torch.tensor([req['goal']], **f32), joint_names=cw.JOINTS)
    planner.attachment_manager.update(torch.tensor(req['plate'], **f32), start, link_name='attached_object')

    stages = []                                       # (label, [paths (n, 6) rad])
    get_seeds, solve = planner._get_graph_seed_trajectories, planner.trajopt_solver.solve_cspace

    def seeds_spy(*a, **k):
        s = get_seeds(*a, **k)
        if s is not None:
            stages.append(('그래프 탐색 시작 경로', [p.cpu().numpy() for p in s.reshape(-1, s.shape[-2], s.shape[-1])]))
        return s

    def solve_spy(*a, **k):
        r = solve(*a, **k)
        n_try = 1 + sum(lbl.startswith('최적화 끝') for lbl, _ in stages)
        trace = (r.debug_info or {}).get('solver', {}).get('debug', [])
        for i, act in enumerate(trace):
            stages.append((f'시도 {n_try} 최적화 반복 {i + 1}/{len(trace)}', [p.cpu().numpy() for p in act]))
        pos = r.js_solution.position
        ok = r.success.reshape(-1).cpu().numpy()
        stages.append((f'최적화 끝 (시도 {n_try}): 성공 {int(ok.sum())}/{len(ok)}',
                       [p.cpu().numpy() for p in pos.reshape(-1, pos.shape[-2], pos.shape[-1])]))
        return r
    planner._get_graph_seed_trajectories, planner.trajopt_solver.solve_cspace = seeds_spy, solve_spy
    t = time.perf_counter()
    r = planner.plan_cspace(goal, start, max_attempts=attempts, enable_graph_attempt=1)
    took = time.perf_counter() - t
    final = None
    if r is not None and bool(r.success.any()):
        pos = r.js_solution.position
        final = pos.reshape(-1, pos.shape[-2], pos.shape[-1])[int(torch.nonzero(r.success.reshape(-1))[0])].cpu().numpy()
        stages.append(('최종 경로 (cuRobo)', [final]))
    return planner, stages, final, took


def main():
    a = argparse.ArgumentParser()
    a.add_argument('plan')
    a.add_argument('--standing', action='append', default=[], metavar='Q:PIN', help='plate already in the basket')
    a.add_argument('--port', type=int, default=8080)
    a.add_argument('--export', help=argparse.SUPPRESS)
    a = a.parse_args()
    if a.export:
        return export(a.plan, a.standing, a.export)
    req_path = Path(tempfile.mkstemp(suffix='.json')[1])
    cmd = [str(CONDA_PY), __file__, a.plan, '--export', str(req_path)] + sum((['--standing', s] for s in a.standing), [])
    subprocess.run(cmd, check=True, cwd=ROOT)
    req = json.loads(req_path.read_text())
    print('planning with the iteration record (CUDA graphs off, a few s) ...', flush=True)
    planner, stages, final, took = plan_with_trace(req)
    print(f'{len(stages)} stages, {"found" if final is not None else "NOT found"} in {took:.1f} s', flush=True)
    show(req, planner, stages, final, a.port)


def show(req, planner, stages, final, port):
    import torch
    import trimesh
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf
    from curobo.types import JointState
    sys.path.insert(0, str(ROOT / 'calibration'))
    import curobo_worker as cw
    dev = planner.default_joint_state.position.device

    def fk(paths):                                    # [(n, 6)] -> link_6 positions, robot spheres
        q = torch.tensor(np.concatenate(paths), device=dev, dtype=torch.float32)
        k = planner.compute_kinematics(JointState.from_position(q, joint_names=cw.JOINTS))
        p = k.tool_poses.get_link_pose('link_6').position.reshape(-1, 3).cpu().numpy()
        out, i = [], 0
        for path in paths:
            out.append(p[i:i + len(path)])
            i += len(path)
        return out, k.robot_spheres.reshape(len(q), -1, 4).cpu().numpy()

    server = viser.ViserServer(host='0.0.0.0', port=port)
    server.scene.set_up_direction('+z')
    share = Path(req['urdf']).parents[1]
    urdf = yourdfpy.URDF.load(req['urdf'], load_meshes=True, build_scene_graph=True,
                              filename_handler=lambda fname: str(share / fname.replace('package://dsr_description2/', '')))
    robot = ViserUrdf(server, urdf, root_node_name='/robot')
    for c in req['cuboids']:                          # obstacles (grown by the margin part cuRobo uses)
        name = c['name']
        col = (200, 120, 60) if 'plate_' in name else (90, 90, 200) if 'marker' in name else \
              (150, 110, 70) if 'wall' in name else (160, 160, 160)
        server.scene.add_box(f'/obstacles/{name}', dimensions=c['dims'], position=c['pose'][:3], wxyz=c['pose'][3:],
                             color=col, opacity=0.35)
    Tb = np.asarray(req['T_basket'])                  # floor area: plate spheres must stay above this slab
    hx, hy = req['floor']['half']
    slab = trimesh.creation.box(extents=[2 * hx, 2 * hy, 0.004])
    slab.apply_transform(Tb)
    slab.vertices[:, 2] = req['floor']['z']
    server.scene.add_mesh_simple('/floor_limit', vertices=slab.vertices, faces=slab.faces, color=(230, 60, 60), opacity=0.3)
    saved_tcp, _ = fk([np.asarray(req['saved'])])
    server.scene.add_spline_catmull_rom('/saved_carry', positions=saved_tcp[0], color=(120, 120, 120), line_width=2)

    tcp = [fk(paths)[0] for _, paths in stages]
    gui_stage = server.gui.add_slider('단계', min=0, max=max(0, len(stages) - 1), step=1, initial_value=max(0, len(stages) - 1))
    gui_label = server.gui.add_text('지금', initial_value='', disabled=True)
    gui_seed = server.gui.add_slider('보는 시드 (로봇)', min=0, max=7, step=1, initial_value=0)
    gui_t = server.gui.add_slider('경로 위치', min=0.0, max=1.0, step=0.01, initial_value=0.0)
    gui_sph = server.gui.add_checkbox('cuRobo 구 모델 보기', initial_value=False)
    gui_play = server.gui.add_button('단계 재생')
    server.gui.add_markdown('회색 선: 저장된 RRT 운반 / 색 선: 시드별 손끝 경로 / 빨간 판: 바구니 위 높이 제한 / '
                            '상자: cuRobo 장애물 (여유 포함)')
    colors = [(230, 80, 60), (60, 160, 230), (80, 200, 90), (220, 180, 40), (170, 90, 220), (40, 200, 200),
              (240, 120, 200), (120, 120, 40)]
    handles = []

    def draw_stage():
        for h in handles:
            h.remove()
        handles.clear()
        s = int(gui_stage.value)
        if not stages:
            gui_label.value = '기록 없음'
            return
        label, paths = stages[s]
        gui_label.value = label
        for i, line in enumerate(tcp[s]):
            handles.append(server.scene.add_spline_catmull_rom(f'/stage/seed{i}', positions=line, color=colors[i % 8],
                                                               line_width=4 if i == int(gui_seed.value) else 1.5))
        draw_robot()

    def draw_robot():
        s = int(gui_stage.value)
        if not stages:
            return
        paths = stages[s][1]
        path = paths[min(int(gui_seed.value), len(paths) - 1)]
        x = float(gui_t.value) * (len(path) - 1)
        i = min(int(x), len(path) - 2) if len(path) > 1 else 0
        q = path[i] + (path[min(i + 1, len(path) - 1)] - path[i]) * (x - i)
        robot.update_cfg(np.asarray(q, float))
        if gui_sph.value:
            _, sp = fk([q[None]])
            sp = sp[0][sp[0][:, 3] > 0]
            server.scene.add_point_cloud('/spheres', points=sp[:, :3], colors=(60, 200, 90), point_size=0.012)
        else:
            server.scene.add_point_cloud('/spheres', points=np.zeros((1, 3)), colors=(0, 0, 0), point_size=0.0)

    gui_stage.on_update(lambda _: draw_stage())
    gui_seed.on_update(lambda _: draw_stage())
    gui_t.on_update(lambda _: draw_robot())
    gui_sph.on_update(lambda _: draw_robot())

    @gui_play.on_click
    def _(_):
        for s in range(len(stages)):
            gui_stage.value = s
            time.sleep(0.05)
    draw_stage()
    print(f'open http://localhost:{port}  (Ctrl+C ends)', flush=True)
    while True:
        time.sleep(1.0)


if __name__ == '__main__':
    main()
