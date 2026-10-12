"""Offline A1: compare recorded grasp heights with the executed plan and a candidate library.

No hardware/ROS imports. The 165 mm TCP is a model assumption, not a physical calibration.
The gripper event is logged after closing; compare the preceding stationary state.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from robot_model import Kinematics

ROOT = Path(__file__).resolve().parents[1]


def analyze(records, library, tool_length=0.165):
    kin = Kinematics()
    lib = np.load(library, allow_pickle=False)
    cfg = yaml.safe_load((ROOT / 'rl/config.yaml').read_text())
    rest = cfg['geometry']['table_top_z'] + cfg['geometry']['box_h']
    good = np.flatnonzero(lib['ok'])
    rows, excluded = [], []

    def tcp(q):
        T = kin.fk(np.radians(q))
        return T[:3, 3] + T[:3, 2] * tool_length

    for file in sorted(records):
        events = [json.loads(x) for x in file.read_text().splitlines() if x.strip()]
        if not any(x.get('event') == 'lifted' for x in events):
            excluded.append(dict(run=file.parent.name, reason='no_lifted_event'))
            continue
        close = next((x for x in events if x.get('event') == 'gripper'
                      and x.get('where') == 'grasp' and x.get('choice') == 'c'), None)
        plan_event = next((x for x in events if x.get('event') == 'plan'), None)
        if close is None or plan_event is None:
            excluded.append(dict(run=file.parent.name, reason='missing_close_or_plan'))
            continue
        t = close['t']
        state = [x for x in events if x['kind'] == 'state' and 0 <= t - x['t'] <= 1]
        fp = [x for x in events if x['kind'] == 'fp' and 0 <= t - x['t'] <= 1]
        if not state or not fp:
            excluded.append(dict(run=file.parent.name, reason='missing_preclose_1s_samples'))
            continue
        state = max(state, key=lambda x: x['t'])
        if t - state['t'] > 0.25:
            excluded.append(dict(run=file.parent.name, reason='stale_joints', age_s=t-state['t']))
            continue
        plan_path = Path(plan_event['plan'])
        if not plan_path.exists():
            excluded.append(dict(run=file.parent.name, reason='executed_plan_missing'))
            continue
        plan = json.loads(plan_path.read_text())
        # Legacy exports have a separate grasp leg; libraries combine approach and grasp.
        leg = plan['legs'].get('grasp', plan['legs'].get('to_grasp'))
        q_sim = leg['waypoints_deg'][-1]
        sim_root = plan.get('cube_centre_m', [0, 0, rest])[2]
        p_close = np.median([x['plate_root'] for x in fp], axis=0)
        p_initial = np.asarray(plan_event['plate_root_fp'])
        r, az = np.hypot(*plan_event['rollout_plate']), np.arctan2(*plan_event['rollout_plate'][::-1])
        i = int(good[np.argmin(np.abs(lib['radius_m'][good] - r))])
        q_lib = lib['q_deg'][int(lib['close_step'][i]), i]
        real_z, sim_z, lib_z = tcp(state['posj'])[2], tcp(q_sim)[2], tcp(q_lib)[2]
        F = kin.fk(np.radians(state['posj']))
        rows.append(dict(run=file.parent.name, record=str(file), executed_plan=str(plan_path),
                         executed_checkpoint=plan.get('checkpoint'), executed_residual=plan.get('residual', {}).get('enabled'),
                         radius_m=float(r), azimuth_deg=float(np.degrees(az)),
                         tcp_model_z_mm=float(real_z*1000), fp_close_z_mm=float(p_close[2]*1000),
                         fp_initial_z_mm=float(p_initial[2]*1000), fp_close_samples=len(fp),
                         fp_window_span_mm=float(np.ptp([x['plate_root'][2] for x in fp])*1000),
                         joint_age_s=t-state['t'], fp_latest_age_s=t-max(x['t'] for x in fp),
                         delta_real_mm=float((real_z-p_close[2])*1000),
                         delta_executed_sim_mm=float((sim_z-sim_root)*1000),
                         delta_library_sim_mm=float((lib_z-rest)*1000),
                         executed_bias_mm=float((real_z-p_close[2]-sim_z+sim_root)*1000),
                         library_bias_mm=float((real_z-p_close[2]-lib_z+rest)*1000),
                         bias_using_initial_fp_mm=float((real_z-p_initial[2]-sim_z+sim_root)*1000),
                         fp_height_change_mm=float((p_close[2]-p_initial[2])*1000),
                         joint_plan_error_max_deg=float(np.max(np.abs(np.asarray(state['posj'])-q_sim))),
                         flange_fk_posx_error_mm=float(np.linalg.norm(F[:3, 3]-np.asarray(state['posx'][:3])/1000)*1000)))
    if not rows:
        raise RuntimeError('No usable lifted runs; cannot infer a height correction')
    summary = {}
    for key in ('executed_bias_mm', 'library_bias_mm', 'bias_using_initial_fp_mm', 'fp_height_change_mm'):
        x = np.array([r[key] for r in rows])
        summary[key] = dict(mean=float(x.mean()), std_sample=float(x.std(ddof=1)) if len(x)>1 else None,
                            min=float(x.min()), max=float(x.max()))
    return dict(tcp_assumption_m=tool_length, physical_tcp_calibrated=False, sim_rest_z_m=rest,
                library=str(library), successful_runs=len(rows), rows=rows, excluded=excluded, summary=summary,
                decision='Do not apply a fixed z correction. Preclose FP moves relative to initial FP, '
                         'and executions use different paths/policies from the candidate library. '
                         'Test a separate height-axis library using pre-approach FP; validate on 10/12 with Human.',
                limitations=['lifted is a recorded execution event, not independent physical success ground truth',
                             'gripper event time is after command completion, not first jaw contact',
                             'unmeasured adapter/pinch TCP and rim height prevent physical error attribution'])


def write_report(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    (output/'analysis.json').write_text(json.dumps(result, indent=2))
    rows = result['rows']
    header = '| Run | r (m) | az (°) | Δ real (mm) | Δ executed sim | Δ library sim | executed bias | FP z change |'
    table = [header, '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        table.append('| ' + ' | '.join([r['run'].split('_plate_')[0], f"{r['radius_m']:.3f}", f"{r['azimuth_deg']:.1f}",
                      *[f'{r[k]:.2f}' for k in ('delta_real_mm','delta_executed_sim_mm','delta_library_sim_mm',
                                                'executed_bias_mm','fp_height_change_mm')]]) + ' |')
    (output/'report.md').write_text('\n'.join(table)+'\n\n'+result['decision']+'\n\n'+
                                  json.dumps(result['summary'], indent=2)+'\n')
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(rows))
    for key,label in [('executed_bias_mm','Executed plan / preclose FP'),
                      ('library_bias_mm','Candidate library / preclose FP'),
                      ('bias_using_initial_fp_mm','Executed plan / initial FP')]:
        ax.plot(x, [r[key] for r in rows], 'o-', label=label)
    ax.set_xticks(x, [r['run'][9:15] for r in rows]); ax.set_xlabel('Run start (HHMMSS)'); ax.set_ylabel('Model height difference (mm)')
    ax.spines[['top','right']].set_visible(False); ax.grid(False)
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0,1.30))
    fig.tight_layout(); fig.savefig(output/'height_bias.png',dpi=180,bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, default=ROOT/'calibration/data/plan_library/20261008_133418/library.npz')
    p.add_argument('--output', type=Path, default=ROOT/'rl/reports/20261008_grasp_height')
    p.add_argument('--tool_length', type=float, default=0.165)
    a=p.parse_args()
    result=analyze((ROOT/'calibration/data/real_pipeline').glob('20261008_1[23]*_plate_*/robot_state.jsonl'), a.library,a.tool_length)
    write_report(result,a.output)
    print(json.dumps(dict(successful_runs=result['successful_runs'],summary=result['summary'],decision=result['decision']),indent=2))
