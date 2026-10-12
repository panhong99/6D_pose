"""Bounded goal-1 simulation jobs. No real-robot/ROS/gripper commands; no shared config writes.

Stages run sequentially on one GPU and stop on a failed command. All outputs are candidates.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import yaml

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'rl/reports/20261008_goal1'
BASE='/home/pan/IsaacLab/logs/rsl_rl/m1013_2f85_ik_sweep_pw1_nofinger/2026-10-08_12-25-33_stage2_bc/model_0.pt'
ISAAC='/home/pan/IsaacLab/isaaclab.sh'
PYTHON='/home/pan/isaacsim-5.1.0/python.sh'


def run(name,cmd):
    log=ROOT/'rl/logs'/f'codex_goal1_{name}.log'
    env=os.environ.copy();env.pop('CONDA_PREFIX',None);env.pop('VIRTUAL_ENV',None);env['TERM']='xterm'
    state=REPORT/f'{name}_status.json'
    REPORT.mkdir(parents=True,exist_ok=True)
    with log.open('w') as f:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        state.write_text(json.dumps(dict(state='running',pid=process.pid,cmd=cmd,log=str(log),started=time.time()),indent=2))
        print(f'START {name} pid={process.pid} log={log}',flush=True)
        code=process.wait()
    state.write_text(json.dumps(dict(state='complete' if code==0 else 'failed',exit_code=code,log=str(log),ended=time.time()),indent=2))
    if code:
        raise RuntimeError(f'{name} failed ({code}); inspect {log}')
    print(f'FINISH {name}',flush=True)
    return log


def collect():
    bank=str(REPORT/'grasp_bank_train.npz')
    for name,noise,seed in [('clean',0,8101),('dart',.15,8102)]:
        output=str(ROOT/'IL'/f'data_goal1_variants_{name}_20261008')
        run(f'collect_{name}',[ISAAC,'-p','IL/collect.py','--headless','--episodes','200','--num_envs','64',
            '--max_batches','12','--seed',str(seed),'--grasp_bank',bank,'--bank_azimuth_deg','25',
            '--dart_noise',str(noise),'--output',output])
        run(f'verify_{name}',[ISAAC,'-p','IL/collect.py','--verify','--output',output])
        manifest=json.loads((Path(output)/'manifest.json').read_text())
        if manifest['status']!='complete':raise RuntimeError('Incomplete demonstrations; no BC training')


def heights():
    entries=[]
    for mm in (-8,-4,0,4,8):
        name=f'height_{mm:+d}_grid';output=REPORT/name
        run(name,[ISAAC,'-p','rl/eval.py','library','--headless','--checkpoint',BASE,'--approach','ik',
                  '--residual','off','--r_min','1.04','--r_max','1.32','--r_step','.02',
                  '--plate_dz',str(mm/1000),'--export_hold_s','5','--output',str(output)])
        info=json.loads((output/'library.json').read_text())
        entries.append(dict(plate_dz_m=mm/1000,plate_root_z_m=info['plate_root_z_m'],
                            library=str(output/'library.npz'),n_ok=info['n_ok'],n=info['n']))
        (REPORT/'height_index.json').write_text(json.dumps(dict(candidate_only=True,physical_validation_pending=True,
                                                               height_tolerance_m=.0025,entries=entries),indent=2))


def train():
    cfg=yaml.safe_load((ROOT/'rl/config.yaml').read_text())
    cfg['bc'].update(data=['../IL/data_wide2_claude','../IL/data_wide2_claude_dart',
        '../IL/data_ikwide_clean','../IL/data_ikwide_dart','../IL/data_goal1_variants_clean_20261008',
        '../IL/data_goal1_variants_dart_20261008'],ignore_obs=[[0,16],[16,18],[25,27],[31,34],[36,43]],
        phase_weight=1.0,steps=4000,template_checkpoint=BASE)
    cfg['residual']['enabled']=False
    cfg['algorithm']='BC'
    path=REPORT/'bc_candidate_config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    entries=[]
    for seed in (1,2,3):
        log=run(f'bc_seed{seed}',[PYTHON,'rl/train.py','--config',str(path),'bc',
            '--experiment',f'm1013_goal1_phi_pitch_slow_20261008_seed{seed}','--seed',str(seed)])
        marker=[line for line in log.read_text().splitlines() if line.startswith('BC_CHECKPOINT ')]
        if not marker:raise RuntimeError('Missing checkpoint marker')
        entries.append(dict(seed=seed,checkpoint=marker[-1].split(' ',1)[1]))
        (REPORT/'bc_candidates.json').write_text(json.dumps(entries,indent=2))


def gates(filter_seed=None):
    entries=[dict(seed=0,checkpoint=BASE,baseline=True)]+json.loads((REPORT/'bc_candidates.json').read_text())
    if filter_seed is not None:
        entries=[a for a in entries if a['seed']==filter_seed]
        result=json.loads((REPORT/'gate_comparison.json').read_text())
        result=[a for a in result if a['seed']!=filter_seed]
    else:
        result=[]
    for item in entries:
        name='gate_baseline' if item.get('baseline') else f"gate_seed{item['seed']}"
        log=run(name,[ISAAC,'-p','rl/eval.py','gate','--headless','--checkpoint',item['checkpoint'],
                      '--num_envs','1024','--approach','ik','--residual','off','--seed','8201',
                      '--grasp_bank',str(REPORT/'grasp_bank_holdout.npz'),'--bank_azimuth_deg','25',
                      '--dump',str(REPORT/(name+'.npz'))])
        values=[json.loads(x.split(' ',1)[1]) for x in log.read_text().splitlines() if x.startswith('EVAL_JSON ')]
        if not values:raise RuntimeError('Missing deterministic evaluation result')
        metrics=values[-1];result.append(dict(**item,metrics=metrics))
        (REPORT/'gate_comparison.json').write_text(json.dumps(result,indent=2))
        # Upload scalar evaluation metrics separately from training losses, never models/data.
        from train import start_tracking
        cfg=yaml.safe_load((REPORT/'bc_candidate_config.yaml').read_text())
        cfg['paths']['experiment']=f'm1013_goal1_phi_pitch_slow_20261008_seed{item["seed"]}'
        tracking=start_tracking(cfg,SimpleNamespace(stage=2,num_envs=1024),REPORT,
                                job_type='evaluation',role='gate',tag='holdout8201',checkpoint=item['checkpoint'])
        if tracking:
            scalar={f'Evaluation/{k}':v for k,v in metrics.items() if isinstance(v,(int,float)) and v is not None}
            tracking.log(dict(scalar,**{'Progress/iteration':0}))
            tracking.summary.update(scalar);tracking.finish()
    # Successful contact is necessary but not sufficient for the full home-to-object task.
    candidates=[a for a in result if not a.get('baseline')]
    best=max(candidates,key=lambda x:(x['metrics']['success'],-x['metrics']['collision']))
    (REPORT/'selected_candidate.json').write_text(json.dumps(dict(candidate=best,
        activated=False,needs_full_path_and_workspace_gate=True,needs_physical_validation=True),indent=2))


def gate3():
    gates(3)


def libraries():
    selected=json.loads((REPORT/'selected_candidate.json').read_text())['candidate']
    run('variant_library',[ISAAC,'-p','rl/eval.py','library','--headless','--checkpoint',selected['checkpoint'],
        '--approach','ik','--residual','off','--grasp_bank',str(REPORT/'grasp_bank_train.npz'),
        '--export_hold_s','5','--seed','8301','--output',str(REPORT/'variant_library')])


def qualify():
    gate3()
    heights()
    libraries()


def workspace_gate():
    selected=json.loads((REPORT/'selected_candidate.json').read_text())['candidate']
    log=run('gate_workspace',[ISAAC,'-p','rl/eval.py','gate','--headless','--checkpoint',selected['checkpoint'],
        '--num_envs','1024','--approach','ik','--residual','off','--seed','8402',
        '--grasp_bank',str(REPORT/'grasp_bank_workspace.npz'),'--bank_azimuth_deg','0',
        '--dump',str(REPORT/'gate_workspace.npz')])
    result=[json.loads(x.split(' ',1)[1]) for x in log.read_text().splitlines() if x.startswith('EVAL_JSON ')][-1]
    result['scope']='uniform Cartesian positions in frame intersect radius <=1.32m; selected phi=-10 pitch=-10; pregrasp contact gate'
    (REPORT/'workspace_gate.json').write_text(json.dumps(result,indent=2))


def variant_heights():
    selected=json.loads((REPORT/'selected_candidate.json').read_text())['candidate']
    zero=REPORT/'variant_library_dense'
    info=json.loads((zero/'library.json').read_text())
    entries=[dict(plate_dz_m=0,plate_root_z_m=info['plate_root_z_m'],library=str(zero/'library.npz'),n_ok=info['n_ok'],n=info['n'])]
    for mm in (-8,-4,4,8):
        name=f'variant_height_{mm:+d}';output=REPORT/name
        run(name,[ISAAC,'-p','rl/eval.py','library','--headless','--checkpoint',selected['checkpoint'],
            '--approach','ik','--residual','off','--grasp_bank',str(REPORT/'grasp_bank_train.npz'),
            '--plate_dz',str(mm/1000),'--export_hold_s','5','--output',str(output)])
        info=json.loads((output/'library.json').read_text())
        entries.append(dict(plate_dz_m=mm/1000,plate_root_z_m=info['plate_root_z_m'],library=str(output/'library.npz'),n_ok=info['n_ok'],n=info['n']))
        (REPORT/'variant_height_index.json').write_text(json.dumps(dict(candidate_only=True,physical_validation_pending=True,
            height_tolerance_m=.0025,entries=sorted(entries,key=lambda x:x['plate_dz_m'])),indent=2))


def selected_gate_and_extra():
    selected=json.loads((REPORT/'selected_candidate.json').read_text())['candidate']
    log=run('gate_workspace_selected',[ISAAC,'-p','rl/eval.py','gate','--headless','--checkpoint',selected['checkpoint'],
        '--approach','ik','--residual','off','--num_envs','1024','--bank_fixed','--seed','8402',
        '--grasp_bank',str(REPORT/'grasp_bank_workspace_selected.npz'),'--dump',str(REPORT/'gate_workspace_selected.npz')])
    result=[json.loads(x.split(' ',1)[1]) for x in log.read_text().splitlines() if x.startswith('EVAL_JSON ')][-1]
    result['scope']='1024 unique Cartesian targets; pose-clearance direction selection, pregrasp contact gate; validation after direction tuning'
    (REPORT/'workspace_gate_selected.json').write_text(json.dumps(result,indent=2))
    run('dense_extra_library',[ISAAC,'-p','rl/eval.py','library','--headless','--checkpoint',selected['checkpoint'],
        '--approach','ik','--residual','off','--grasp_bank',str(REPORT/'grasp_bank_dense_extra.npz'),
        '--export_hold_s','5','--output',str(REPORT/'variant_library_dense_extra')])


def workspace_refine():
    for name,noise,seed in [('clean',0,8601),('dart',.15,8602)]:
        output=str(ROOT/'IL'/f'data_goal1_workspace_{name}_20261008')
        run(f'workspace_collect_{name}',[ISAAC,'-p','IL/collect.py','--headless','--episodes','300','--num_envs','64',
            '--max_batches','16','--seed',str(seed),'--grasp_bank',str(REPORT/'grasp_bank_workspace_selected.npz'),
            '--dart_noise',str(noise),'--output',output])
        run(f'workspace_verify_{name}',[ISAAC,'-p','IL/collect.py','--verify','--output',output])
        if json.loads((Path(output)/'manifest.json').read_text())['status']!='complete':raise RuntimeError('Incomplete workspace data')
    cfg=yaml.safe_load((REPORT/'bc_candidate_config.yaml').read_text())
    cfg['bc']['data']+=['../IL/data_goal1_workspace_clean_20261008','../IL/data_goal1_workspace_dart_20261008']
    cfg['bc']['steps']=6000
    path=REPORT/'bc_workspace_config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    entries=[]
    for seed in (1,2,3):
        log=run(f'workspace_bc_seed{seed}',[PYTHON,'rl/train.py','--config',str(path),'bc',
            '--experiment',f'm1013_goal1_workspace_slow_20261008_seed{seed}','--seed',str(seed)])
        marker=[x for x in log.read_text().splitlines() if x.startswith('BC_CHECKPOINT ')]
        if not marker:raise RuntimeError('No BC checkpoint')
        checkpoint=marker[-1].split(' ',1)[1]
        log=run(f'workspace_val_seed{seed}',[ISAAC,'-p','rl/eval.py','gate','--headless','--checkpoint',checkpoint,
            '--approach','ik','--residual','off','--num_envs','1024','--bank_fixed','--seed','8603',
            '--grasp_bank',str(REPORT/'grasp_bank_workspace_selected.npz'),
            '--dump',str(REPORT/f'workspace_val_seed{seed}.npz')])
        metrics=[json.loads(x.split(' ',1)[1]) for x in log.read_text().splitlines() if x.startswith('EVAL_JSON ')][-1]
        entries.append(dict(seed=seed,checkpoint=checkpoint,metrics=metrics))
        (REPORT/'workspace_refine_candidates.json').write_text(json.dumps(entries,indent=2))


def workspace_scaled():
    cfg=yaml.safe_load((REPORT/'bc_workspace_config.yaml').read_text())
    cfg['bc'].update(obs_scale=[[18,19,20.],[27,28,20.],[43,46,20.]],
                     max_demo_tcp_speed_mps=.05,demo_speed_tolerance_mps=1e-5)
    path=REPORT/'bc_workspace_scaled_config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    entries=[]
    for seed in (1,2,3):
        log=run(f'workspace_scaled_bc_seed{seed}',[PYTHON,'rl/train.py','--config',str(path),'bc',
            '--experiment',f'm1013_goal1_workspace_scaled_slow_20261008_seed{seed}','--seed',str(seed)])
        checkpoint=[x.split(' ',1)[1] for x in log.read_text().splitlines() if x.startswith('BC_CHECKPOINT ')][-1]
        log=run(f'workspace_scaled_val_seed{seed}',[ISAAC,'-p','rl/eval.py','gate','--headless','--checkpoint',checkpoint,
            '--approach','ik','--residual','off','--num_envs','1024','--bank_fixed','--seed','8603',
            '--grasp_bank',str(REPORT/'grasp_bank_workspace_selected.npz'),
            '--dump',str(REPORT/f'workspace_scaled_val_seed{seed}.npz')])
        metrics=[json.loads(x.split(' ',1)[1]) for x in log.read_text().splitlines() if x.startswith('EVAL_JSON ')][-1]
        entries.append(dict(seed=seed,checkpoint=checkpoint,metrics=metrics))
        (REPORT/'workspace_scaled_candidates.json').write_text(json.dumps(entries,indent=2))


def highlift():
    """Bounded new low-speed, higher-hold-margin BC comparison; no existing checkpoint replacement."""
    clean=ROOT/'IL/data_goal1_highlift_clean_20261008'
    if json.loads((clean/'manifest.json').read_text())['status']!='complete':raise RuntimeError('Finish clean collection first')
    dart=ROOT/'IL/data_goal1_highlift_dart_20261008'
    bank=REPORT/'grasp_bank_workspace_flat.npz'
    run('highlift_dart',[ISAAC,'-p','IL/collect.py','--headless','--episodes','300','--num_envs','64',
        '--seed','8902','--max_batches','12','--output',str(dart),'--lift_m','.11','--total_s','30',
        '--speed_cap','.04','--hold_windup_m','.002','--arm_joint_speed_deg_s','6',
        '--grasp_bank',str(bank),'--dart_noise','.15'])
    if json.loads((dart/'manifest.json').read_text())['status']!='complete':raise RuntimeError('DART collection incomplete')
    for name,path in [('clean',clean),('dart',dart)]:
        run('highlift_verify_'+name,[PYTHON,'IL/collect.py','--verify','--output',str(path)])
    cfg=yaml.safe_load((REPORT/'bc_workspace_scaled_config.yaml').read_text())
    cfg['bc'].update(data=[str(clean),str(dart)],max_demo_arm_joint_speed_deg_s=6.,demo_joint_speed_tolerance_deg_s=.01)
    cfg['task']['episode_s'][2]=30.;cfg['action'].update(max_tcp_speed=.04,hold_windup=.002)
    path=REPORT/'bc_highlift_config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    entries=[]
    for seed in (1,2,3):
        log=run(f'highlift_bc_seed{seed}',[PYTHON,'rl/train.py','--config',str(path),'bc',
            '--experiment',f'm1013_goal1_highlift_slow_20261008_seed{seed}','--seed',str(seed)])
        checkpoint=[x.split(' ',1)[1] for x in log.read_text().splitlines() if x.startswith('BC_CHECKPOINT ')][-1]
        log=run(f'highlift_val_seed{seed}',[ISAAC,'-p','rl/eval.py','gate','--headless','--checkpoint',checkpoint,
            '--approach','ik','--residual','off','--num_envs','1024','--bank_fixed','--seed','8702',
            '--grasp_bank',str(bank),'--speed_cap','.04','--hold_windup_m','.002',
            '--arm_joint_speed_deg_s','6','--episode_s','30','--dump',str(REPORT/f'highlift_val_seed{seed}.npz')])
        metrics=[json.loads(x.split(' ',1)[1]) for x in log.read_text().splitlines() if x.startswith('EVAL_JSON ')][-1]
        entries.append(dict(seed=seed,checkpoint=checkpoint,metrics=metrics))
        (REPORT/'highlift_candidates.json').write_text(json.dumps(entries,indent=2))


def publish():
    """Publish completed comparisons, including failures; evaluation index is not a training curve."""
    from train import start_tracking
    entries=[]
    for log in sorted((ROOT/'rl/logs').glob('codex_goal1_*.log')):
        tagged=[x.split(' ',1)[1] for x in log.read_text(errors='replace').splitlines() if x.startswith('EVAL_JSON ')]
        if tagged:entries.append(dict(name=log.stem.removeprefix('codex_goal1_'),log=str(log),metrics=json.loads(tagged[-1])))
    cfg=yaml.safe_load((REPORT/'bc_workspace_scaled_config.yaml').read_text())
    cfg['algorithm']='BC';cfg['paths']['experiment']='m1013_goal1_offline_comparison_20261008'
    out=REPORT/'wandb_evaluations';out.mkdir(exist_ok=True)
    tracker=start_tracking(cfg,SimpleNamespace(stage=2,num_envs=1024),out,job_type='evaluation',role='comparison',tag='candidate_v1')
    if tracker is not None:
        tracker.config.update(dict(evaluation_order=[x['name'] for x in entries],physical_validated=False,
            warning='Repeated workspace sets are tuning validation. Candidate controller profiles differ; not learning progress.'),allow_val_change=True)
        for i,item in enumerate(entries):
            metrics=item['metrics']
            values={f'Eval/{k}':v for k,v in metrics.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
            values.update({'Progress/iteration':i,'Qualification/performance_pass':metrics['success']>=.95,
                           'Qualification/physical_validated':False})
            if 'speed_over_original_limit_episodes' in metrics:values['Qualification/tcp_speed_pass']=metrics['speed_over_original_limit_episodes']==0
            if 'arm_joint_over_6deg_episodes' in metrics:values['Qualification/joint_speed_pass']=metrics['arm_joint_over_6deg_episodes']==0
            tracker.log(values)
        url=tracker.url;tracker.finish()
    else:url=None
    (REPORT/'evaluation_summary.json').write_text(json.dumps(dict(candidate_only=True,physical_validated=False,
        wandb_url=url,unit='evaluation_set',entries=entries),indent=2))
    print('EVALUATION_REPORT',url,len(entries),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['collect','heights','train','gates','gate3','libraries','qualify','workspace_gate','variant_heights','selected_gate_and_extra','workspace_refine','workspace_scaled','highlift','publish'])
    a=p.parse_args();globals()[a.phase]()
