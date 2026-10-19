import contextlib, csv, json, math, os, pathlib, sys, tempfile, time, multiprocessing as mp
import numpy as np
ROOT=pathlib.Path('/home/pan/pan/6D_pose'); CAL=ROOT/'calibration'; os.chdir(CAL)
sys.path.insert(0,str(CAL))
import plate_plans as pp
from calib_core import DATA, cpath, latest_t_base_cam, load_json

OUT=ROOT/'calibration/data/basket/stand_feasibility_20261020_BK_v7'
TBB_PATH=DATA/'basket/wrist_look/20261019_095350_Q3.json'
TBB=np.asarray(load_json(TBB_PATH)['T_base_basket'],float)
TBB[2,3]=0.144  # Boss-confirmed basket height; keep marker x/y/yaw.
TBC_PATH=latest_t_base_cam(); TBC=np.asarray(load_json(TBC_PATH)['T_base_cam'],float)
QRED={**pp.C['plate_rl'],**pp.C['plate_ik_bc']}
QWHITE=dict(QRED); QWHITE['offset_m']=[QRED['offset_m'][0]+pp.C['white_plate']['grasp_deeper_m'],QRED['offset_m'][1]]
RLO,RHI=pp.library_ok_radii(QRED)
QMAP=pp.robot_quadrants(TBB)
sys.path.insert(0,str(ROOT/'rl'))
import basket_env as be

def region_pins(n,region,occupied=(),spacing=4):
    occupied=[int(v) for v in occupied]
    if region=='front': cand=list(range(n,n//2,-1))
    elif region=='back':
        top=min(occupied)-spacing if occupied else n//2
        cand=list(range(top,0,-1))
    else: raise ValueError(region)
    return [c for c in cand if all(abs(c-o)>=spacing for o in occupied)]

def fresh_grasp(root,q):
    radius,az=pp.library_target(root,q)
    if not RLO<=radius<=RHI:
        return None, radius, az, 'library range'
    pp.DATA=pathlib.Path(tempfile.mkdtemp(prefix='bk_v4_plan_'))
    try:
        p=pp.plan_from_library(radius,az,q,TBC,plate_z=float(root[2]))
        legs=load_json(p)['legs']
        return legs,radius,az,None
    except Exception as e:
        return None,radius,az,f'library grasp: {type(e).__name__}: {e}'

def classify(reason):
    s=str(reason).lower()
    if 'ik/limit' in s:return 'ik/limit'
    if 'insert clearance' in s:return 'insert clearance'
    if 'high carry' in s:return 'high carry'
    if 'straight' in s or 'rrt' in s or 'path' in s:return 'carry/path'
    return s.split(':')[0][:60] or 'unknown'

OCCUPANCY={'empty':[], 'q3_10':[('Q3',10)],
           'q3_12_8_q4_11':[('Q3',12),('Q3',8),('Q4',11)]}

def evaluate(root,qname,region,occupancy,q,grasp,lib_r,lib_az):
    q=dict(q)
    qd=QMAP[qname]; n=len(pp.basket_gaps()[qd]); occupied=[pin for qn,pin in OCCUPANCY[occupancy] if qn==qname]
    colour='white' if np.allclose(q['offset_m'],QWHITE['offset_m']) else 'red'
    lib_dir=cpath(q['basket_carry_library']) if q.get('basket_carry_library') else None
    lib_file=lib_dir/f'{qname}_{colour}.json' if lib_dir is not None else None
    if lib_file is not None and lib_file.is_file():
        q['_carry_prior']=load_json(lib_file)
    sq=pp.C['basket_quadrant_sim'][qd]
    sim_kw=dict(tilt_deg=sq['lean']*be.B['full']['tine_lean_deg'][qd],
                centre_h=sq['centre_h']+q.get('basket_quadrant_sim_real_dh_m',0.0),
                x_shift_m=sq['aim_m']+q.get('basket_quadrant_model_dx_m',{}).get(qd,0.0),
                y_extra_m=q.get('basket_quadrant_sim_real_dy_m',{}).get(qd,0.0))
    face_order=(-1.0,1.0) if sq['flip'] else (1.0,-1.0)
    cands=region_pins(n,region,occupied,int(q.get('basket_region_min_spacing',4)))
    H,tool=pp.hand_plate(np.asarray(grasp['to_grasp']['waypoints_deg'][-1],float),root)
    plates=[]
    plates=[(QMAP[qn],pp.pin_gap(TBB,QMAP[qn],pin)) for qn,pin in OCCUPANCY[occupancy]]
    first=None; reasons={}
    for faces in ((face_order[0],),face_order):
        for pin in cands:
            kw=dict(**sim_kw,y_shift_m=None,
                    yaw_trim_deg=q.get('basket_quadrant_yaw_trim_deg',0.0),
                    grasp_deg=grasp['to_grasp']['waypoints_deg'][-1],plates=plates,
                    grasp_low_z=float(root[2]),faces=faces)
            try:
                with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):
                    path,rep=pp.quadrant_plan(TBB,qd,pp.pin_gap(TBB,qd,pin),H,tool,
                       grasp['lift']['waypoints_deg'][-1],TBC,q,**kw)
            except Exception as e:
                return dict(result='exception',first_failure=f'{type(e).__name__}: {e}',candidate_pins=cands,success_pin=None)
            if path is not None:
                return dict(result='success',first_failure=None,candidate_pins=cands,success_pin=pin,
                            face=rep.get('face'),phi_deg=rep.get('phi_deg'),carry=rep.get('carry'),
                            clearance_ratio=rep.get('clearance_ratio'))
            for _,reason in rep.get('tried',[]):
                label=classify(reason); reasons[label]=reasons.get(label,0)+1
                if first is None:first=label
    return dict(result='no_plan',first_failure=first or 'no candidate pins',failure_counts=reasons,candidate_pins=cands,success_pin=None)

def run_point(name,root,q,scenarios):
    root=np.asarray(root,float); grasp,lr,la,err=fresh_grasp(root,q)
    print(f'{name}: root={root.tolist()} root_r={np.linalg.norm(root[:2]):.4f} root_az={math.degrees(math.atan2(root[1],root[0])):.2f} library_r={lr:.4f} library_az={math.degrees(la):.2f} bounds=[{RLO:.4f},{RHI:.4f}]',flush=True)
    if err:
        print(f'  grasp result={err}',flush=True); return []
    out=[]
    for qn,reg,occ in scenarios:
        a=evaluate(root,qn,reg,occ,q,grasp,lr,la)
        print(' ',qn,reg,'occupancy='+str(occ),a,flush=True);out.append((qn,reg,occ,a))
    return out

def evaluate_location(item):
    r,adeg,root_z=item; az=math.radians(adeg)
    root=[r*math.cos(az),r*math.sin(az),root_z]
    grasp,lr,la,gerr=fresh_grasp(root,QWHITE)
    rows=[]
    for qn in ('Q3','Q4'):
      for reg in ('front','back'):
       for occ in OCCUPANCY:
        base=dict(r_m=r,az_deg=adeg,quadrant=qn,region=reg,occupancy=occ,
                  library_target_r_m=lr,library_target_az_deg=math.degrees(la))
        if gerr:
            row=dict(base,result=gerr,first_failure=gerr,candidate_pins=[],success_pin=None)
        else:
            row=dict(base,**evaluate(root,qn,reg,occ,QWHITE,grasp,lr,la))
        rows.append(row)
    return rows

def plot_map(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'success':'#2e8b57','library range':'#7b3294','library grasp':'#e08214',
            'ik/limit':'#fdb863','insert clearance':'#d7191c','high carry':'#2c7bb6',
            'carry/path':'#abdda4','no candidate pins':'#777777','exception':'#000000'}
    labels={'success':'ok','library range':'range','library grasp':'grasp','ik/limit':'IK',
            'insert clearance':'insert','high carry':'carry','carry/path':'path',
            'no candidate pins':'none','exception':'err'}
    fig,axes=plt.subplots(2,6,figsize=(19,8),sharex=True,sharey=True)
    for ax,(qn,reg,occ) in zip(axes.flat,[(q,r,o) for q in ('Q3','Q4') for r in ('front','back') for o in OCCUPANCY]):
        sub=[x for x in rows if x['quadrant']==qn and x['region']==reg and x['occupancy']==occ]
        for row in sub:
            status=row['result']
            tag='success' if status=='success' else (status if status in colors else
                ('library grasp' if str(status).startswith('library grasp') else row.get('first_failure','exception')))
            if tag not in colors: tag='exception'
            ax.scatter(row['az_deg'],row['r_m'],s=145,marker='s',c=colors[tag],edgecolors='none')
            text=(f"{row['success_pin']}" if status=='success' else labels.get(tag,'err'))
            ax.text(row['az_deg'],row['r_m'],text,ha='center',va='center',fontsize=4,color='white' if tag not in ('ik/limit',) else 'black')
        ax.set_ylabel(f'{qn} {reg} | {occ}')
        ax.set_xticks(list(range(0,41,10))); ax.set_yticks([.8,.9,1.0,1.1,1.2,1.3])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_xlim(-2,42); ax.set_ylim(.78,1.32)
    for ax in axes[-1,:]:ax.set_xlabel('azimuth (deg)')
    from matplotlib.patches import Patch
    fig.legend([Patch(color=v,label=k) for k,v in colors.items()],list(colors),loc='lower center',ncol=5,frameon=False)
    fig.tight_layout(rect=[0,.09,1,1]); fig.savefig(path,dpi=180); plt.close(fig)

if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else 'pilot'
    # Exact red-root coordinates from the saved 11:45 grasp event.
    red_run=DATA/'real_pipeline/20261019_114554_plate_udj66qw7'
    red_events=[json.loads(s) for s in (red_run/'robot_state.jsonl').read_text().splitlines() if '"event"' in s]
    red_event=next(e for e in red_events if e['event']=='plan')
    red_root=red_event['plate_root_fp']
    red_scen=[('Q3','front','empty')]
    white_root=[1.276*math.cos(math.radians(18)),1.276*math.sin(math.radians(18)),red_root[2]]
    white_scen=[(qn,reg,occ) for qn in ('Q3','Q4') for reg in ('front','back') for occ in OCCUPANCY]
    if mode=='pilot':
        run_point('red_reference',red_root,QRED,red_scen)
        run_point('white_1130_taskboard_location',white_root,QWHITE,[('Q3','back','q3_10')])
    elif mode=='grid':
        if OUT.exists():raise SystemExit(f'refuse overwrite {OUT}')
        OUT.mkdir(parents=True)
        radii=[round(.80+i*.05,2) for i in range(11)]
        azimuths=list(range(0,41,5))
        cases=[(r,a,qn,reg,occ) for r in radii for a in azimuths for qn in ('Q3','Q4') for reg in ('front','back') for occ in OCCUPANCY]
        meta=dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),basket_pose_source=str(TBB_PATH.relative_to(ROOT)),T_base_basket=TBB.tolist(),
          camera_pose_source=str(TBC_PATH),T_base_cam=TBC.tolist(),library_radius_bounds_m=[RLO,RHI],
          red_reference_root=red_root,white_reference_root=white_root,white_plate_radial_extra_m=pp.C['white_plate']['grasp_deeper_m'],
          q_red_offset=QRED['offset_m'],q_white_offset=QWHITE['offset_m'],radii_m=radii,azimuth_deg=azimuths,
          qnames=['Q3','Q4'],regions=['front','back'],occupancies=OCCUPANCY,spacing=int(QWHITE.get('basket_region_min_spacing',4)),
          sim_q=pp.C['basket_quadrant_sim'],carry_library=QWHITE.get('basket_carry_library'),
          sim_real_dh_m=QWHITE.get('basket_quadrant_sim_real_dh_m',0.0),
          sim_real_dy_m=QWHITE.get('basket_quadrant_sim_real_dy_m',{}))
        (OUT/'metadata.json').write_text(json.dumps(meta,indent=2))
        records=[]; start=time.monotonic(); locations=[(r,a,float(red_root[2])) for r in radii for a in azimuths]
        with (OUT/'progress.jsonl').open('w') as f:
            ctx=mp.get_context('fork')
            with ctx.Pool(min(4,os.cpu_count() or 1),initializer=lambda:None) as pool:
                for i,group in enumerate(pool.imap_unordered(evaluate_location,locations,chunksize=1),1):
                    for row in group:f.write(json.dumps(row)+'\n')
                    f.flush(); records.extend(group)
                    print(f'progress {i}/{len(locations)} locations ({len(records)}/{len(cases)} cases) elapsed={time.monotonic()-start:.0f}s',flush=True)
        fields=['r_m','az_deg','quadrant','region','occupancy','library_target_r_m','library_target_az_deg','result','success_pin','face','phi_deg','carry','first_failure','failure_counts','candidate_pins','clearance_ratio']
        with (OUT/'feasibility.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for row in records:w.writerow({k:json.dumps(row.get(k),sort_keys=True) if isinstance(row.get(k),(dict,list)) else row.get(k) for k in fields})
        summary={}
        for qn in ('Q3','Q4'):
          for reg in ('front','back'):
            for occ in OCCUPANCY:
              sub=[x for x in records if x['quadrant']==qn and x['region']==reg and x['occupancy']==occ]
              counts={}
              for x in sub:
                tag='success' if x['result']=='success' else str(x['first_failure'])
                counts[tag]=counts.get(tag,0)+1
              summary[f'{qn}_{reg}_{occ}']={'total':len(sub),'success':sum(x['result']=='success' for x in sub),'failure_reasons':counts}
        (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
        plot_map(records,OUT/'feasibility_map.png')
        print('DONE',json.dumps(summary,sort_keys=True),flush=True)
    else:raise SystemExit('use pilot or grid')
