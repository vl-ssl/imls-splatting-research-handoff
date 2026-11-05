"""Matched Ref-NeRF boundary probes, then a fixed six-scene confirmation."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path('/root/autodl-tmp/opaque_boundary_20261105')
BASE=Path('/root/autodl-tmp')
CODE=ROOT/'code'
PY='/root/miniconda3/envs/imls/bin/python'
SCENES=('ball','car','coffee','helmet','teapot','toaster')
VARIANTS={'zero':(0.,0.,0.),'control':(0.,0.,0.),'v38':(0.,0.,.5),
          'region':(.02,0.,0.),'edge':(0.,.005,0.),'combined':(.02,.005,0.)}


def run(cmd,label):
    log=ROOT/'logs'/(label+'.log')
    log.parent.mkdir(exist_ok=True)
    if log.exists(): log=log.with_name(log.stem+f'.retry_{int(time.time())}.log')
    with log.open('x') as f:
        f.write('COMMAND '+json.dumps(cmd)+'\n'); f.flush()
        child=subprocess.Popen(cmd,cwd=CODE,stdout=f,stderr=subprocess.STDOUT)
        (ROOT/'status.json').write_text(json.dumps({'stage':label,'pid':child.pid,'log':str(log),'state':'running'}))
        status=child.wait()
    if status:
        raise RuntimeError(f'{label} exited {status}: {log}')


def train(scene,variant,phase,steps):
    out=ROOT/'models'/phase/variant/scene
    iterations=1 if variant=='zero' else steps+1
    ckpt=out/'point_cloud'/f'iteration_{iterations}'/'point_cloud.ply'
    if ckpt.exists() and (ckpt.parent/'ckpt.th').exists():
        return out,iterations
    if out.exists(): raise RuntimeError('Partial run must be preserved before retry: '+str(out))
    source=BASE/'output'/f'{scene}_v32_mask_geometry_candidate'/'point_cloud/iteration_1/point_cloud.ply'
    assert source.is_file() and (source.parent/'ckpt.th').is_file(),source
    sm=scene not in ('ball','teapot')
    grid=80 if scene=='ball' else 256
    region,edges,v38=VARIANTS[variant]
    cmd=[PY,'train.py','--eval','--white_background','--resolution','1','--SSAA','1',
        '--data_device','cpu','--grid_resolution',str(grid),'--meshscale','2.1',
        '--mesh_normal_weight','1','--normal_smooth_weight','0.25' if sm else '0',
        '--normal_smooth_iterations','2' if sm else '0','--mesh_vertex_smooth_weight','0',
        '--sdf_smooth_weight','0','--resample_until','0','--upsample_until','0',
        '--lambda_depth','0','--lambda_lpips','0','--reflection_grad_scale','0',
        '--feature_lr','0','--rendernet_lr','0','--residual_lr','0','--position_residual_lr','0',
        '--environment_lr','0','--reflection_map_lr','0','--reflection_detail_lr','0',
        '--reflection_normal_lr','0','--reflection_factor_lr','0','--freeze_base_rendernet',
        '--scaling_lr','0','--skip_final_render','--save_final_geometry_maps','0',
        '--save_final_depth_maps','0','--final_render_train','0','--geometry_only',
        '--iterations',str(iterations),'--start_checkpoint_ply',str(source),
        '-s',str(BASE/'refnerf'/scene),'-m',str(out),
        '--opaque_boundary_holdout_stride','8' if phase=='probe' else '0',
        '--opaque_boundary_max_displacement_voxels','0.75',
        '--position_lr_init','0.000005','--position_lr_final','0.000005','--normal_lr','0.0001',
        '--lambda_normal','0.005','--lambda_mask','0.1','--lambda_silhouette_iou','0.05',
        '--lambda_silhouette_recall','0.1','--lambda_silhouette_boundary','0.05',
        '--lambda_gt_normal_gradient','0.002','--lambda_point_anchor','0.01','--lambda_point_tangent','0.005',
        '--lambda_visible_contour',str(v38),'--lambda_opaque_boundary_region',str(region),
        '--lambda_opaque_boundary_edges',str(edges)]
    if region or edges: cmd+=['--opaque_boundary_audit_gradient']
    run(cmd,f'{phase}_{variant}_{scene}_train')
    assert ckpt.is_file() and (ckpt.parent/'ckpt.th').is_file(),ckpt
    return out,iterations


def evaluate(scene,variant,phase,out,iteration,reference=None):
    dest=ROOT/'evaluation'/phase/variant/scene
    result=dest/'evaluation.json'
    if result.exists(): return json.loads(result.read_text())
    cmd=[PY,str(ROOT/'scripts/evaluate_opaque_boundary.py'),'--code',str(CODE),'--model',str(out),
         '--iteration',str(iteration),'--data',str(BASE/'refnerf'/scene),'--output',str(dest),
         '--split','heldout' if phase=='probe' else 'test']
    if variant in ('zero','control'): cmd+=['--save-support']
    if reference: cmd+=['--reference',str(reference)]
    if phase=='full': cmd+=['--save-images']
    run(cmd,f'{phase}_{variant}_{scene}_evaluate')
    value=json.loads(result.read_text())
    if phase=='full': assert value['count']==200 and not value['limited'],value['count']
    return value


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('probe','domain-audit','full'),required=True)
    p.add_argument('--variant',choices=('region','edge','combined'))
    p.add_argument('--steps',type=int,default=200)
    a=p.parse_args()
    assert a.steps>0
    refreal=Path('/root/autodl-tmp/refreal_runs_20261104/report/status.json')
    assert json.loads(refreal.read_text())['state']=='complete','RefReal gate still open'
    if a.phase=='full' and not a.variant: p.error('A heldout-selected candidate is required')
    if a.phase=='full':
        result=json.loads((ROOT/'probe_selection.json').read_text())
        if result['selected'] != a.variant or not result['results'][a.variant]['pass_gate']:
            p.error('Requested candidate did not pass the predeclared heldout gate')
    lock=(ROOT/'experiment.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if gpu: raise RuntimeError('GPU already occupied: '+gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',CUDA_HOME='/usr/local/cuda',OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',IMLS_RENDER_CHUNK='32768',
                      PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                      PATH='/root/miniconda3/envs/imls/bin:/usr/local/cuda/bin:/usr/bin:/bin')
    summary={}
    if a.phase=='domain-audit':
        for scene in ('car','coffee'):
            dest=ROOT/'evaluation/domain_audit'/scene
            model=ROOT/'models/probe/zero'/scene
            reference=ROOT/'evaluation/probe/zero'/scene
            run([PY,str(ROOT/'scripts/evaluate_opaque_boundary.py'),'--code',str(CODE),'--model',str(model),
                 '--iteration','1','--data',str(BASE/'refnerf'/scene),'--output',str(dest),
                 '--split','heldout','--aligned-domain-audit','--reference',str(reference),'--save-images'],
                f'domain_audit_{scene}')
            old=json.loads((reference/'evaluation.json').read_text())
            new=json.loads((dest/'evaluation.json').read_text())
            assert old['checkpoint_sha256']==new['checkpoint_sha256'] and old['count']==new['count']==13
            assert [(v['source_id'],v['gt_sha256']) for v in old['views']]==[(v['source_id'],v['gt_sha256']) for v in new['views']]
            summary[scene]={'domain':new['domain'], 'new':new['metrics'],
                            'delta':{k:new['metrics'][k]-old['metrics'][k] for k in old['metrics'] if k in new['metrics']}}
            (ROOT/'domain_audit_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        (ROOT/'status.json').write_text(json.dumps({'phase':a.phase,'state':'complete','finished':list(summary)}))
        return
    scenes=('car','coffee') if a.phase=='probe' else SCENES
    variants=list(VARIANTS) if a.phase=='probe' else ['zero','control',a.variant]
    for scene in scenes:
        reference=None
        scene_results={}
        for variant in variants:
            model,it=train(scene,variant,a.phase,a.steps)
            result=evaluate(scene,variant,a.phase,model,it,reference)
            if variant=='zero' or (a.phase=='full' and variant=='control'):
                reference=ROOT/'evaluation'/a.phase/variant/scene
            scene_results[variant]=result
            summary[f'{scene}/{variant}']=result['metrics']
            (ROOT/f'{a.phase}_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
            print('FINISHED',a.phase,scene,variant,result['count'],flush=True)
        if a.phase=='full':
            same_views=lambda report: [(v['source_id'],v['source_frame'],v['gt_sha256']) for v in report['views']]
            identities=same_views(scene_results['zero'])
            assert len(identities)==200 and len({i[0] for i in identities})==200
            assert all(same_views(result)==identities for result in scene_results.values())
    (ROOT/'status.json').write_text(json.dumps({'phase':a.phase,'state':'complete','finished':list(summary)}))


if __name__=='__main__':
    try: main()
    except Exception as e:
        (ROOT/'failure.txt').write_text(str(e))
        status=ROOT/'status.json'; s=json.loads(status.read_text()) if status.exists() else {}
        s.update(state='failed',error=str(e)); status.write_text(json.dumps(s)); raise
