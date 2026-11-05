"""Apply the predeclared heldout gate; do not read official test results."""
import argparse
import json
from pathlib import Path

import numpy as np


def paired_normals(control, candidate):
    deltas, recovered = [], []
    shared_count, recovered_count = 0, 0
    control_paths=sorted((control/'support').glob('*.npz'))
    assert control_paths and [p.name for p in control_paths]==sorted(p.name for p in (candidate/'support').glob('*.npz'))
    for p in control_paths:
        with np.load(p) as base,np.load(candidate/'support'/p.name) as new:
            assert str(base['gt_sha256'])==str(new['gt_sha256']) and str(base['source_frame'])==str(new['source_frame'])
            e0,e1=base['normal_error'],new['normal_error']
            shared=np.isfinite(e0)&np.isfinite(e1)
            added=np.isfinite(e1)&~base['coverage']
            if shared.any():
                deltas.append(float((e1[shared]-e0[shared]).mean()))
                shared_count+=int(shared.sum())
            if added.any():
                recovered.append(float(e1[added].mean()))
                recovered_count+=int(added.sum())
    return dict(delta=float(np.mean(deltas)) if deltas else None,shared_pixels=shared_count,
                recovered_mae=float(np.mean(recovered)) if recovered else None,recovered_pixels=recovered_count)


def main():
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); a=p.parse_args()
    root=a.root/'evaluation/probe'
    output={}
    fields=('boundary_f1','boundary_f2','boundary_f4','assd_px','hd95_px','missing','extra','silhouette_iou','psnr','ssim','lpips','boundary_rgb_rmse')
    for method in ('v38','region','edge','combined'):
        cases={}
        for scene in ('car','coffee'):
            control,candidate=root/'control'/scene,root/method/scene
            b=json.loads((control/'evaluation.json').read_text()); c=json.loads((candidate/'evaluation.json').read_text())
            assert b['split']==c['split']=='heldout' and b['count']==c['count']==13 and not b['limited'] and not c['limited']
            assert [(v['source_id'],v['source_frame'],v['gt_sha256']) for v in b['views']]==[(v['source_id'],v['source_frame'],v['gt_sha256']) for v in c['views']]
            d={k:c['metrics'][k]-b['metrics'][k] for k in fields}
            normal=paired_normals(control,candidate)
            gate=d['boundary_f2']>=-.001 and d['assd_px']<=.05 and d['missing']<=.001 and d['extra']<=.001 and d['psnr']>=-.05 and d['lpips']<=.001 and normal['delta'] is not None and normal['delta']<=.1
            cases[scene]=dict(delta=d,paired_normal=normal,guard_pass=gate,
                              material_boundary_gain=d['boundary_f2']>=.002 or d['assd_px']<=-.05)
        output[method]=dict(cases=cases,pass_gate=all(x['guard_pass'] for x in cases.values()) and any(x['material_boundary_gain'] for x in cases.values()))
    passed=[m for m in ('region','edge','combined') if output[m]['pass_gate']]
    # Prioritize better equal-scene F2, then closer contours, then LPIPS.
    def rank(method):
        cases=output[method]['cases'].values()
        return (sum(x['delta']['boundary_f2'] for x in cases),-sum(x['delta']['assd_px'] for x in cases),-sum(x['delta']['lpips'] for x in cases))
    selected=max(passed,key=rank) if passed else None
    result=dict(selected=selected,results=output,note='Training-heldout only; existing V32 checkpoint previously saw these views.')
    (a.root/'probe_selection.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    for m,v in output.items():
        print(m,v['pass_gate'],[(s,round(c['delta']['boundary_f2'],6),round(c['delta']['assd_px'],4),round(c['delta']['lpips'],6),c['paired_normal']['delta']) for s,c in v['cases'].items()])
    print('SELECTED',selected)


if __name__=='__main__': main()
