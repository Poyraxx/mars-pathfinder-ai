"""Reproduce a validation-selected single navigation head without a runtime ensemble.

The coefficient is chosen on validation plus explicitly consumed training controls,
before a new test. Fresh test data must never be used to choose this argument.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
_dll=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--challenger',type=Path,required=True)
    ap.add_argument('--new-weight',type=float,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    assert 0<args.new_weight<1
    a=torch.load(args.baseline/'best-operational.pt',map_location='cpu',weights_only=True)
    b=torch.load(args.challenger/'best-operational.pt',map_location='cpu',weights_only=True)
    assert a.keys()==b.keys()
    assert all(torch.equal(a[k],b[k]) for k in a if k.startswith('base.')), 'Frozen terrain branches differ'
    merged={k:torch.lerp(a[k],b[k],args.new_weight) if a[k].is_floating_point() else a[k].clone() for k in a}
    assert all(torch.equal(a[k],merged[k]) for k in a if k.startswith('base.'))
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint=args.output/'best-operational.pt'
    torch.save(merged,checkpoint)
    settings=json.loads((args.baseline/'settings.json').read_text('utf-8'))
    member=json.loads((args.challenger/'settings.json').read_text('utf-8'))
    settings.update(output=str(args.output),manifest=member['manifest'],manifest_sha256=member['manifest_sha256'],
                    weight_mix_sources=[str(args.baseline),str(args.challenger)],weight_mix_new_weight=args.new_weight,
                    weight_mix_baseline_sha256=sha(args.baseline/'best-operational.pt'),
                    weight_mix_challenger_sha256=sha(args.challenger/'best-operational.pt'),
                    model_construction='Parameter mixture; no additional gradient updates',
                    trained_member_settings=member,
                    selection_split='validation; large-object controls are already consumed training scenes',
                    frozen_terrain_weights_unchanged=True,navigation_only=True)
    (args.output/'settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    print(json.dumps({'checkpoint_sha256':sha(checkpoint),'new_weight':args.new_weight,
                      'terrain_weights_unchanged':True,'fresh_test_evaluated':False}),flush=True)


if __name__=='__main__':
    main()
