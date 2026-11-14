"""Candidate-only BatchNorm calibration on full TRAIN images, never heldouts.

Tiny crop batches can shift running statistics away from native full-frame
inference. Recompute only navigation normalization buffers; retain learned
weights and frozen terrain branch. Select/reject this variant on validation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
_dll=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
import torch
from operational_model import OperationalSurface
from train_operational_hazcam import state_sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(2);args.output.mkdir(exist_ok=True,parents=True)
    settings=json.loads((args.source/'settings.json').read_text('utf8'));size=settings['size']
    model=OperationalSurface(navigation_classes=4,spatial_refinement=settings.get('spatial_refinement',False),separate_navigation_encoder=settings.get('train_navigation_encoder',False)).eval()
    checkpoint=args.source/'best-operational.pt';model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    terrain_before=state_sha(model.base)
    norms=[m for m in model.navigation_head.modules() if isinstance(m,torch.nn.BatchNorm2d)]
    assert norms
    for norm in norms:norm.reset_running_stats();norm.momentum=None;norm.train()
    entries=[e for e in json.loads(args.manifest.read_text('utf8'))['entries'] if e['split']=='train']
    assert entries
    with torch.no_grad():
        for entry in entries:
            path=Path(entry['image']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['source_sha256']
            source=Image.open(path).convert('RGB');pixels=np.asarray(source.resize((size,size),Image.Resampling.BILINEAR),dtype='float32')/255
            x=((pixels-[.485,.456,.406])/[.229,.224,.225]).transpose(2,0,1)[None].astype('float32')
            model(torch.from_numpy(x))
    assert state_sha(model.base)==terrain_before
    torch.save(model.state_dict(),args.output/'best-operational.pt')
    settings.update(normalization_calibration='Full native training images only; cumulative BatchNorm statistics',calibration_source=str(args.source))
    (args.output/'settings.json').write_text(json.dumps(settings,indent=2),'utf8')
    report={'train_images':len(entries),'validation_or_test_images_used':0,'normalization_layers':len(norms),
            'source_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'checkpoint_sha256':hashlib.sha256((args.output/'best-operational.pt').read_bytes()).hexdigest(),
            'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),'frozen_terrain_state_sha256':terrain_before,
            'terrain_weights_unchanged':True,'candidate_only':True,'test_evaluated':False}
    (args.output/'normalization-report.json').write_text(json.dumps(report,indent=2),'utf8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
