"""Check both ONNX heads and unchanged terrain weights against actual inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
_dll=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
import torch
import onnxruntime as ort
from operational_model import OperationalSurface


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);args=ap.parse_args()
    settings=json.loads((args.candidate/'settings.json').read_text('utf8'))
    torch.set_num_threads(2);model=OperationalSurface(navigation_classes=len(settings.get('navigation_classes',[0,1,2,3])),spatial_refinement=settings.get('spatial_refinement',False),separate_navigation_encoder=settings.get('train_navigation_encoder',False)).eval()
    model.load_state_dict(torch.load(args.candidate/'best-operational.pt',map_location='cpu',weights_only=True))
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    path=args.candidate/'surface.onnx';session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider']);size=session.get_inputs()[0].shape[-1]
    entries=[e for e in json.loads(args.manifest.read_text('utf8'))['entries'] if e['split']=='validation'];checks=[]
    for entry in entries:
        source=Image.open(entry['image']).convert('RGB');image=np.asarray(source.resize((size,size),Image.Resampling.BILINEAR),dtype='float32')/255
        x=((image-[.485,.456,.406])/[.229,.224,.225]).transpose(2,0,1)[None].astype('float32')
        with torch.no_grad():reference=[v.numpy() for v in model(torch.from_numpy(x))]
        if settings.get('navigation_only'):reference=reference[1:]
        actual=session.run(None,{session.get_inputs()[0].name:x})
        for index,(expected,result) in enumerate(zip(reference,actual)):
            agreement=float((expected.argmax(1)==result.argmax(1)).mean())
            assert agreement>=.999, 'Export diverges from trained checkpoint'
            checks.append({'source_id':entry['source_id'],'head':'navigation' if settings.get('navigation_only') or index==1 else 'terrain','pixel_agreement':agreement,'max_abs_logit_error':float(np.max(np.abs(expected-result)))})
    report={'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'passed':True,'checks':checks}
    (args.candidate/'export-parity.json').write_text(json.dumps(report,indent=2),'utf8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
