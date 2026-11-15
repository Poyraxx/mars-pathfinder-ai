"""Check actual validation images against the frozen checkpoint, not random noise."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
_dll=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
import torch
import onnxruntime as ort
from train_rock_refinement import make_model
from train_surface import MarsDataset


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--architecture',choices=['lraspp','segformer-b0'],required=True)
    ap.add_argument('--manifest',type=Path,required=True);args=ap.parse_args();torch.set_num_threads(2)
    model=make_model(args.architecture).eval();model.load_state_dict(torch.load(args.candidate/'best-surface.pt',map_location='cpu',weights_only=True))
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    path=args.candidate/'surface.onnx';session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
    entries=[e for e in json.loads(args.manifest.read_text('utf8'))['entries'] if e['split']=='validation']
    from PIL import Image
    rock=next(e for e in entries if (np.asarray(Image.open(e['label']))==3).any())
    dataset=MarsDataset([entries[0],entries[len(entries)//2],rock],size=session.get_inputs()[0].shape[-1]);checks=[]
    for image,_ in dataset:
        with torch.no_grad():reference=model(image[None])['out'].numpy()
        actual=session.run(None,{session.get_inputs()[0].name:image[None].numpy()})[0]
        agreement=float((reference.argmax(1)==actual.argmax(1)).mean())
        checks.append({'pixel_agreement':agreement,'max_abs_logit_error':float(np.max(np.abs(reference-actual)))})
        assert agreement>=.999, 'ONNX diverges from trained checkpoint'
    report={'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checks':checks,'passed':True}
    (args.candidate/'export-parity.json').write_text(json.dumps(report,indent=2),'utf8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
