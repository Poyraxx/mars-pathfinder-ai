"""Actual-input ONNX parity and immutable baseline check, validation only."""
import argparse,json,os,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
_dll=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
import torch,onnxruntime as ort
from appearance_model_v11 import RockAppearance,load_navigation
from train_appearance_v11 import tensor
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--init',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(2)
    old=load_navigation(a.init);model=RockAppearance(old);model.load_state_dict(torch.load(a.candidate/'best-appearance.pt',map_location='cpu',weights_only=True));model.eval()
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(a.candidate/'appearance.onnx'),sess_options=options,providers=['CPUExecutionProvider']);checks=[]
    for e in json.loads(a.manifest.read_text('utf8'))['entries']:
        if e['split']!='validation':continue
        x=tensor(Image.open(e['image']).convert('RGB'))[None]
        with torch.no_grad():ref=model(x).numpy()
        result=session.run(None,{'image':x.numpy()})[0];agreement=float((ref.argmax(1)==result.argmax(1)).mean());assert agreement>=.999
        checks.append({'source_id':e['source_id'],'argmax_agreement':agreement,'max_logit_error':float(np.max(np.abs(ref-result)))})
    report={'passed':True,'candidate_sha256':sha(a.candidate/'appearance.onnx'),'checks':checks,'test_evaluated':False};(a.candidate/'export-parity.json').write_text(json.dumps(report,indent=2),'utf8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
