"""Sparse FIVE-class appearance metrics; composite old-ground pixels are excluded.

These diagnostics are not full-image accuracy, physical passability or a fresh
test. The separate original four-class evaluation retains its historical truth.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
from PIL import Image
_dll=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
import onnxruntime as ort
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.surface_inference import predict_outputs
from train_operational_v10 import metrics
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--split',choices=['validation','consumed_control'],required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    session=ort.InferenceSession(str(args.candidate/'surface.onnx'),providers=['CPUExecutionProvider']);cm=np.zeros((5,5),np.int64);rows=[]
    for e in json.loads(args.manifest.read_text('utf8'))['entries']:
        if e['split']!=args.split:continue
        source=Image.open(e['image']).convert('RGB');truth=np.asarray(Image.open(e['label']));partial=np.asarray(Image.open(e['partial_ground_mask']),dtype=bool)
        assert hashlib.sha256(Path(e['label']).read_bytes()).hexdigest()==e['label_sha256']
        outputs,_=predict_outputs(session,source,'whole');p=outputs[0];assert p.shape[0]==5
        prediction=np.asarray(Image.fromarray(p.argmax(0).astype('uint8')).resize(source.size,Image.Resampling.NEAREST));valid=(truth<5)&~partial
        single=np.bincount(5*truth[valid]+prediction[valid],minlength=25).reshape(5,5);cm+=single
        rows.append({'source_id':e['source_id'],'sol':e['sol'],'metrics':metrics(single),'explicit_small_rock_pixels':int((truth==4).sum()),'explicit_bare_ground_pixels':int(((truth==0)&~partial).sum())})
    report={'split':args.split,'images':len(rows),'model_sha256':hashlib.sha256((args.candidate/'surface.onnx').read_bytes()).hexdigest(),'metrics':metrics(cm),'per_scene':rows,'limitations':'Strict sparse appearance labels only: untraced old ground is a composite0-or4 and excluded here. Not exhaustive foreground precision, independent expert labels, new unseen benchmark or physical traversability.'}
    args.output.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k!='per_scene'}),flush=True)
if __name__=='__main__':main()
