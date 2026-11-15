"""Actual ONNX evaluation of manually source-reviewed exclusion labels."""
import argparse,json
from pathlib import Path
from evaluate_live_demo_v12 import Runtime,sha,dump,torch
from train_demo_constraints_v13 import exclusions

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--constraints',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(2)
    entries=json.loads(a.manifest.read_text('utf8'))['entries'];cs=json.loads(a.constraints.read_text('utf8'));assert cs['base_manifest_sha256']==sha(a.manifest)
    for e,c in zip(entries,cs['entries']):
        assert e['image_sha256']==c['image_sha256'] and sha(e['image'])==e['image_sha256']
        for k in ['not_class1','not_class2']:assert sha(c[k])==c[k+'_sha256']
    result=exclusions(Runtime(a.model),entries,cs['entries'],'cpu');result.update(model_sha256=sha(a.model),manifest_sha256=sha(a.manifest),constraints_sha256=sha(a.constraints),not_independent_test=True);dump(a.output,result);print(json.dumps({k:v for k,v in result.items() if k!='frames'}),flush=True)
if __name__=='__main__':main()
