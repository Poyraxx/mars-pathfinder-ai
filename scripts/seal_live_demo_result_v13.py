"""Preserve the final actual weights, complement labels and checks separately."""
import hashlib,json,os,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=T/'backups/live-demo-v13-candidate-sealed';out.mkdir(exist_ok=True,parents=True);sources=[T/'v24-live-demo-v13'/n for n in ['navigation.onnx','best-navigation.pt','settings.json','candidate-report.json','paired-onnx-report.json','exclusion-onnx-conformance.json','cpu-runtime.json','release-visual-review.json']];sources += [R/'data/annotations'/n for n in ['live-demo-v13-exclusions.json','live-demo-v13-exclusion-masks.zip','live-demo-v13-protocol.json']];rows=[]
    for p in sources:
        q=out/p.name
        if q.exists():assert sha(q)==sha(p),'Refusing to alter a sealed candidate copy'
        else:shutil.copy2(p,q)
        rows.append({'file':q.name,'sha256':sha(q),'bytes':q.stat().st_size})
    original=T/'backups/live-demo-v12-original-sealed/navigation.onnx';assert sha(original)=='6dcf488232cc3d46cfe7250c1bbfe1311437c9f3f65555cd6858708602ec82b0';(out/'seal.json').write_text(json.dumps({'original_navigation_preserved':True,'files':rows},indent=2),encoding='utf8');print('Preserved original and corrected candidate, labels and reports',flush=True)
if __name__=='__main__':main()
