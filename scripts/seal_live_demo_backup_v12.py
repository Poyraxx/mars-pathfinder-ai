"""Non-destructive rollback copy and composition QA record for this dataset."""
import hashlib,json,os,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1]
T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    target=T/'backups/live-demo-v12-original-sealed';target.mkdir(exist_ok=True,parents=True)
    source=[R/'data/prepared'/n for n in ['navigation.onnx','navigation-model-card.json','surface.onnx','model-card.json']]
    source+=[T/'v11-navigation-512/best-operational.pt']
    source+=list((R/'data/annotations').glob('live-demo-v12-*'))
    rows=[]
    for p in source:
        q=target/p.name
        if q.exists():assert sha(q)==sha(p),f'Refusing to change sealed backup {q}'
        else:shutil.copy2(p,q)
        rows.append({'source':str(p),'backup':q.name,'sha256':sha(p),'bytes':p.stat().st_size})
    record={'purpose':'Immutable originals and reviewed v12 labels; candidate and deployment never overwrite these files.','files':rows};(target/'seal.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(json.dumps({'backup':str(target),'files':len(rows),'verified':True}),flush=True)
if __name__=='__main__':main()
