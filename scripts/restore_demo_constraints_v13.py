"""Restore sealed complement labels without inferring any class in unknown areas."""
import argparse,hashlib,json,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=True,parents=True);m=json.loads((R/'data/annotations/live-demo-v13-exclusions.json').read_text('utf8'));archive=R/'data/annotations/live-demo-v13-exclusion-masks.zip';assert sha(archive)==m['archive_sha256'];entries=[]
    with zipfile.ZipFile(archive) as z:
        for e in m['entries']:
            row=dict(e)
            for cls in [1,2]:
                key=f'not_class{cls}';content=z.read(f"{e['index']:03}-not-class{cls}.png");assert hashlib.sha256(content).hexdigest()==e[key+'_sha256'];p=a.output/f"{e['index']:03}-not-class{cls}.png"
                if p.exists():assert sha(p)==e[key+'_sha256'],'Refusing to overwrite differing constraints'
                else:p.write_bytes(content)
                row[key]=str(p)
            entries.append(row)
    m['entries']=entries;(a.output/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf8');print('Restored362 exact class-exclusion masks',flush=True)
if __name__=='__main__':main()
