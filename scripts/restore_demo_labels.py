"""Restore versioned training-only demo labels with source and archive hashes."""
import argparse,gzip,hashlib,json,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--version',type=int,required=True,choices=[12,14]);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with gzip.open(R/f'data/annotations/live-demo-v{a.version}-manifest.json.gz','rt',encoding='utf8') as f:m=json.load(f)
    archive=R/f'data/annotations/live-demo-v{a.version}-masks.zip';assert sha(archive)==m['label_archive_sha256'];entries=[]
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==543
        for e in m['entries']:
            row={**e,'image':str(R/'data/prepared'/e['image_relative']),'xyz_path':str(R/'data/prepared'/e['xyz_relative'])}
            for key,h in [('image','image_sha256'),('xyz_path','xyz_sha256')]:assert sha(row[key])==e[h]
            for key in ['label','semantic_label','weight']:
                content=z.read(e[key+'_member']);assert hashlib.sha256(content).hexdigest()==e[key+'_sha256'];dest=a.output/f"{e['index']:03}-{key}.png"
                if dest.exists():assert sha(dest)==e[key+'_sha256'],'Refusing to overwrite a differing label'
                else:dest.write_bytes(content)
                row[key]=str(dest)
            entries.append(row)
    restored={**m,'entries':entries,'restored_from_portable_archive':True};(a.output/'manifest.json').write_text(json.dumps(restored,indent=2),encoding='utf8');print(json.dumps({'verified_sources':len(entries),'verified_label_files':3*len(entries),'version':a.version,'archive_sha256':sha(archive)}),flush=True)
if __name__=='__main__':main()
