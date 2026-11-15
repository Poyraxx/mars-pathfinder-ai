"""Restore the exact committed labels using actual prepared source images."""
import argparse,gzip,hashlib,json,os,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training/live-demo-v12-restored');a=ap.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    with gzip.open(R/'data/annotations/live-demo-v12-manifest.json.gz','rt',encoding='utf8') as f:m=json.load(f)
    archive=R/'data/annotations/live-demo-v12-masks.zip';assert sha(archive)==m['label_archive_sha256'];entries=[]
    with zipfile.ZipFile(archive) as z:
        for e in m['entries']:
            row={**e,'image':str(R/'data/prepared'/e['image_relative']),'xyz_path':str(R/'data/prepared'/e['xyz_relative'])};assert sha(row['image'])==e['image_sha256'] and sha(row['xyz_path'])==e['xyz_sha256']
            for key,h in [('label','label_sha256'),('semantic_label','semantic_label_sha256'),('weight','weight_sha256')]:
                content=z.read(e[key+'_member']);assert hashlib.sha256(content).hexdigest()==e[h];dest=a.output/f"{e['index']:03}-{key}.png"
                if dest.exists():assert sha(dest)==e[h],'Refusing to overwrite a differing label'
                else:dest.write_bytes(content)
                row[key]=str(dest)
            entries.append(row)
    restored={**m,'entries':entries,'restored_from_portable_archive':True};(a.output/'manifest.json').write_text(json.dumps(restored,indent=2),encoding='utf8');print('Verified and restored181 sources and543 label/weight PNGs',flush=True)
if __name__=='__main__':main()
