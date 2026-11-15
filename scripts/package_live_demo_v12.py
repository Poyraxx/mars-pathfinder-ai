"""Portable, hash-checked labels for all demo frames; never a runtime lookup."""
import gzip,hashlib,json,os,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training/live-demo-v12'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    source=T/'manifest.json';m=json.loads(source.read_text('utf8'));out=R/'data/annotations/live-demo-v12-masks.zip';entries=[]
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for e in m['entries']:
            row={k:v for k,v in e.items() if k not in ['image','xyz_path','label','semantic_label','weight']}
            row['image_relative']=Path(e['image']).relative_to(R/'data/prepared').as_posix();row['xyz_relative']=Path(e['xyz_path']).relative_to(R/'data/prepared').as_posix()
            for key in ['label','semantic_label','weight']:
                p=Path(e[key]);name=f"{e['index']:03}/{p.name}";info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16;z.writestr(info,p.read_bytes());row[key+'_member']=name
            entries.append(row)
    with zipfile.ZipFile(out) as z:
        for e in entries:
            for key,h in [('label','label_sha256'),('semantic_label','semantic_label_sha256'),('weight','weight_sha256')]:assert hashlib.sha256(z.read(e[key+'_member'])).hexdigest()==e[h]
    portable={**{k:v for k,v in m.items() if k!='entries'},'local_manifest_sha256':sha(source),'label_archive_sha256':sha(out),'entries':entries}
    dest=R/'data/annotations/live-demo-v12-manifest.json.gz'
    with gzip.GzipFile(str(dest),'wb',mtime=0) as f:f.write(json.dumps(portable,indent=1).encode())
    report={'version':12,'frames':181,'masks':543,'archive_sha256':sha(out),'portable_manifest_sha256':sha(dest),'local_manifest_sha256':sha(source),'all_archive_members_verified':True,'archive_bytes':out.stat().st_size,'purpose':'Training/review labels only. The inference backend never reads this archive.'};(R/'data/annotations/live-demo-v12-package.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
