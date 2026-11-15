"""Audit and archive additive v14 labels; no production inference lookup."""
import gzip,hashlib,json,os,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training';V=T/'live-demo-v14'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    mp=V/'manifest.json';m=json.loads(mp.read_text());qa=json.loads((R/'data/annotations/live-demo-v14-core-qa.json').read_text());review=json.loads((R/'data/annotations/live-demo-v14-review.json').read_text());assert m['version']==14 and m['status']=='final_reviewed' and qa['complete'] and review['complete'];assert sha(R/'data/annotations/live-demo-v14-review.json')==m['v14_review_sha256'] and sha(R/'data/annotations/live-demo-v14-core-qa.json')==m['v14_qa_sha256']
    assert m['base_manifest_sha256']==sha(T/'live-demo-v12/manifest.json');expected={f for s in json.loads((R/'data/prepared/scenarios.json').read_text()) for f in s['frames']};assert {e['frame_id'] for e in m['entries']}==expected and len(expected)==181
    archive=R/'data/annotations/live-demo-v14-masks.zip';portable=[];counts=Counter();coverage=[];tiles=[];out=R/'artifacts/live-demo-v14-composition';out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for e in m['entries']:
            for key in ['image','xyz_path','label','semantic_label','weight']:assert sha(e[key])==e[{'xyz_path':'xyz_sha256'}.get(key,key+'_sha256')]
            y=np.asarray(Image.open(e['label']));s=np.asarray(Image.open(e['semantic_label']));w=np.asarray(Image.open(e['weight']));im=Image.open(e['image']).convert('RGB');assert y.shape==s.shape==w.shape==tuple(e['shape']) and im.size==y.shape[::-1];assert set(np.unique(y))<={0,1,2,3,255};assert (y[s==4]==255).all() and (w[y==255]==0).all() and (w[y<4]>0).all()
            for ob in e['objects']:
                x0,y0,x1,y1=ob['bbox'];assert (y[y0:y1,x0:x1]==1).sum()>=ob['pixels']>=3
            row={k:v for k,v in e.items() if k not in ['image','xyz_path','label','semantic_label','weight']};row['image_relative']=Path(e['image']).relative_to(R/'data/prepared').as_posix();row['xyz_relative']=Path(e['xyz_path']).relative_to(R/'data/prepared').as_posix()
            for key in ['label','semantic_label','weight']:
                p=Path(e[key]);member=f"{e['index']:03}/{p.name}";info=zipfile.ZipInfo(member,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes());row[key+'_member']=member
            portable.append(row);counts.update({str(c):int((y==c).sum()) for c in range(4)});coverage.append(float((y<4).mean()))
            src=np.asarray(im.resize((256,256)));lab=np.asarray(Image.fromarray(s).resize((256,256),Image.Resampling.NEAREST));marked=src.copy();palette=np.array([[40,210,90],[250,65,50],[55,150,250],[245,220,60],[225,80,220]],np.uint8);valid=lab<5;marked[valid]=(.65*marked[valid]+.35*palette[lab[valid]]).astype('uint8');tile=Image.new('RGB',(512,280),'#101820');tile.paste(Image.fromarray(src),(0,24));tile.paste(Image.fromarray(marked),(256,24));ImageDraw.Draw(tile).text((4,5),f"{e['index']:03} native-source overview / partial labels, dark = unknown",fill='white');tiles.append(tile)
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==543
        for e in portable:
            for key in ['label','semantic_label','weight']:assert hashlib.sha256(z.read(e[key+'_member'])).hexdigest()==e[key+'_sha256']
    dest=R/'data/annotations/live-demo-v14-manifest.json.gz'
    with gzip.GzipFile(str(dest),'wb',mtime=0) as f:f.write(json.dumps({**{k:v for k,v in m.items() if k!='entries'},'local_manifest_sha256':sha(mp),'label_archive_sha256':sha(archive),'entries':portable},indent=1).encode())
    for start in range(0,181,16):
        sheet=Image.new('RGB',(2048,1120),'#101820')
        for j,tile in enumerate(tiles[start:start+16]):sheet.paste(tile,((j%4)*512,(j//4)*280))
        sheet.save(out/f'page-{start//16:02}.jpg',quality=95)
    report={'version':14,'passed':True,'frames':181,'masks':543,'all_archive_members_verified':True,'scenario_frame_set_exact':True,'all_source_and_label_hashes_verified':True,'archive_sha256':sha(archive),'portable_manifest_sha256':sha(dest),'local_manifest_sha256':sha(mp),'archive_bytes':archive.stat().st_size,'proposal_regions_source_reviewed':len(review['decisions']),'core_crops_reviewed':332,'core_crops_rejected':len(qa['rejected_region_keys']),'new_accepted_core_regions':332-len(qa['rejected_region_keys']),'positive_annotation_region_proxies':sum(len(e['objects']) for e in m['entries']),'not_unique_physical_objects':True,'counts':dict(counts),'mean_per_frame_label_coverage':float(np.mean(coverage)),'minimum_per_frame_label_coverage':min(coverage),'weak_ground_conflict_pixels_withheld':m['weak_ground_conflict_pixels_withheld_as_unknown'],'teacher_predictions_are_not_positive_ground_truth':True,'population':'All181 demo sources used for fitting/selection; partial labels, no held-out/generalization or whole-image zero-error claim.','production_reads_these_annotations':False}
    (R/'data/annotations/live-demo-v14-package.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
