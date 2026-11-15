"""Add reviewed appearance supervision without overwriting historical datasets."""
import hashlib,json,os,gzip
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage
R=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf8')
def polygon(region,scale=1):
    pts=np.asarray(region['polygon'],float)*2;center=pts.mean(0);pts=center+(pts-center)*scale
    out=Image.new('L',(1024,1024));ImageDraw.Draw(out).polygon([tuple(v) for v in pts],fill=1);return np.asarray(out,bool)
def main():
    a=R/'data/annotations/hazcam-operational-v11.json';q=R/'data/annotations/operational-v11-native-qa.json';p=R/'data/annotations/operational-v11-protocol.json'
    annotation=json.loads(a.read_text('utf8'));assert len(annotation['scenes'])==22
    sources={s['id']:s for s in json.loads((T/'manual-native/v11-new-sources.json').read_text('utf8')) if 'error' not in s}
    oldpath=T/'manifest-operational-v10.json';old=json.loads(oldpath.read_text('utf8'))
    out=T/'manual-native/operational-v11-labels';out.mkdir(exist_ok=True);entries=[];counts=Counter()
    for e in old['entries']:
        assert sha(e['image'])==e['source_sha256'] and sha(e['label'])==e['label_sha256']
        label=np.asarray(Image.open(e['label']));partial=np.asarray(Image.open(e['partial_ground_mask']),bool)
        binary=np.full(label.shape,255,np.uint8);binary[np.isin(label,[1,4])]=1;binary[np.isin(label,[2,3])|((label==0)&~partial)]=0
        path=out/(e['source_id']+'-binary.png');Image.fromarray(binary).save(path)
        entries.append({**e,'appearance_label':str(path),'appearance_label_sha256':sha(path),'new_source':False})
    for s in annotation['scenes']:
        assert s['native_object_qa_completed'];src=sources[s['id']];assert sha(src['image'])==s['image_sha256']
        label=np.full((1024,1024),255,np.uint8);metric=np.full_like(label,255);positive=np.zeros_like(label,bool);uncertain=positive.copy();objects=[]
        # Approved negative patches only; inset3px excludes ROI boundaries.
        for i,r in enumerate(s['regions']):
            if r['qa_status']!='accepted_visual_interior' or r['class_id'] not in (0,2,3):continue
            mask=ndimage.binary_erosion(polygon(r),iterations=3);label[mask]=r['class_id']
            if not r.get('exclude_hardware_metric'):metric[mask]=r['class_id']
        cores=[]
        for i,r in enumerate(s['regions']):
            if r['class_id'] not in (1,4):continue
            outline=polygon(r);uncertain |= ndimage.binary_dilation(outline,iterations=12)
            if r['qa_status']!='accepted_visual_interior':continue
            core=ndimage.binary_erosion(polygon(r,r['interior_scale']),iterations=3)
            if core.sum()<8:continue
            cores.append((r['class_id'],core));positive|=core
            yy,xx=np.where(core);objects.append({'region':i,'class_id':r['class_id'],'pixels':int(core.sum()),'bbox':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]})
        # Compose unions before cores: adjacent objects cannot erase each other.
        for r in s['regions']:
            if r['class_id']==255 and r['qa_status']=='accepted_visual_interior':uncertain|=polygon(r)
        label[uncertain]=255;metric[uncertain]=255
        for c,core in cores:label[core]=c;metric[core]=1 if c==1 else 255
        binary=np.full_like(label,255);binary[np.isin(label,[1,4])]=1;binary[np.isin(label,[0,2,3])]=0
        paths={k:out/(s['id']+'-'+k+'.png') for k in ('semantic','binary','metric')}
        for k,arr in [('semantic',label),('binary',binary),('metric',metric)]:Image.fromarray(arr).save(paths[k])
        row={'source_id':s['id'],'sol':s['sol'],'image':src['image'],'source_sha256':sha(src['image']),'split':s['split'],'new_source':True,'label':str(paths['semantic']),'label_sha256':sha(paths['semantic']),'appearance_label':str(paths['binary']),'appearance_label_sha256':sha(paths['binary']),'navigation_metric_label':str(paths['metric']),'navigation_metric_label_sha256':sha(paths['metric']),'objects':objects,'regions':[{'class_id':o['class_id'],'bbox':o['bbox']} for o in objects],'source_url':s['source_url'],'rmc':s['rmc'],'pixel_counts':{str(c):int((label==c).sum()) for c in [0,1,2,3,4,255]}}
        entries.append(row);counts.update({'new_positive_objects':len(objects),'new_substantial_candidate_objects':sum(o['class_id']==1 for o in objects),'new_reviewed_negative_pixels':int((binary==0).sum())})
    manifest={'version':11,'prior_manifest_sha256':sha(oldpath),'annotation_sha256':sha(a),'qa_sha256':sha(q),'protocol_sha256':sha(p),'metric_policy':'Sparse verified interiors; navigation class4 and unreviewed regions ignored. Class1 is a substantial visual candidate, not measured size. Broad equipment negative ROIs excluded from hardware metrics; only checked narrow equipment interiors used.','entries':entries}
    dest=T/'manifest-operational-v11.json';dump(dest,manifest)
    portable={**manifest,'entries':[{k:v for k,v in e.items() if k not in ('image','label','partial_ground_mask','appearance_label','navigation_metric_label')} for e in entries]}
    with gzip.GzipFile(str(R/'data/annotations/operational-v11-splits.json.gz'),'wb',mtime=0) as f:f.write(json.dumps(portable,indent=1).encode())
    stats={'manifest_sha256':sha(dest),'roles':dict(Counter(e['split'] for e in entries)),**counts,'fresh_appearance_objects':sum(len(e['objects']) for e in entries if e['split']=='test_v11'),'fresh_substantial_objects':sum(sum(o['class_id']==1 for o in e['objects']) for e in entries if e['split']=='test_v11')}
    dump(R/'data/annotations/operational-v11-revision.json',stats);print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
