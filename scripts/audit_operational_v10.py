"""Check revised inputs, unchanged historical truth and all source split roles."""
import argparse,gzip,hashlib,json,os,re
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--external',action='store_true');args=ap.parse_args()
    m=json.loads((T/'manifest-operational-v10.json').read_text('utf8'));old=json.loads((T/'manifest-operational-v9.json').read_text('utf8'));old_by={e['source_id']:e for e in old['entries']}
    seals=json.loads((ROOT/'data/annotations/operational-v10-training-seal.json').read_text('utf8'))
    assert sha(T/'manifest-operational-v10.json')==seals['manifest_sha256']
    stats=Counter();seen={k:{} for k in ['source','rmc','group']};small_roles=Counter()
    for e in m['entries']:
        prev=old_by[e['source_id']];a=np.asarray(Image.open(prev['label']));b=np.asarray(Image.open(e['label']))
        assert sha(prev['image'])==prev['source_sha256']==e['source_sha256']
        assert sha(prev['label'])==prev['label_sha256']==e['prior_label_sha256']
        assert sha(e['label'])==e['label_sha256'] and b.shape==a.shape==(1024,1024)
        partial=np.asarray(Image.open(e['partial_ground_mask']),dtype=bool)
        assert sha(e['partial_ground_mask'])==e['partial_ground_sha256']
        assert partial.shape==b.shape and not (partial&(b!=0)).any()
        assert int(((b==0)&~partial).sum())==e['known_bare_ground_pixels']
        assert set(np.unique(b))<={0,1,2,3,4,255}
        assert all(np.array_equal(a==c,b==c) for c in [1,2,3])
        assert not ((b==4)&np.isin(a,[1,2,3])).any()
        assert e['split']==('consumed_control' if prev['split']=='test_v9' else prev['split'])
        for k,value in [('source',e['source_sha256']),('rmc',re.search(r'RAD_F(\d{3})(\d{4})',e['source_id']).groups()),('group',e['group'])]:
            assert seen[k].setdefault(value,e['split'])==e['split']
        stats[e['split']]+=1;small_roles[e['split']]+=int((b==4).sum())
    revisions=json.loads((ROOT/'data/annotations/operational-v10-revision.json').read_text('utf8'));current={e['source_id']:e for e in m['entries']}
    checked_cores=0
    from PIL import ImageDraw
    for scene in revisions['scenes']:
        e=current[scene['source_id']];b=np.asarray(Image.open(e['label']));a=np.asarray(Image.open(old_by[scene['source_id']]['label']))
        for item in scene['accepted']:
            x,y,rx,ry=item['interior_ellipse_512'];core_image=Image.new('L',(1024,1024));ImageDraw.Draw(core_image).ellipse((2*(x-rx),2*(y-ry),2*(x+rx),2*(y+ry)),fill=1)
            core=np.asarray(core_image,bool)&~np.isin(a,[1,2,3]);assert core.any() and (b[core]==4).all(),item['proposal_id'];checked_cores+=1
    report={'images_checked':len(m['entries']),'roles':dict(stats),'small_rock_pixels_by_role':dict(small_roles),'historical_masks_unchanged':True,'all_original_obstacle_hardware_sky_pixels_preserved':True,'source_and_label_sha_checks_passed':True,'cross_split_source_group_rmc_collisions':0,'manifest_sha256':sha(T/'manifest-operational-v10.json'),'sparse_labels_not_exhaustive_masks':True}
    report['all_accepted_stone_cores_preserved']=checked_cores
    if args.external:
        external=json.loads((T/'manifest-v4-final.json').read_text('utf8'));counts=Counter();empty=Counter();pixels=np.zeros(4,dtype=np.int64);digest=hashlib.sha256();excluded=[];clean=[];empty_rows=[]
        for e in external['entries']:
            path=Path(e['label']);a=np.asarray(Image.open(path));im=Image.open(e['image'])
            assert a.shape==im.size[::-1] and set(np.unique(a))<={0,1,2,3,255}
            counts[e['split']]+=1;empty[e['split']]+=int(not (a<4).any());pixels+=np.bincount(a[a<4],minlength=4)
            digest.update(str(e.get('stem',e.get('source_id'))).encode());digest.update(path.read_bytes())
            is_empty=not (a<4).any()
            if is_empty:empty_rows.append({'source_id':e.get('stem',e.get('source_id')),'split':e['split'],'label_sha256':sha(path)})
            if is_empty and e['split']=='train':excluded.append(e.get('stem',e.get('source_id')))
            else:clean.append(e)
        report['external_terrain_audit']={'entries':len(external['entries']),'counts':dict(counts),'empty_review_masks':dict(empty),'class_pixel_counts':pixels.tolist(),'mask_collection_sha256':digest.hexdigest(),'labels_modified':False,'visual_relabel_claimed':False,'taxonomy':'Publisher terrain labels: bedrock/soil/sand/big_rock; this taxonomy is not the five-class operational hardware/rock-appearance task.'}
        clean_path=T/'manifest-v10-terrain-clean.json';clean_manifest={**external,'entries':clean,'v10_excluded_empty_training_masks':excluded,'note_v10':'Only all255 training entries excluded; original files and frozen400/400 validation/test membership remain unchanged.'};clean_path.write_text(json.dumps(clean_manifest,indent=2),encoding='utf8')
        cleaning={'original_manifest_sha256':sha(T/'manifest-v4-final.json'),'clean_manifest_sha256':sha(clean_path),'empty_entries':empty_rows,'excluded_training_ids':excluded,'counts':dict(Counter(e['split'] for e in clean)),'validation_and_test_membership_unchanged':True,'all_original_images_and_labels_unchanged':True,'does_not_relabel_publisher_taxonomy':True}
        (ROOT/'data/annotations/operational-v10-external-cleaning.json').write_text(json.dumps(cleaning,indent=2),encoding='utf8')
    out=ROOT/'data/annotations/operational-v10-integrity.json'
    if not args.external and out.exists():
        prior=json.loads(out.read_text('utf8'))
        if 'external_terrain_audit' in prior:report['external_terrain_audit']=prior['external_terrain_audit']
    out.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
