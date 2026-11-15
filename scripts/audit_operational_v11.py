"""Verify sealed annotations, immutable priors and fresh source separation."""
import json,hashlib,os,re
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    mpath=T/'manifest-operational-v11.json';m=json.loads(mpath.read_text('utf8'));old=json.loads((T/'manifest-operational-v10.json').read_text('utf8'));oldmap={e['source_id']:e for e in old['entries']}
    assert len({e['source_id'] for e in m['entries']})==108
    assert sha(R/'data/annotations/hazcam-operational-v11.json')==m['annotation_sha256'] and sha(R/'data/annotations/operational-v11-native-qa.json')==m['qa_sha256']
    assert sha(R/'data/annotations/operational-v11-protocol.json')==m['protocol_sha256']
    oldrmc={re.search(r'RAD_F(\d{3})(\d{4})',e['source_id']).groups() for e in old['entries']};newrmc={};roles=Counter()
    sources={s['id']:s for s in json.loads((R/'data/annotations/operational-v11-source-pool.json').read_text('utf8'))['sources'] if 'error' not in s}
    for e in m['entries']:
        roles[e['split']]+=1
        for path,expected in [('image','source_sha256'),('label','label_sha256'),('appearance_label','appearance_label_sha256')]:assert sha(e[path])==e[expected],e['source_id']
        y=np.asarray(Image.open(e['label']));b=np.asarray(Image.open(e['appearance_label']));assert y.shape==b.shape==(1024,1024);assert set(np.unique(b))<={0,1,255}
        assert np.array_equal(b==1,np.isin(y,[1,4]))
        if not e['new_source']:
            prior=oldmap[e['source_id']];assert e['label_sha256']==prior['label_sha256'] and e['source_sha256']==prior['source_sha256']
            partial=np.asarray(Image.open(e['partial_ground_mask']),bool);assert not ((b==0)&partial).any()
        else:
            src=sources[e['source_id']];rmc=tuple(src['rmc']);assert rmc not in oldrmc;assert rmc not in newrmc or newrmc[rmc]==e['split'];newrmc[rmc]=e['split']
            if e['split']=='test_v11':assert not src['terrain_training_buffer_overlap'] and not src['old_custom_rmc_collision']
            metric=np.asarray(Image.open(e['navigation_metric_label']));assert sha(e['navigation_metric_label'])==e['navigation_metric_label_sha256'];assert not ((y==4)&(metric!=255)).any()
            for o in e['objects']:
                x0,y0,x1,y1=o['bbox'];assert int((y[y0:y1,x0:x1]==o['class_id']).sum())>=o['pixels']
    assert dict(roles)=={'train':84,'validation':8,'consumed_control':8,'test_v11':8}
    report={'passed':True,'manifest_sha256':sha(mpath),'sources':108,'prior86_sources_and_labels_unchanged':True,'composite_ground_not_binary_negative':True,'roles':dict(roles),'fresh_known_pretraining_and_custom_group_guards_pass':True,'warning':'No image predictions were run in this integrity audit.'}
    (R/'data/annotations/operational-v11-integrity.json').write_text(json.dumps(report,indent=2),'utf8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
