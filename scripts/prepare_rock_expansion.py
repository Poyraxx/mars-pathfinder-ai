"""Expand real publisher-labelled training data; reserve previously unseen expert gold.

Keeps every previous validation/test group out of training. Original manifests
are immutable. Expert-gold pixels are downloaded but never inspected here.
"""
from __future__ import annotations
import argparse
import hashlib
import gzip
import io
import json
import os
import re
import concurrent.futures
import threading
import time
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import ndimage
from PIL import Image
from remote_zip import open_ai4mars


def group_of(stem):
    match=re.search(r'N[LR][A-Z]?_(\d{9})',stem)
    return str(int(match[1])//(88775*3)) if match else None


def boxes_of(label):
    labels,_=ndimage.label(label==3);out=[]
    for cid,sl in enumerate(ndimage.find_objects(labels),1):
        if sl is None:continue
        area=int((labels[sl]==cid).sum())
        if area>=8:out.append([sl[1].start,sl[0].start,sl[1].stop-sl[1].start,sl[0].stop-sl[0].start,area])
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=4500);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    root=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training'
    baseline_path=root/'v3-audit-baseline/manifest.json'
    if baseline_path.exists():
        baseline=json.loads(baseline_path.read_text('utf8'))
    else:
        portable=Path(__file__).resolve().parents[1]/'data/annotations/dataset-v4-splits.json.gz'
        with gzip.open(portable,'rt',encoding='utf8') as stream:published=json.load(stream)
        baseline={'source':published['source'],'license':published['license'],'entries':published['baseline_v3']}
        baseline_path.parent.mkdir(exist_ok=True,parents=True);baseline_path.write_text(json.dumps(baseline,indent=2),'utf8')
    existing=baseline['entries'];seen_groups={e['group'] for e in existing}
    excluded={e['group'] for e in existing if e['split']!='train'}
    archive=open_ai4mars();prefix='ai4mars-dataset-merged-0.6/'
    names=archive.namelist()
    image_names={Path(n).stem:n for n in names if '/msl/ncam/images/edr/' in n and n.lower().endswith(('.jpg','.png'))}
    gold=[]
    for n in names:
        if '/msl/ncam/labels/test/masked-gold-min3-100agree/' not in n or not n.endswith('.png'):continue
        stem=Path(n).stem.removesuffix('_merged');group=group_of(stem)
        if group not in seen_groups and stem in image_names:
            gold.append({'image_member':image_names[stem],'label_member':n,'group':group,'split':'sealed_gold_test','stem':stem})
    gold_groups={e['group'] for e in gold};excluded|=gold_groups
    print('Reserved unseen expert gold',len(gold),'images',len(gold_groups),'groups',flush=True)
    def local_path(name):
        if not name.startswith(prefix) or '..' in Path(name).parts:raise ValueError('Invalid member path')
        return root/'members'/name.removeprefix(prefix)
    def ensure(name):
        p=local_path(name);p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.write_bytes(archive.read(name))
        return p
    pool=[]
    source_labels=[n for n in names if '/msl/ncam/labels/train/' in n and n.endswith('.png')]
    for i,name in enumerate(sorted(source_labels,key=lambda n:archive.getinfo(n).header_offset)):
        stem=Path(name).stem.removesuffix('_merged');group=group_of(stem)
        if not group or group in excluded or stem not in image_names:continue
        sclk=int(re.search(r'_(\d{9})',stem)[1])
        if abs(sclk-402556045)<88775*3:continue
        label_path=ensure(name)
        label=np.asarray(Image.open(label_path));boxes=boxes_of(label)
        pool.append({'image_member':image_names[stem],'label_member':name,'group':group,'split':'train','stem':stem,
                     'rock_boxes':boxes,'big_rock_pixels':int((label==3).sum())})
        if i%300==0:print('Scanned publisher labels',i,'/',len(source_labels),'rock images',sum(bool(e['rock_boxes']) for e in pool),flush=True)
    rocky=[e for e in pool if e['rock_boxes']];plain=[e for e in pool if not e['rock_boxes']]
    # Prefer diversity of actual acquisition groups and include small-object scenes.
    def ordering(e):return hashlib.sha256(('rock-v4-'+e['stem']).encode()).hexdigest()
    groups=Counter();selected=[]
    for items,limit in [(rocky,int(args.count*.70)),(plain,args.count)]:
        for e in sorted(items,key=ordering):
            if len(selected)>=limit:break
            if groups[e['group']]>=30:continue
            selected.append(e);groups[e['group']]+=1
    entries=[e for e in existing if e['split']!='train']+selected
    needed=sorted({n for e in entries+gold+existing for n in [e['image_member'],e['label_member']] if not local_path(n).exists()},key=lambda n:archive.getinfo(n).header_offset)
    print('Missing members to fetch',len(needed),flush=True)
    progress=[0];lock=threading.Lock()
    def fetch_slice(subset):
        # Separate ZIP/read cursors per worker; contiguous offsets preserve the
        # bounded 2 MB range cache. Shared ZipFile cursors must not race.
        remote=open_ai4mars()
        try:
            for name in subset:
                p=local_path(name);p.parent.mkdir(parents=True,exist_ok=True)
                for attempt in range(4):
                    try:
                        payload=remote.read(name);temporary=p.with_suffix(p.suffix+'.partial');temporary.write_bytes(payload);temporary.replace(p);break
                    except Exception:
                        if attempt==3:raise
                        time.sleep(2*(attempt+1))
                with lock:
                    progress[0]+=1
                    if progress[0]%100==0:print('Downloaded new members',progress[0],'/',len(needed),flush=True)
        finally:remote.close()
    chunk=max(1,(len(needed)+args.workers-1)//args.workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(fetch_slice,[needed[i:i+chunk] for i in range(0,len(needed),chunk)]))
    for e in entries+gold:
        e['image']=str(local_path(e['image_member']));e['label']=str(local_path(e['label_member']))
    manifest={**{k:v for k,v in baseline.items() if k!='entries'},'entries':entries,
              'train_sampling':'70% rock-bearing native images where available; 30/group cap; native crop curriculum',
              'reserved_expert_groups':sorted(gold_groups),'previous_test_not_used_for_selection':True}
    (root/'manifest-v4-expanded.json').write_text(json.dumps(manifest,indent=2),'utf8')
    (root/'manifest-sealed-gold.json').write_text(json.dumps({'source':baseline['source'],'license':'CC-BY-4.0',
          'protocol':'Official min3 unanimous expert test, excludes ALL groups previously used by v3; no pixel inspection during preparation',
          'entries':gold},indent=2),'utf8')
    frozen=[]
    for entry in existing:
        e={**entry,'image':str(local_path(entry['image_member'])),'label':str(local_path(entry['label_member']))}
        if e['split']=='train':e['rock_boxes']=boxes_of(np.asarray(Image.open(e['label'])))
        frozen.append(e)
    (root/'manifest-v4-crops.json').write_text(json.dumps({**baseline,'entries':frozen},indent=2),'utf8')
    print('COMPLETE',len(selected),'train',sum(bool(e['rock_boxes']) for e in selected),'rock scenes','gold',len(gold),flush=True)


if __name__=='__main__':main()
