"""Prepare authentic native Hazcam RAD images for explicitly reviewed polygons."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
from pathlib import Path
import httpx
import numpy as np
from PIL import Image
import prepare_pds as P


def main():
    ap=argparse.ArgumentParser();choice=ap.add_mutually_exclusive_group(required=True)
    choice.add_argument('--sols',type=int,nargs='+');choice.add_argument('--annotations',type=Path)
    args=ap.parse_args()
    root=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training/manual-native';root.mkdir(parents=True,exist_ok=True)
    path=root/'sources.json';existing=json.loads(path.read_text('utf8')) if path.exists() else []
    jobs=json.loads(args.annotations.read_text('utf8'))['scenes'] if args.annotations else [{'sol':s} for s in args.sols]
    def prepare(job):
        sol=job['sol'];pinned='image_sha256' in job
        if pinned:
            out=root/'images'/(job['id']+'.png')
            if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest()==job['image_sha256']:
                return {k:v for k,v in job.items() if k in ('id','sol','shape','source_url','label_url','raw_sha256','image_sha256','preparation','split','group')}|{'image':str(out)}
        base=f'https://planetarydata.jpl.nasa.gov/img/data/msl/MSLHAZ_1XXX/DATA/SOL{sol:05d}/'
        if pinned:pid=job['id']
        else:
            response=httpx.get(base,timeout=45);response.raise_for_status()
            names=sorted(set(re.findall(r'href="(FL[AB]_\d+RAD_F[^"/]+)\.LBL"',response.text)))
            if not names:raise ValueError(f'No native front Hazcam at sol{sol}')
            pid=names[0]
        raw=root/'raw';raw.mkdir(exist_ok=True)
        label_url=job.get('label_url',base+pid+'.LBL');source_url=job.get('source_url',base+pid+'.IMG')
        label=P.fetch(label_url,raw/(pid+'.LBL'));binary=P.fetch(source_url,raw/(pid+'.IMG'))
        raw_sha=hashlib.sha256(binary).hexdigest()
        if pinned and raw_sha!=job['raw_sha256']:raise ValueError(f'Pinned original PDS bytes differ: {pid}')
        values,_=P.decode_pds(label,binary);a=values[0].astype(float);finite=a[np.isfinite(a)&(a>0)];lo,hi=np.percentile(finite,[1,99])
        image=Image.fromarray(np.uint8(np.clip((a-lo)/(hi-lo),0,1)*255)).convert('RGB')
        out=root/'images'/(pid+'.png');out.parent.mkdir(exist_ok=True)
        temporary=out.with_suffix('.preparing.png');image.save(temporary)
        image_sha=hashlib.sha256(temporary.read_bytes()).hexdigest()
        if pinned and (image_sha!=job['image_sha256'] or list(a.shape)!=job['shape']):raise ValueError(f'Pinned native image preparation differs: {pid}')
        os.replace(temporary,out)
        return {'id':pid,'sol':sol,'image':str(out),'shape':list(a.shape),'source_url':source_url,'label_url':label_url,
                'raw_sha256':raw_sha,'image_sha256':image_sha,
                'preparation':'Actual PDS RAD percentile1-99 stretch; no synthetic scene content',
                'split':job.get('split','unassigned'),'group':job.get('group','manual-sol-'+str(sol))}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:rows=list(ex.map(prepare,jobs))
    by_id={e['id']:e for e in existing+rows};path.write_text(json.dumps(list(by_id.values()),indent=2),'utf8')
    print([(r['sol'],r['id'],r['split']) for r in rows])


if __name__=='__main__':main()
