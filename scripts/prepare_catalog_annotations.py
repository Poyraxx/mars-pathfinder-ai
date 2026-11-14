"""Copy genuine native catalogue frames into the reviewed-label workspace.

The training sols are restricted to old v3 training acquisition groups. The
validation sol must be absent, with its neighbours, from ALL trained groups.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]


def group(frame):return str(int(re.search(r'_(\d{9})',frame['id'])[1])//(88775*3))
def buffered(groups):return {str(int(g)+d) for g in groups for d in [-1,0,1] if g.isdigit()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-sols',type=int,nargs='+',default=[48,54,55,82,93,100,111,147])
    ap.add_argument('--validation-sols',type=int,nargs='*',default=[])
    ap.add_argument('--include-jake-navigation-training',action='store_true',help='Explicit known demo exception: sol45 is navigation training, never held-out evaluation')
    args=ap.parse_args()
    if args.include_jake_navigation_training:args.train_sols=sorted(set(args.train_sols+[45]))
    training=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training';target=training/'manual-native'
    old=json.loads((training/'v3-audit-baseline/manifest.json').read_text('utf8'))['entries']
    expanded=json.loads((training/'manifest-v4-final.json').read_text('utf8'))['entries']
    old_train={e['group'] for e in old if e['split']=='train'}
    old_heldout=buffered({e['group'] for e in old if e['split']!='train'})
    seen=buffered({e['group'] for e in old+expanded if e['split']=='train'})
    catalogue=json.loads((ROOT/'data/prepared/frames.json').read_text('utf8'))
    sources=json.loads((target/'sources.json').read_text('utf8'))
    for split,sols in [('train',args.train_sols),('validation',args.validation_sols)]:
        for sol in sols:
            frame=next(e for e in catalogue if e['sol']==sol and e.get('camera_model') and 'HAZCAM' in e['instrument'])
            g=group(frame)
            if split=='train':
                assert g in old_train
                assert g not in old_heldout or (sol==45 and args.include_jake_navigation_training), 'Adjacent old held-out episode; only the explicitly disclosed Jake demo exception is allowed'
            else:assert g not in seen, 'Manual validation episode was already used for training'
            product=frame['products'][0];pid=product['product_id']
            image=ROOT/'data/prepared'/(frame.get('display_path') or frame['image_path']);output=target/'images'/(pid+'.png');shutil.copyfile(image,output)
            source={'id':pid,'sol':sol,'image':str(output),'shape':list(Image.open(output).size[::-1]),
                    'source_url':product['image_url'],'label_url':product['label_url'],'raw_sha256':product['image_sha256'],
                    'image_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'split':split,'group':'manual-sol-'+str(sol),
                    'preparation':'Actual PDS RAD percentile1-99 stretch; no synthetic scene content','catalogue_frame_id':frame['id']}
            sources=[e for e in sources if e['id']!=pid]+[source]
            preview=Image.open(output).convert('RGB');draw=ImageDraw.Draw(preview)
            for q in range(0,1024,100):
                draw.line([(q,0),(q,1023)],fill=(80,140,150));draw.line([(0,q),(1023,q)],fill=(80,140,150))
                draw.text((q+2,210),str(q),fill='orange');draw.text((10,q+2),str(q),fill='orange')
            preview.save(ROOT/'artifacts'/f'manual-grid-{sol}.png')
    (target/'sources.json').write_text(json.dumps(sources,indent=2),'utf8')
    print('Prepared actual catalogue sources',args.train_sols,args.validation_sols)


if __name__=='__main__':main()
