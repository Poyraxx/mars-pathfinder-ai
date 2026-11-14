"""Build four-class native masks from manually reviewed operational polygons."""
import hashlib
import argparse
import json
import os
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,default=5);args=parser.parse_args()
    root=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training/manual-native'
    path=ROOT/f'data/annotations/hazcam-operational-v{args.version}.json'
    if not path.exists():path=ROOT/f'data/annotations/history/hazcam-operational-v{args.version}.json'
    spec=json.loads(path.read_text('utf8'))
    sources={e['id']:e for e in json.loads((root/'sources.json').read_text('utf8'))}
    entries=[];classes=4 if args.version>=2 else 3;totals=np.zeros(classes,np.int64)
    for scene in spec['scenes']:
        source=sources[scene['id']];path=Path(source['image'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==scene['image_sha256']
        h,w=scene['shape'];target=np.full((h,w),255,'uint8');regions=[]
        def raster(polygon):
            canvas=Image.new('L',(w,h));ImageDraw.Draw(canvas).polygon([tuple(p) for p in polygon],fill=1)
            return np.asarray(canvas,dtype=bool)
        for poly in scene.get('background_roi',[]):target[raster(poly)]=0
        for poly in scene.get('sky_regions',[]):target[raster(poly)]=3
        for poly in scene.get('ignore_regions',[]):target[raster(poly)]=255
        for region in sorted(scene['regions'],key=lambda r:r['class_id']):
            canvas=Image.new('L',(w,h));polygon=[tuple(p) for p in region['polygon']]
            ImageDraw.Draw(canvas).polygon(polygon,fill=1);interior=ndimage.binary_erosion(np.asarray(canvas,dtype=bool),iterations=2)
            conflict=interior&(target<3)&(target!=region['class_id'])
            if args.version==1 and region['class_id']!=2 and conflict.any():
                raise ValueError(f"Conflicting ground/obstacle review: sol{scene['sol']} {region['id']} {int(conflict.sum())}px")
            target[interior]=region['class_id']
            if args.version>=2 and region['class_id'] in (1,2):
                collar=ndimage.binary_dilation(np.asarray(canvas,dtype=bool),iterations=10)&~interior
                target[collar&(target!=region['class_id'])]=255
            points=np.asarray(polygon);regions.append({'class_id':region['class_id'],'bbox':[*points.min(0).tolist(),*points.max(0).tolist()]})
        # Do not teach a fabricated, pixel-exact boundary at semantic transitions.
        boundaries=np.zeros((h,w),bool)
        for axis in [0,1]:
            pairs=np.diff(target.astype('int16'),axis=axis)!=0
            a,b=[slice(None)]*2,[slice(None)]*2;a[axis]=slice(None,-1);b[axis]=slice(1,None)
            boundaries[tuple(a)]|=pairs;boundaries[tuple(b)]|=pairs
        target[boundaries]=255
        out=root/f'operational-v{args.version}-labels'/(scene['id']+'.png');out.parent.mkdir(exist_ok=True);Image.fromarray(target).save(out)
        counts=np.bincount(target[target<classes],minlength=classes);totals+=counts
        entries.append({'source_id':scene['id'],'image':str(path),'label':str(out),'group':scene['group'],'split':scene['split'],'sol':scene['sol'],
                        'regions':regions,'pixel_counts':counts.tolist(),'source_sha256':scene['image_sha256'],
                        'label_sha256':hashlib.sha256(out.read_bytes()).hexdigest()})
        image=np.asarray(Image.open(path).convert('RGB')).astype('float32')
        for cls,colour in enumerate([[40,210,130],[255,65,45],[45,155,255],[170,160,210]]):
            mask=target==cls;image[mask]=image[mask]*.5+np.asarray(colour)*.5
        preview=Image.new('RGB',(w,h+32),'#10171e');preview.paste(Image.fromarray(image.clip(0,255).astype('uint8')),(0,32))
        ImageDraw.Draw(preview).text((8,8),f"REVIEWED LABELS / Sol {scene['sol']} / red: substantial obstacle / blue: rover / green: ground-small stone",fill='white')
        preview.save(ROOT/'artifacts'/f"operational-v{args.version}-review-{scene['sol']}.jpg",quality=94)
    output=root.parent/f'manifest-operational-v{args.version}.json'
    output.write_text(json.dumps({'classes':spec['classes'],'protocol':spec['protocol'],'entries':entries},indent=2),'utf8')
    print('Operational masks:',dict(Counter(e['split'] for e in entries)),'native class pixels:',totals.tolist(),flush=True)


if __name__=='__main__':main()
