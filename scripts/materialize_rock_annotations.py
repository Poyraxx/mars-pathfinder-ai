"""Rasterize reviewed, sparse native Hazcam polygons; never fill unknown pixels.

Use prepare_manual_hazcam.py first for the annotation source sols. Source hashes
must match. Polygon edges are excluded from training to avoid invented precision.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from prepare_rock_expansion import boxes_of

ROOT=Path(__file__).resolve().parents[1]


def main():
    training=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training'
    root=training/'manual-native'
    annotations=json.loads((ROOT/'data/annotations/rocks-manual-v1.json').read_text('utf8'))
    sources={e['id']:e for e in json.loads((root/'sources.json').read_text('utf8'))}
    entries=[]
    for scene in annotations['scenes']:
        source=sources[scene['id']]
        assert hashlib.sha256(Path(source['image']).read_bytes()).hexdigest()==scene['image_sha256']
        h,w=scene['shape'];label=np.full((h,w),255,np.uint8)
        for region in scene['regions']:
            mask=Image.new('L',(w,h));ImageDraw.Draw(mask).polygon([tuple(p) for p in region['polygon']],fill=1)
            interior=ndimage.binary_erosion(np.asarray(mask,dtype=bool),iterations=2)
            assert ((label[interior]==255)|(label[interior]==region['class_id'])).all(), f"Conflicting reviewed regions: {scene['sol']} / {region['id']}"
            label[interior]=region['class_id']
        output=root/'labels'/(scene['id']+'.png');output.parent.mkdir(exist_ok=True);Image.fromarray(label).save(output)
        entries.append({'image':source['image'],'label':str(output),'group':scene['group'],'split':scene['split'],
                        'source_id':scene['id'],'annotation':'sparse-reviewed-polygons-v1','sampling_weight':35,
                        'rock_boxes':boxes_of(label),'big_rock_pixels':int((label==3).sum())})
        image=np.asarray(Image.open(source['image']).convert('RGB')).copy()
        image[label==3]=(image[label==3]*.4+[150,20,20]).clip(0,255).astype('uint8')
        image[(label<3)]=(image[(label<3)]*.5+[0,90,50]).clip(0,255).astype('uint8')
        review=ROOT/'artifacts'/('manual-review-'+str(scene['sol'])+'.png');Image.fromarray(image).save(review)
    (training/'manifest-manual-native.json').write_text(json.dumps({'protocol':annotations['protocol'],'entries':entries},indent=2),'utf8')
    print('Sparse reviewed scenes:',len(entries),'rock polygons:',sum(r['class_id']==3 for s in annotations['scenes'] for r in s['regions']))


if __name__=='__main__':main()
