"""Pixel zoom for native256px Navcam labels; display only, no new detail."""
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage
from materialize_live_demo_v12 import R,T,raster,candidates
def main():
    sources=json.loads((T/'sources.json').read_text('utf8'))['entries'];rows=json.loads((R/'data/annotations/live-demo-v12-review.json').read_text('utf8'))['frames'];draft=json.loads((R/'data/annotations/live-demo-v12-labels.json').read_text('utf8'))['scenes'];transfer=json.loads((R/'data/annotations/live-demo-v12-transfer-proposals.json').read_text('utf8'))['rows'];pieces={p['key']:p for p in candidates(sources,rows,draft,transfer)};out=R/'artifacts/live-demo-v12-manual-qa';indices=json.loads((out/'index.json').read_text('utf8'));font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
    for page in [6,7,8]:
        sheet=Image.new('RGB',(1440,1080),'#101820')
        for n,q in enumerate([x for x in indices if x['page']==page]):
            piece=pieces[q['key']];src=sources[piece['index']];raw=np.asarray(Image.open(src['image']).convert('RGB'));h,w=raw.shape[:2];size=(w,h);r=piece['region'];c=r['class_id'];full=raster(r,size);core=raster(r,size,.75 if c==2 and r['shape']=='ellipse' else .85 if r['shape']=='ellipse' else 1)
            if w>=512:core=ndimage.binary_erosion(core,iterations=2)
            yy,xx=np.where(full);pad=max(8,int(12*w/512));box=(max(0,xx.min()-pad),max(0,yy.min()-pad),min(w,xx.max()+1+pad),min(h,yy.max()+1+pad));display=(255*(raw/255.)**(.6 if np.median(raw)<45 else 1)).astype('uint8');arr=display[box[1]:box[3],box[0]:box[2]].copy();local=core[box[1]:box[3],box[0]:box[2]];arr[local&~ndimage.binary_erosion(local)]=[255,80,80] if c==1 else [90,170,255] if c==2 else [240,100,240]
            crop=Image.fromarray(arr);scale=min(174/crop.width,150/crop.height);crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.NEAREST);tile=Image.new('RGB',(180,180),'#101820');tile.paste(crop,((180-crop.width)//2,26+(150-crop.height)//2));ImageDraw.Draw(tile).text((3,4),f"{q['key']} c{c}",font=font,fill='white');sheet.paste(tile,((n%8)*180,(n//8)*180))
        sheet.save(out/f'page-{page:02}-pixelzoom.jpg',quality=97)
if __name__=='__main__':main()
