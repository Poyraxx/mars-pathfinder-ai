"""Sparse visual revision sheets; no model prediction becomes a ground-truth label."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageEnhance
ROOT=Path(__file__).resolve().parents[1]
T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training'
REVIEW=ROOT/'artifacts/v10-review'

def entries():return sorted(json.loads((T/'manifest-operational-v9.json').read_text('utf8'))['entries'],key=lambda e:e['sol'])

def main():
    REVIEW.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
    rows=entries()
    for page in range((len(rows)+3)//4):
        canvas=Image.new('RGB',(1024,1088),'#101820');draw=ImageDraw.Draw(canvas)
        for pos,e in enumerate(rows[page*4:page*4+4]):
            x=(pos%2)*512;y=(pos//2)*544
            image=Image.open(e['image']).convert('RGB').resize((512,512),Image.Resampling.BILINEAR)
            # Review-only exposure compensation; source bytes stay immutable.
            a=np.asarray(image);median=float(np.median(a[180:450,100:410]))
            if median<65:image=Image.fromarray((255*np.power(a.astype(float)/255,.65)).astype('uint8'))
            d=ImageDraw.Draw(image)
            for v in range(64,512,64):
                d.line((v,0,v,511),fill='#54717a');d.line((0,v,511,v),fill='#54717a')
                d.text((v+2,2),str(v),fill='#79edff',font=font,stroke_width=1,stroke_fill='black')
                d.text((2,v+2),str(v),fill='#79edff',font=font,stroke_width=1,stroke_fill='black')
            canvas.paste(image,(x,y+32));draw.text((x+7,y+7),f"[{page*4+pos}] Sol{e['sol']} / {e['split']} / coordinates512",font=font,fill='white')
        canvas.save(REVIEW/f'page-{page:02}.jpg',quality=96)
    (REVIEW/'index.json').write_text(json.dumps([{'index':i,'sol':e['sol'],'source_id':e['source_id']} for i,e in enumerate(rows)],indent=2),encoding='utf8')
    print('Review sheets:',len(rows),'sources')

if __name__=='__main__':main()
