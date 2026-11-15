"""Render every proposed new positive/negative region before accepting labels.

Display-only gamma never modifies a source. Numbered regions refer to the
portable annotation JSON. This does not run a model or change label truth.
"""
import json,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training'
A=json.loads((R/'data/annotations/hazcam-operational-v11.json').read_text('utf8'))
S={s['id']:s for s in json.loads((T/'manual-native/v11-new-sources.json').read_text('utf8')) if 'error' not in s}
out=R/'artifacts/v11-native-qa';out.mkdir(exist_ok=True,parents=True)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
tiles=[];index=[]
for scene in A['scenes']:
    raw=np.asarray(Image.open(S[scene['id']]['image']).convert('RGB'))
    display=Image.fromarray((255*(raw/255.)**(.5 if np.median(raw)<45 else 1)).astype('uint8'))
    for i,region in enumerate(scene['regions']):
        # The generic top strip was not individually traced; remove it in the
        # final materializer, since some arm poses occlude the top centre.
        if i==2:continue
        pts=[(2*x,2*y) for x,y in region['polygon']];xs,ys=zip(*pts)
        box=(max(0,min(xs)-24),max(0,min(ys)-24),min(1024,max(xs)+24),min(1024,max(ys)+24))
        crop=display.crop(box);draw=ImageDraw.Draw(crop)
        draw.line([(x-box[0],y-box[1]) for x,y in pts]+[(pts[0][0]-box[0],pts[0][1]-box[1])],fill={0:'#40e080',1:'#ff6040',2:'#70a0ff',3:'#ffff60',4:'#ff80ff',255:'#aaaaaa'}[region['class_id']],width=2)
        crop.thumbnail((194,170));tile=Image.new('RGB',(200,200),'#101820');tile.paste(crop,((200-crop.width)//2,26+(170-crop.height)//2))
        key=f"{scene['sol']}.{i}";ImageDraw.Draw(tile).text((4,4),f"{key} c{region['class_id']}",fill='white',font=font)
        index.append({'key':key,'sol':scene['sol'],'region':i,'class_id':region['class_id'],'page':len(tiles)//30});tiles.append(tile)
for start in range(0,len(tiles),30):
    sheet=Image.new('RGB',(1200,1000),'#101820')
    for n,tile in enumerate(tiles[start:start+30]):sheet.paste(tile,((n%6)*200,(n//6)*200))
    sheet.save(out/f'page-{start//30:02}.jpg',quality=95)
(out/'index.json').write_text(json.dumps(index,indent=2),encoding='utf8');print(len(tiles),'regions',max(x['page'] for x in index)+1,'pages')
