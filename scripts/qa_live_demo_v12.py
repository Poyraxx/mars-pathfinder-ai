"""Numbered native review crops for every manually proposed positive region."""
import json,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training/live-demo-v12'
def main():
    sources=json.loads((T/'sources.json').read_text('utf8'))['entries'];a=json.loads((R/'data/annotations/live-demo-v12-labels.json').read_text('utf8'));out=R/'artifacts/live-demo-v12-qa';out.mkdir(exist_ok=True,parents=True);tiles=[];index=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
    for s in a['scenes']:
        if not s['reviewed']:continue
        src=sources[s['index']];rgb=np.asarray(Image.open(src['image']).convert('RGB'));gamma=.6 if np.median(rgb)<45 else 1;im=Image.fromarray((255*(rgb/255.)**gamma).astype('uint8'));scale=im.width/512
        for i,r in enumerate(s['regions']):
            if r['class_id'] not in (1,4):continue
            if r['shape']=='ellipse':
                x,y=r['center'];rx,ry=r['radius'];bounds=[x-rx,y-ry,x+rx,y+ry]
            else:
                xs,ys=zip(*r['points']);bounds=[min(xs),min(ys),max(xs),max(ys)]
            pad=14;box=[int(max(0,bounds[0]-pad)*scale),int(max(0,bounds[1]-pad)*scale),int(min(512,bounds[2]+pad)*scale),int(min(512,bounds[3]+pad)*scale)]
            crop=im.crop(box);d=ImageDraw.Draw(crop);color='#ff7060' if r['class_id']==1 else '#ff80ff'
            if r['shape']=='ellipse':d.ellipse((scale*(x-rx)-box[0],scale*(y-ry)-box[1],scale*(x+rx)-box[0],scale*(y+ry)-box[1]),outline=color,width=2)
            else:d.line([(x*scale-box[0],y*scale-box[1]) for x,y in r['points']+[r['points'][0]]],fill=color,width=2)
            crop.thumbnail((174,150));tile=Image.new('RGB',(180,180),'#101820');tile.paste(crop,((180-crop.width)//2,26+(150-crop.height)//2));key=f"{s['index']}.{i}";ImageDraw.Draw(tile).text((3,4),f"{key} c{r['class_id']}",font=font,fill='white');index.append({'key':key,'index':s['index'],'region':i,'class_id':r['class_id'],'page':len(tiles)//48});tiles.append(tile)
    for start in range(0,len(tiles),48):
        sheet=Image.new('RGB',(1440,1080),'#101820')
        for n,tile in enumerate(tiles[start:start+48]):sheet.paste(tile,((n%8)*180,(n//8)*180))
        sheet.save(out/f'page-{start//48:02}.jpg',quality=96)
    (out/'index.json').write_text(json.dumps(index,indent=2),encoding='utf8');print(len(tiles),'positive proposals',(len(tiles)+47)//48,'pages')
if __name__=='__main__':main()
