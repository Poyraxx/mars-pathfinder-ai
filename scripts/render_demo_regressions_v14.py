"""Native-pixel source crops for adjudication; magnification adds no detail."""
import json,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage
R=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training';V=T/'live-demo-v14'
def main():
    record=json.loads((V/'proposals.json').read_text('utf8'));sources=json.loads((T/'live-demo-v12/manifest.json').read_text('utf8'))['entries'];out=R/'artifacts/live-demo-v14-regression-review';out.mkdir(exist_ok=True,parents=True);tiles=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    for p in record['proposals']:
        e=sources[p['index']];native=Image.open(e['image']).convert('RGB');arr=np.asarray(native);gamma=.6 if np.median(arr)<45 else 1.;arr=(255*(arr/255.)**gamma).astype('uint8');raw=Image.fromarray(arr);scale=native.width/512;box=p['bbox512'];pad=12;cropbox=[max(0,box[0]-pad),max(0,box[1]-pad),min(512,box[2]+pad),min(512,box[3]+pad)];nativebox=tuple(int(v*scale) for v in cropbox)
        components=np.load(V/f"{p['index']:03}-proposals.npz")['components'];region=components==p['component'];border=region&~ndimage.binary_erosion(region);border=np.asarray(Image.fromarray(border).resize(native.size,Image.Resampling.NEAREST));colored=arr.copy();colored[border]=[255,75,50]
        context=raw.resize((512,512),Image.Resampling.BILINEAR);draw=ImageDraw.Draw(context)
        for v in range(0,512,32):draw.line((v,0,v,511),fill='#406070');draw.line((0,v,511,v),fill='#406070');draw.text((v+1,1),str(v),fill='#3ce1ff');draw.text((1,v+1),str(v),fill='#3ce1ff')
        context=context.resize((288,288));tile=Image.new('RGB',(640,320),'#101820');tile.paste(context,(0,26))
        for x,im in [(302,raw),(470,Image.fromarray(colored))]:
            crop=im.crop(nativebox);factor=min(160/crop.width,230/crop.height);crop=crop.resize((max(1,round(crop.width*factor)),max(1,round(crop.height*factor))),Image.Resampling.NEAREST);tile.paste(crop,(x+(160-crop.width)//2,40+(230-crop.height)//2))
        d=ImageDraw.Draw(tile);d.text((3,4),f"{p['key']} / native{native.width}px / lost{p['candidate_missed_fraction']:.0%} ground{p['weak_ground_overlap_fraction']:.0%}",font=font,fill='white');d.text((304,280),str(box),font=font,fill='#a8bac5');d.text((304,300),'Source pixels / proposed outline',font=font,fill='white');tiles.append(tile)
    for start in range(0,len(tiles),8):
        canvas=Image.new('RGB',(1280,1280),'#101820')
        for n,tile in enumerate(tiles[start:start+8]):canvas.paste(tile,((n%2)*640,(n//2)*320))
        canvas.save(out/f'page-{start//8:02}.jpg',quality=96)
    print('Rendered',len(tiles),'native source inspection crops',flush=True)
if __name__=='__main__':main()
