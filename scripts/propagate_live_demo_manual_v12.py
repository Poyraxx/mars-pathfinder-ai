"""Assist stationary-scene labelling; every resulting crop still needs review.

Manual native-source anchor points, local template matching and explicit rejection
of inconsistent transfers. No semantic model, generated content or invented3D.
"""
import hashlib,json,os
from pathlib import Path
import numpy as np
from scipy.signal import correlate2d
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training/live-demo-v12'
ANCHOR=132
# Points chosen on the actual1024px anchor, coordinates normalized to512.
POINTS=[(1,185,180,5,4),(1,189,195,7,4),(1,131,206,4,5),(1,154,204,5,5),(1,190,214,6,6),(1,209,218,5,6),(1,161,223,4,4),(1,159,239,5,4),(1,197,259,11,6),(1,225,265,4,6),(1,246,256,7,4),(1,262,240,3,3),(1,300,230,17,8),(1,396,324,5,6),(4,234,401,7,9)]
def main():
    sources=json.loads((T/'sources.json').read_text('utf8'))['entries'];gray=[]
    def image(i):return Image.open(sources[i]['image']).convert('RGB')
    def plane(i):return np.asarray(image(i).resize((512,512),Image.Resampling.BILINEAR).convert('L'),float)/255
    def match(template_image,target,x,y):
        patch=template_image[y-15:y+16,x-15:x+16];window=target[y-35:y+36,x-35:x+36];centered=patch-patch.mean();ones=np.ones(patch.shape);n=patch.size
        sums=correlate2d(window,ones,mode='valid');squares=correlate2d(window**2,ones,mode='valid');denom=np.sqrt(np.maximum(0,squares-sums**2/n))*np.linalg.norm(centered)
        score=np.divide(correlate2d(window,centered,mode='valid'),denom,out=np.full_like(denom,-1),where=denom>1e-8);dy,dx=np.unravel_index(score.argmax(),score.shape)
        return np.array([x+dx-20,y+dy-20]),float(score[dy,dx])
    src=plane(ANCHOR);rows=[];tiles=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13);out=R/'artifacts/live-demo-v12-transfer';out.mkdir(exist_ok=True,parents=True)
    for i in range(131,142):
        dest=plane(i);raw=np.asarray(image(i));display=Image.fromarray((255*(raw/255.)**(.6 if np.median(raw)<45 else 1)).astype('uint8'))
        for j,(c,x,y,rx,ry) in enumerate(POINTS):
            q,score=match(src,dest,x,y);back,reverse_score=match(dest,src,*q);fb=float(np.linalg.norm(back-[x,y]));eligible=fb<=1.5 and score>=.35 and reverse_score>=.35
            row={'frame_index':i,'anchor_index':ANCHOR,'point':j,'class_id':c,'shape':'ellipse','center':[float(q[0]),float(q[1])],'radius':[rx,ry],'anchor_center':[x,y],'forward_backward_error_px512':fb,'normalized_correlation':score,'reverse_correlation':reverse_score,'eligible_for_manual_review':bool(eligible),'manual_accepted':False,'source_pixel_sha256':sources[i]['pixel_sha256']};rows.append(row)
            box=[max(0,int((q[0]-rx-14)*2)),max(0,int((q[1]-ry-14)*2)),min(1024,int((q[0]+rx+14)*2)),min(1024,int((q[1]+ry+14)*2))];crop=display.crop(box);d=ImageDraw.Draw(crop);d.ellipse((2*(q[0]-rx)-box[0],2*(q[1]-ry)-box[1],2*(q[0]+rx)-box[0],2*(q[1]+ry)-box[1]),outline='#ff6060' if c==1 else '#ff80ff',width=2);crop.thumbnail((174,150));tile=Image.new('RGB',(180,180),'#101820');tile.paste(crop,((180-crop.width)//2,26+(150-crop.height)//2));ImageDraw.Draw(tile).text((3,4),f"{i}.{j} c{c} {'ok' if eligible else 'FAIL'}",fill='white',font=font);tiles.append(tile)
    for start in range(0,len(tiles),48):
        sheet=Image.new('RGB',(1440,1080),'#101820')
        for n,tile in enumerate(tiles[start:start+48]):sheet.paste(tile,((n%8)*180,(n//8)*180))
        sheet.save(out/f'page-{start//48:02}.jpg',quality=96)
    record={'version':12,'method':'Manual visible native-anchor cores, local normalized template correlation, forward/backward consistency. Every proposal must be visually accepted; this file is not a truth mask.','anchor_index':ANCHOR,'coordinate_shape':[512,512],'rows':rows};(R/'data/annotations/live-demo-v12-transfer-proposals.json').write_text(json.dumps(record,indent=2),encoding='utf8');print(len(rows),'proposals',sum(r['eligible_for_manual_review'] for r in rows),'consistent')
if __name__=='__main__':main()
