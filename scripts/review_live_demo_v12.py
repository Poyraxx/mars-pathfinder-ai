"""Inventory and render every actual frame exposed by Live GPS Test.

No segmentation predictions become truth. Annotation decisions are stored
separately; the rendering itself never marks a scene reviewed.
"""
import argparse,hashlib,json,os
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training/live-demo-v12'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf8')
def main():
    T.mkdir(exist_ok=True,parents=True);out=R/'artifacts/live-demo-v12';out.mkdir(exist_ok=True,parents=True)
    scenarios=json.loads((R/'data/prepared/scenarios.json').read_text('utf8'));catalog=json.loads((R/'data/prepared/frames.json').read_text('utf8'));byid={e['id']:e for e in catalog}
    ids=list(dict.fromkeys(fid for s in scenarios for fid in s['frames']));entries=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12);tiles=[]
    for i,fid in enumerate(ids):
        e=byid[fid];image=R/'data/prepared'/(e.get('display_path') or e['image_path']);im=Image.open(image).convert('RGB');arr=np.asarray(im);size=im.size
        row={'index':i,'frame_id':fid,'source_id':e['reference_image_product_id'],'sol':e['sol'],'instrument':e['instrument'],'shape':list(size[::-1]),'image':str(image),'image_sha256':sha(image),'pixel_sha256':hashlib.sha256(arr.tobytes()).hexdigest(),'xyz_path':str(R/'data/prepared'/e['xyz_path']),'xyz_sha256':sha(R/'data/prepared'/e['xyz_path']),'scenarios':[s['id'] for s in scenarios if fid in s['frames']],'source_url':e.get('source_url'),'products':e['products'],'split':'demo_fit_and_conformance','not_independent_test':True}
        entries.append(row);gamma=.6 if np.median(arr)<45 else 1
        display=Image.fromarray((255*(arr/255.)**gamma).astype('uint8')).resize((512,512),Image.Resampling.BILINEAR);draw=ImageDraw.Draw(display)
        for v in range(64,512,64):
            draw.line((v,0,v,511),fill='#4c6570');draw.line((0,v,511,v),fill='#4c6570');draw.text((v+2,18),str(v),font=small,fill='#6eeeff');draw.text((2,v+2),str(v),font=small,fill='#6eeeff')
        tile=Image.new('RGB',(512,544),'#101820');tile.paste(display,(0,32));ImageDraw.Draw(tile).text((5,6),f"#{i} Sol{e['sol']} {e['instrument']} / {size[0]}px",font=font,fill='white');tiles.append(tile)
    for start in range(0,len(tiles),4):
        sheet=Image.new('RGB',(1024,1088),'#101820')
        for n,tile in enumerate(tiles[start:start+4]):sheet.paste(tile,((n%2)*512,(n//2)*544))
        sheet.save(out/f'page-{start//4:02}.jpg',quality=96)
    manifest={'version':12,'scope':'Every unique frame in all four scenarios returned by Live GPS Test','scenario_sha256':sha(R/'data/prepared/scenarios.json'),'catalog_sha256':sha(R/'data/prepared/frames.json'),'entries':entries};dump(T/'sources.json',manifest)
    portable={**manifest,'entries':[{k:v for k,v in e.items() if k not in ['image','xyz_path']} for e in entries]};dump(R/'data/annotations/live-demo-v12-inventory.json',portable)
    annotation=R/'data/annotations/live-demo-v12-labels.json'
    if not annotation.exists():dump(annotation,{'version':12,'coordinate_shape':[512,512],'classes':{'0':'reviewed_ground_or_flat_continuous_bedrock','1':'substantial_raised_or_detached_visual_candidate','2':'rover_hardware','3':'sky_or_distant_unresolved','4':'other_visible_rock_size_unverified','255':'unreviewed_or_uncertain'},'evaluation':'All demo frames may be used for fitting. Result is demo conformance, never independent test accuracy.','scenes':[{'index':e['index'],'frame_id':e['frame_id'],'image_sha256':e['image_sha256'],'reviewed':False,'regions':[]} for e in entries]})
    print(json.dumps({'unique_frames':len(entries),'scenarios':{s['id']:len(s['frames']) for s in scenarios},'instruments':dict(Counter(e['instrument'] for e in entries)),'unique_pixel_hashes':len({e['pixel_sha256'] for e in entries}),'review_pages':(len(entries)+3)//4}),flush=True)
if __name__=='__main__':main()
