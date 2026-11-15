"""Correct weak-ground conflicts using manually reviewed native-source interiors."""
import argparse,hashlib,json,os
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage
R=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training';V=T/'live-demo-v14'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf8')
def raster(r,size,inset=1):
    scale=size[0]/512;x,y=r['center'];rx,ry=np.asarray(r['radius'])*inset;im=Image.new('L',size);ImageDraw.Draw(im).ellipse(((x-rx)*scale,(y-ry)*scale,(x+rx)*scale,(y+ry)*scale),fill=1);return np.asarray(im,bool)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');a=ap.parse_args();reviewpath=R/'data/annotations/live-demo-v14-review.json';review=json.loads(reviewpath.read_text('utf8'));basepath=T/'live-demo-v12/manifest.json';base=json.loads(basepath.read_text('utf8'));proposal=json.loads((V/'proposals.json').read_text('utf8'));bykey={p['key']:p for p in proposal['proposals']};assert review['complete'] and len(review['decisions'])==326 and review['reviewed_pages']==list(range(41));assert proposal['base_manifest_sha256']==sha(basepath)
    qa_path=R/'data/annotations/live-demo-v14-core-qa.json';qa=json.loads(qa_path.read_text('utf8')) if qa_path.exists() else {};rejected=set(qa.get('rejected_region_keys',[]));adjustments=qa.get('adjusted_regions',{})
    if a.final:assert qa['all_rendered_core_pages_viewed'] and qa['complete']
    out=V/('labels' if a.final else 'draft-labels');out.mkdir(exist_ok=True,parents=True);render=R/'artifacts/live-demo-v14-core-qa';render.mkdir(exist_ok=True,parents=True);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13);entries=[];tiles=[];index=[];counts=Counter();uncertain_removed=0
    for e in base['entries']:
        im=Image.open(e['image']).convert('RGB');size=im.size;y=np.asarray(Image.open(e['semantic_label'])).copy();w=np.asarray(Image.open(e['weight'])).copy();oldnav=np.asarray(Image.open(e['label']));original=np.load(V/f"{e['index']:03}-proposals.npz")['original'];uncertain512=ndimage.binary_dilation(original,iterations=2);uncertain=np.asarray(Image.fromarray(uncertain512).resize(size,Image.Resampling.NEAREST));weak=(y==0)&(w==35)&uncertain;uncertain_removed+=int(weak.sum());y[weak]=255;w[weak]=0;pieces=[];newobjects=[]
        for key,d in review['decisions'].items():
            if d['index']!=e['index']:continue
            assert d['source_image_sha256']==e['image_sha256']
            for j,original_region in enumerate(d['regions']):
                rk=f'{key}:{j}';r=adjustments.get(rk,original_region);c=r['class_id'];core=raster(r,size,.75)
                if size[0]>=512:core=ndimage.binary_erosion(core,iterations=1)
                halo=ndimage.binary_dilation(raster(r,size),iterations=max(1,round(3*size[0]/512)));ambiguous=halo&(y==0)&(w==35);y[ambiguous]=255;w[ambiguous]=0
                if rk not in rejected and core.sum():pieces.append((c,core,rk))
                p=bykey[key];bx=p['bbox512'];x,yc=r['center'];rx,ry=r['radius'];box=[min(bx[0]-12,x-rx-6),min(bx[1]-12,yc-ry-6),max(bx[2]+12,x+rx+6),max(bx[3]+12,yc+ry+6)];box=tuple(round(np.clip(v,0,512)*size[0]/512) for v in box);raw=np.asarray(im);crop=raw[box[1]:box[3],box[0]:box[2]]
                if not crop.size:continue
                gamma=.55 if np.median(crop)<65 else 1.;arr=(255*(crop/255.)**gamma).astype('uint8');marked=arr.copy();local=core[box[1]:box[3],box[0]:box[2]];border=local&~ndimage.binary_erosion(local);color={0:[40,220,90],1:[255,65,55],2:[70,160,255],3:[245,210,50],4:[235,90,220]}[c];marked[border]=color
                tile=Image.new('RGB',(256,192),'#101820')
                for xoff,plane in [(0,arr),(128,marked)]:
                    pic=Image.fromarray(plane);factor=min(124/pic.width,154/pic.height);pic=pic.resize((max(1,round(pic.width*factor)),max(1,round(pic.height*factor))),Image.Resampling.NEAREST);tile.paste(pic,(xoff+(128-pic.width)//2,28+(154-pic.height)//2))
                ImageDraw.Draw(tile).text((3,4),f'{rk} c{c}',font=font,fill='white');index.append({'region_key':rk,'class_id':c,'page':len(tiles)//48,'index':e['index']});tiles.append(tile)
        for cls in [0,3,4,1,2]:
            for c,core,key in pieces:
                if c==cls:y[core]=c;w[core]=100
        objects=[]
        for old in e['objects']:
            x0,y0,x1,y1=old['bbox'];core=(oldnav[y0:y1,x0:x1]==1)&(y[y0:y1,x0:x1]==1)
            if core.sum()>=3:objects.append({**old,'pixels':int(core.sum())})
        for c,core,key in pieces:
            surviving=core&(y==1)
            if c!=1 or surviving.sum()<3:continue
            yy,xx=np.where(surviving);objects.append({'key':'v14-'+key,'pixels':int(surviving.sum()),'bbox':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],'kind':'manual_visual_candidate'})
        semantic=out/f"{e['index']:03}-semantic.png";nav=out/f"{e['index']:03}-navigation.png";weight=out/f"{e['index']:03}-weight.png";Image.fromarray(y).save(semantic);navigation=y.copy();navigation[navigation==4]=255;w[navigation==255]=0;Image.fromarray(navigation).save(nav);Image.fromarray(w).save(weight)
        entries.append({**e,'semantic_label':str(semantic),'semantic_label_sha256':sha(semantic),'label':str(nav),'label_sha256':sha(nav),'weight':str(weight),'weight_sha256':sha(weight),'objects':objects,'regions':[{'class_id':1,'bbox':o['bbox']} for o in objects],'pixel_counts':{str(c):int((navigation==c).sum()) for c in [0,1,2,3,255]}});counts.update({f'class{c}_pixels':int((navigation==c).sum()) for c in range(4)})
    for start in range(0,len(tiles),48):
        sheet=Image.new('RGB',(2048,1152),'#101820')
        for n,tile in enumerate(tiles[start:start+48]):sheet.paste(tile,((n%8)*256,(n//8)*192))
        sheet.save(render/f'page-{start//48:02}.jpg',quality=96)
    manifest={**{k:v for k,v in base.items() if k!='entries'},'version':14,'status':'final_reviewed' if a.final else 'draft_pending_core_QA','base_manifest_sha256':sha(basepath),'v14_review_sha256':sha(reviewpath),'v14_qa_sha256':sha(qa_path) if qa_path.exists() else None,'weak_ground_conflict_pixels_withheld_as_unknown':uncertain_removed,'correction_policy':'Original-model detections flag uncertain weak class0 pixels only. No prediction becomes a positive label. New positives, negatives and hardware are manually chosen native-source interiors. Non-exhaustive; unknown255 remains.','entries':entries};dest=V/('manifest.json' if a.final else 'draft-manifest.json');dump(dest,manifest);dump(render/'index.json',index);print(json.dumps({'frames':len(entries),'core_crops':len(tiles),'core_pages':(len(tiles)+47)//48,'weak_ground_pixels_withheld':uncertain_removed,'annotation_region_proxies':sum(len(e['objects']) for e in entries),**counts}),flush=True)
if __name__=='__main__':main()
