"""Compose all181 reviewed demo frames without changing historical data.

Geometry proposals are weak labels, manual cores retain provenance, all unknown
boundaries stay ignored. Never call this a held-out test: every frame may fit.
"""
import argparse,hashlib,json,os
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage
R=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training/live-demo-v12'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf8')
def raster(region,size,inset=1):
    scale=size[0]/512;im=Image.new('L',size);d=ImageDraw.Draw(im)
    if region['shape']=='ellipse':
        x,y=region['center'];rx,ry=np.asarray(region['radius'])*inset;d.ellipse(((x-rx)*scale,(y-ry)*scale,(x+rx)*scale,(y+ry)*scale),fill=1)
    else:d.polygon([(x*scale,y*scale) for x,y in region['points']],fill=1)
    return np.asarray(im,bool)
def common_rig():
    return [{'class_id':2,'shape':'polygon','points':[(0,0),(113,0),(105,17),(69,28),(47,43),(0,62)]},{'class_id':2,'shape':'polygon','points':[(461,0),(511,0),(511,80),(484,67),(464,47),(454,26)]},{'class_id':2,'shape':'ellipse','center':[7,324],'radius':[16,47]},{'class_id':2,'shape':'ellipse','center':[505,318],'radius':[15,45]}]
def candidates(sources,review,drafts,transfer):
    pieces=[]
    for src in sources:
        index=src['index'];size=tuple(src['shape'][::-1]);row=review[index]
        # Original first-pass RGB point guesses are not silently accepted. Only
        # clearly traced Jake polygons and the reviewed equipment regions carry.
        if index<len(drafts):
            for j,r in enumerate(drafts[index]['regions']):
                if j<4:continue
                if r['class_id']==2 or (index in [3,4,5,6,7] and r['class_id']==1 and r['shape']=='polygon'):
                    pieces.append({'key':f'draft-{index}-{j}','index':index,'region':r,'provenance':'manual_source_region'})
        for j,r in enumerate(row['additional_manual_regions']):pieces.append({'key':f'manual-{index}-{j}','index':index,'region':r,'provenance':'manual_visual_core_no_invented_height'})
        for r in transfer:
            if r['frame_index']==index and r['manual_accepted']:pieces.append({'key':f"transfer-{index}-{r['point']}",'index':index,'region':r,'provenance':'native_anchor_and_individually_reviewed_template_transfer','previous_crop_qa':True})
    return pieces
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');a=ap.parse_args()
    sp=T/'sources.json';sources=json.loads(sp.read_text('utf8'))['entries'];rp=R/'data/annotations/live-demo-v12-review.json';rows=json.loads(rp.read_text('utf8'))['frames'];assert len(rows)==181 and [r['index'] for r in rows]==list(range(181))
    dp=R/'data/annotations/live-demo-v12-labels.json';draft=json.loads(dp.read_text('utf8'))['scenes'];tp=R/'data/annotations/live-demo-v12-transfer-proposals.json';transfer=json.loads(tp.read_text('utf8'))['rows'];geo=json.loads((T/'stereo-proposals.json').read_text('utf8'))['frames'];pieces=candidates(sources,rows,draft,transfer)
    qa_path=R/'data/annotations/live-demo-v12-manual-qa.json';qa=json.loads(qa_path.read_text('utf8')) if qa_path.exists() else {};rejected=set(qa.get('rejected',[]))
    if a.final:assert qa.get('all_rendered_manual_pages_viewed') and qa.get('all_geometry_pages_viewed')
    out=T/('labels' if a.final else 'draft-labels');out.mkdir(exist_ok=True,parents=True);render=R/'artifacts/live-demo-v12-final';render.mkdir(exist_ok=True,parents=True);qadir=R/'artifacts/live-demo-v12-manual-qa';qadir.mkdir(exist_ok=True,parents=True)
    old=json.loads((T.parent/'manifest-operational-v10.json').read_text('utf8'))['entries'];oldmap={e['source_id']:e for e in old};entries=[];stats=Counter();tiles=[];qindex=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
    for src,row,g in zip(sources,rows,geo):
        assert src['index']==row['index']==g['index'] and sha(src['image'])==src['image_sha256'] and sha(g['proposal_path'])==g['proposal_sha256']
        raw=np.asarray(Image.open(src['image']).convert('RGB'));size=raw.shape[:2][::-1];h,w=raw.shape[:2];label=np.full((h,w),255,np.uint8);weight=np.zeros((h,w),np.uint8);objects=[];masks=[]
        geometry=np.load(g['proposal_path']);comp=geometry['components'];flat=np.asarray(Image.fromarray(geometry['flat']).resize(size,Image.Resampling.NEAREST));label[flat]=0;weight[flat]=35
        # Exclude manually identified rock/hardware extents from flat negatives.
        for rect in row['flat_exclude_rectangles_512']:
            x0,y0,x1,y1=rect;r={'shape':'polygon','points':[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]};m=raster(r,size);label[m]=255;weight[m]=0
        for o in g['objects']:
            cid=o['id'];mask=np.asarray(Image.fromarray(comp==cid).resize(size,Image.Resampling.NEAREST));uncertain=ndimage.binary_dilation(mask,iterations=max(1,w//256));label[uncertain]=255;weight[uncertain]=0
            c=1 if cid in row['accepted_rock_components'] else 2 if cid in row['hardware_components'] else 255
            if c!=255:masks.append((c,mask,75,f'stereo-{cid}'))
        for part in row.get('split_geometry_components',[]):
            x0,y0,x1,y1=part['rect_512'];r={'shape':'polygon','points':[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]};mask=np.asarray(Image.fromarray(comp==part['component']).resize(size,Image.Resampling.NEAREST))&raster(r,size);masks.append((part['class_id'],mask,75,f"stereo-split-{part['component']}-{part['class_id']}"))
        prior=oldmap.get(src['source_id'])
        if prior and Image.open(prior['image']).size==size and np.array_equal(np.asarray(Image.open(prior['image']).convert('RGB')),raw):
            legacy=np.asarray(Image.open(prior['label']));assert sha(prior['label'])==prior['label_sha256']
            for c in [2,3]:masks.append((c,legacy==c,80,'legacy-reviewed'))
            stats['exact_prior_equipment_sky_sources']+=1
        if src['instrument']=='FRONT_HAZCAM_LEFT_A':
            fixed=np.zeros((h,w),bool)
            for r in common_rig():fixed|=raster(r,size)
            fixed&=raw.mean(-1)<70;fixed=ndimage.binary_erosion(fixed,iterations=max(1,w//512));masks.append((2,fixed,80,'context-checked-fixed-rig-interior'))
            # Only white top-sky gaps, never default the centre strip to sky.
            sky=np.zeros((h,w),bool)
            for x0,x1 in [(151,178),(391,425)]:sky[1:max(2,int(18*w/512)),int(x0*w/512):int(x1*w/512)]=True
            sky&=raw.mean(-1)>140;sky=ndimage.binary_erosion(sky,iterations=max(1,w//512));masks.append((3,sky,80,'context-checked-bright-top-sky-gap'))
        for piece in [p for p in pieces if p['index']==src['index']]:
            r=qa.get('adjusted_regions',{}).get(piece['key'],piece['region']);c=r['class_id'];full=raster(r,size);uncertain=ndimage.binary_dilation(full,iterations=max(1,int(6*w/512)));label[uncertain]=255;weight[uncertain]=0
            core=raster(r,size,.75 if c==2 and r['shape']=='ellipse' else .85 if r['shape']=='ellipse' else 1)
            if w>=512:core=ndimage.binary_erosion(core,iterations=2)
            if piece['key'] not in rejected and core.sum()>=3:masks.append((c,core,100,piece['key']))
            if not piece.get('previous_crop_qa'):
                yy,xx=np.where(full)
                if len(xx):
                    pad=max(8,int(12*w/512));box=(max(0,xx.min()-pad),max(0,yy.min()-pad),min(w,xx.max()+1+pad),min(h,yy.max()+1+pad));display=(255*(raw/255.)**(.6 if np.median(raw)<45 else 1)).astype('uint8');local=core[box[1]:box[3],box[0]:box[2]];arr=display[box[1]:box[3],box[0]:box[2]].copy();arr[local&~ndimage.binary_erosion(local)]=[255,80,80] if c==1 else [90,170,255] if c==2 else [240,100,240];crop=Image.fromarray(arr);crop.thumbnail((174,150));tile=Image.new('RGB',(180,180),'#101820');tile.paste(crop,((180-crop.width)//2,26+(150-crop.height)//2));ImageDraw.Draw(tile).text((3,4),f"{piece['key']} c{c}",font=font,fill='white');qindex.append({'key':piece['key'],'page':len(tiles)//48,'index':src['index'],'class_id':c});tiles.append(tile)
        # Reapply unions after all uncertainty collars, so neighbours never erase
        # one another. Hardware has final priority over rock appearance.
        for c in [3,4,1,2]:
            for cls,mask,confidence,key in masks:
                if cls!=c or not mask.any():continue
                label[mask]=c;weight[mask]=confidence
        # Instance proxies must refer to pixels that survive final self priority.
        for cls,mask,confidence,key in masks:
            if cls!=1:continue
            core=mask&(label==1)
            if core.sum()<3:continue
            yy,xx=np.where(core);objects.append({'key':key,'pixels':int(core.sum()),'bbox':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],'kind':'stereo' if key.startswith('stereo') else 'manual_visual_candidate'})
        weight[label==255]=0;nav=label.copy();nav[nav==4]=255;navweight=weight.copy();navweight[nav==255]=0
        paths={k:out/f"{src['index']:03}-{k}.png" for k in ['semantic','navigation','weight']}
        for k,arr in [('semantic',label),('navigation',nav),('weight',navweight)]:Image.fromarray(arr).save(paths[k])
        entries.append({**src,'label':str(paths['navigation']),'label_sha256':sha(paths['navigation']),'semantic_label':str(paths['semantic']),'semantic_label_sha256':sha(paths['semantic']),'weight':str(paths['weight']),'weight_sha256':sha(paths['weight']),'objects':objects,'regions':[{'class_id':1,'bbox':o['bbox']} for o in objects],'pixel_counts':{str(c):int((nav==c).sum()) for c in [0,1,2,3,255]},'manual_context_reviewed':True,'weak_geometry_supervision':True})
        stats.update({f'class{c}_pixels':int((nav==c).sum()) for c in range(4)});stats['positive_regions']+=len(objects)
        rgb=(255*(raw/255.)**(.6 if np.median(raw)<45 else 1)).astype(float)
        for c,color in [(0,[30,180,90]),(1,[240,60,45]),(2,[60,130,245]),(3,[220,170,70]),(4,[230,80,220])]:rgb[label==c]=rgb[label==c]*.65+np.asarray(color)*.35
        view=Image.fromarray(rgb.clip(0,255).astype('uint8'));view.thumbnail((512,512));view.save(render/f"{src['index']:03}.jpg",quality=95)
    for start in range(0,len(tiles),48):
        sheet=Image.new('RGB',(1440,1080),'#101820')
        for n,tile in enumerate(tiles[start:start+48]):sheet.paste(tile,((n%8)*180,(n//8)*180))
        sheet.save(qadir/f'page-{start//48:02}.jpg',quality=96)
    dump(qadir/'index.json',qindex)
    manifest={'version':12,'status':'final_reviewed' if a.final else 'draft_pending_native_QA','population':'All181 demo fitting/conformance sources, no heldout accuracy claim','sources_sha256':sha(sp),'review_sha256':sha(rp),'transfer_sha256':sha(tp),'qa_sha256':sha(qa_path) if qa_path.exists() else None,'geometry_policy':json.loads((T/'stereo-proposals.json').read_text('utf8'))['policy'],'class4_policy':'Appearance only, unknown255 for four-class navigation; no metric dimension inferred','weight_policy':'35/100 for heuristic flat pixels,75/100 for adjudicated stereo masks,80/100 legacy/manual-context interiors,100/100 explicit new visual cores. Weak supervision is not certified clearance.','entries':entries}
    dest=T/('manifest.json' if a.final else 'draft-manifest.json');dump(dest,manifest);print(json.dumps({'frames':len(entries),'manual_pieces':len(pieces),'new_manual_crop_pages':(len(tiles)+47)//48,'new_manual_crops':len(tiles),'manifest_sha256':sha(dest),**dict(stats)}),flush=True)
if __name__=='__main__':main()
