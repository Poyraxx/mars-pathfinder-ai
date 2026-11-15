"""Propose original-model detections for source review, never automatic truth."""
import hashlib,json,os,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np
import onnxruntime as ort
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage
from backend.surface_inference import predict_outputs,navigation_masks
T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    m=json.loads((T/'live-demo-v12/manifest.json').read_text('utf8'));out=T/'live-demo-v14';out.mkdir(exist_ok=True,parents=True);render=R/'artifacts/live-demo-v14-regression-review';render.mkdir(exist_ok=True,parents=True);opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;old=T/'backups/live-demo-v12-original-sealed/navigation.onnx';new=T/'v24-live-demo-v13/navigation.onnx';sessions=[ort.InferenceSession(str(p),sess_options=opts,providers=['CPUExecutionProvider']) for p in [old,new]];rows=[];tiles=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for e in m['entries']:
        im=Image.open(e['image']).convert('RGB');truth=np.asarray(Image.open(e['label']).resize((512,512),Image.Resampling.NEAREST));masks=[]
        for session in sessions:
            p=predict_outputs(session,im,'whole')[0][0];obs,hw=navigation_masks(p,im.size,.7,.8,3);masks.append(obs)
        oldobs,newobs=masks;components,n=ndimage.label(oldobs,structure=np.ones((3,3)));np.savez_compressed(out/f"{e['index']:03}-proposals.npz",original=oldobs,candidate=newobs,components=components)
        raw=im.resize((512,512),Image.Resampling.BILINEAR);gray=np.asarray(raw);gamma=.6 if np.median(gray)<45 else 1.;display=(255*(gray/255.)**gamma).astype('uint8')
        for c in range(1,n+1):
            region=components==c;area=int(region.sum()*im.width**2/512**2)
            if area<(3 if im.width==256 else 16):continue
            miss=float((region&~newobs).sum()/region.sum());ground=float((region&(truth==0)).sum()/region.sum());existing=int((region&(truth==1)).sum())
            if existing>=3 and (newobs&region&(truth==1)).sum()>=3:continue
            if miss<.4 and ground<.2:continue
            yy,xx=np.where(region);box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)];pad=24;cropbox=[max(0,box[0]-pad),max(0,box[1]-pad),min(512,box[2]+pad),min(512,box[3]+pad)];key=f"{e['index']}:{c}";rows.append({'key':key,'index':e['index'],'component':c,'bbox512':box,'cropbox512':cropbox,'native_candidate_pixels':area,'candidate_missed_fraction':miss,'weak_ground_overlap_fraction':ground,'existing_class1_pixels512':existing,'proposal_only_not_truth':True})
            source=Image.fromarray(display.copy());d=ImageDraw.Draw(source)
            for v in range(0,512,32):d.line((v,0,v,511),fill='#406070');d.line((0,v,511,v),fill='#406070');d.text((v+1,1),str(v),fill='#3ce1ff');d.text((1,v+1),str(v),fill='#3ce1ff')
            border=region&~ndimage.binary_erosion(region);outlined=display.copy();outlined[border]=[255,75,50];crop=Image.fromarray(outlined).crop(cropbox);crop.thumbnail((310,230));tile=Image.new('RGB',(640,320),'#101820');source.thumbnail((288,288));tile.paste(source,(0,26));tile.paste(crop,(304,42));dt=ImageDraw.Draw(tile);dt.text((4,4),f"{key} area{area} lost{miss:.0%} ground{ground:.0%}",font=font,fill='white');dt.text((304,282),str(box),font=font,fill='#a8bac5');tiles.append(tile)
        if e['index']%30==0:print('FRAME',e['index'],'proposals',len(rows),flush=True)
    for start in range(0,len(tiles),8):
        canvas=Image.new('RGB',(1280,1280),'#101820')
        for n,tile in enumerate(tiles[start:start+8]):canvas.paste(tile,((n%2)*640,(n//2)*320))
        canvas.save(render/f'page-{start//8:02}.jpg',quality=95)
    record={'version':14,'purpose':'Manual source review of missed original proposals and weak-ground conflicts after actual fixture failures; predictions never accepted as labels automatically','original_sha256':sha(old),'rejected_candidate_sha256':sha(new),'base_manifest_sha256':sha(T/'live-demo-v12/manifest.json'),'proposals':rows,'pages':(len(tiles)+7)//8};(out/'proposals.json').write_text(json.dumps(record,indent=2),encoding='utf8');(render/'index.json').write_text(json.dumps(record,indent=2),encoding='utf8');print('COMPLETE',len(rows),'proposals',record['pages'],'pages',flush=True)
if __name__=='__main__':main()
