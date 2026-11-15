"""Find measured label disagreements for visual adjudication, never auto-label them."""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage
_dll=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training';sys.path.insert(0,str(ROOT))
from backend.surface_inference import predict_outputs,navigation_masks
def main():
    out=ROOT/'artifacts/v11-error-audit';out.mkdir(parents=True,exist_ok=True)
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    sessions=[ort.InferenceSession(str(p),sess_options=opts,providers=['CPUExecutionProvider']) for p in [ROOT/'data/prepared/navigation.onnx',T/'v20-v10-navigation-compatible/surface.onnx']]
    entries=[e for e in json.loads((T/'manifest-operational-v9.json').read_text('utf8'))['entries'] if e['split']=='test_v9'];scenes=[];crops=[]
    for e in entries:
        image=Image.open(e['image']).convert('RGB');rgb=np.asarray(image);truth=np.asarray(Image.open(e['label']));masks=[];probs=[]
        for model in sessions:
            p=predict_outputs(model,image,'whole')[0][0];mask,_=navigation_masks(p,image.size,.7,.8,3);mask=np.asarray(Image.fromarray(mask).resize(image.size,Image.Resampling.NEAREST));masks.append(mask);probs.append(p)
        before,after=masks;extra=after&~before&(truth==0);cc,n=ndimage.label(extra);parts=[]
        for k,sl in enumerate(ndimage.find_objects(cc),1):
            if sl is None:continue
            area=int((cc[sl]==k).sum())
            if area<200:continue
            parts.append((area,k,sl))
        parts.sort(reverse=True,key=lambda p:p[0])
        comps=[]
        for rank,(area,k,sl) in enumerate(parts[:3]):
            bbox=[sl[1].start,sl[0].start,sl[1].stop,sl[0].stop];pad=32;box=[max(0,bbox[0]-pad),max(0,bbox[1]-pad),min(1024,bbox[2]+pad),min(1024,bbox[3]+pad)]
            source=rgb[box[1]:box[3],box[0]:box[2]].copy();local=cc[box[1]:box[3],box[0]:box[2]]==k
            if np.median(source)<60:source=(255*np.power(source.astype(float)/255,.45)).astype('uint8')
            boundary=local&~ndimage.binary_erosion(local);source[boundary]=[255,80,210]
            preview=Image.fromarray(source);preview.thumbnail((320,300));tile=Image.new('RGB',(336,340),'#101820');tile.paste(preview,((336-preview.width)//2,30));ImageDraw.Draw(tile).text((5,5),f"Sol{e['sol']} extra{rank} / {area}px / GT0",fill='white')
            crops.append(tile);comps.append({'id':f"{e['sol']}-{rank}",'extra_labeled_ground_pixels':area,'bbox':bbox,'crop_bbox':box,'mean_source_intensity':float(rgb[cc==k].mean())})
        scenes.append({'sol':e['sol'],'source_id':e['source_id'],'ground_pixels':int((truth==0).sum()),'old_ground_obstacle_pixels':int((before&(truth==0)).sum()),'new_ground_obstacle_pixels':int((after&(truth==0)).sum()),'new_only_ground_pixels':int(extra.sum()),'old_only_ground_pixels':int((before&~after&(truth==0)).sum()),'largest_new_disagreements':comps})
        for index,mask in enumerate(masks):
            render=rgb.astype(float).copy();render[mask]=.6*render[mask]+[100,20,10]
            Image.fromarray(render.clip(0,255).astype('uint8')).save(out/f"sol{e['sol']}-{'old' if index==0 else 'new'}.jpg",quality=95)
    for page in range((len(crops)+11)//12):
        canvas=Image.new('RGB',(1344,1020),'#101820')
        for j,crop in enumerate(crops[page*12:page*12+12]):canvas.paste(crop,(j%4*336,j//4*340))
        canvas.save(out/f'new-ground-disagreements-{page}.jpg',quality=95)
    report={'models':[hashlib.sha256(Path(s._model_path).read_bytes()).hexdigest() for s in sessions],'policy':{'obstacle':.7,'hardware':.8,'margin':3},'scenes':scenes,'warning':'GT0 includes small stones and sparse visual ambiguity. These are disagreements requiring native visual review, not automatic negative labels.'}
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(scenes),flush=True)
if __name__=='__main__':main()
