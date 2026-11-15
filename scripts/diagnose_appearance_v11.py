"""Post-test diagnosis only; never changes candidate, labels or thresholds.

The new v11 episodes are consumed. Replaying them here cannot create another
fresh benchmark; the run records this explicitly.
"""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage
_dll=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
import onnxruntime as ort
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from backend.surface_inference import predict_outputs,navigation_masks
T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training';C=T/'v22-v11-appearance'
def up(x):return np.asarray(Image.fromarray(x).resize((1024,1024),Image.Resampling.NEAREST))
def main():
    sealed=json.loads((C/'fresh-test-v11/evaluation.json').read_text('utf8'));assert hashlib.sha256((C/'appearance.onnx').read_bytes()).hexdigest()==sealed['candidate_sha256']
    manifest=json.loads((T/'manifest-operational-v11.json').read_text('utf8'));options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    base=ort.InferenceSession(str(R/'data/prepared/navigation.onnx'),sess_options=options,providers=['CPUExecutionProvider']);aux=ort.InferenceSession(str(C/'appearance.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14);tiles=[];rows=[];substantial=[];allobjects=[]
    for e in manifest['entries']:
        if e['split']!='test_v11':continue
        image=Image.open(e['image']).convert('RGB');rgb=np.asarray(image);gamma=.5 if np.median(rgb)<45 else 1;display=(255*(rgb/255.)**gamma).astype('uint8')
        p=predict_outputs(base,image,'whole')[0][0];a=predict_outputs(aux,image,'whole')[0][0];obs,_=navigation_masks(p,image.size,.7,.8,3);nav=up(obs);app=up((a.argmax(0)==1));oldprob=up(p[1]);auxprob=up(a[1])
        y=np.asarray(Image.open(e['label']));truth=np.asarray(Image.open(e['navigation_metric_label']));fp=nav&(truth==0);high=fp&(oldprob>=.95);weak=fp&~high
        row={'sol':e['sol'],'navigation_ground_fp':int(fp.sum()),'protected_high_confidence_fp':int(high.sum()),'eligible_lower_confidence_fp':int(weak.sum()),'eligible_fp_but_aux_still_rock':int((weak&(auxprob>=.1)).sum()),'objects':[]}
        for o in e['objects']:
            x0,y0,x1,y1=o['bbox'];core=y[y0:y1,x0:x1]==o['class_id'];minimum=max(3,int(.1*core.sum()));n=int((nav[y0:y1,x0:x1]&core).sum());ap=int((app[y0:y1,x0:x1]&core).sum())
            item={**o,'sol':e['sol'],'navigation_hit':n>=minimum,'appearance_hit':ap>=minimum,'appearance_mean_probability':float(auxprob[y0:y1,x0:x1][core].mean()),'navigation_overlap':n,'appearance_overlap':ap}
            row['objects'].append(item);allobjects.append(item)
            if o['class_id']==1:substantial.append(item)
            if o['class_id']!=1 or n>=minimum:continue
            box=(max(0,x0-40),max(0,y0-40),min(1024,x1+40),min(1024,y1+40));crop=display[box[1]:box[3],box[0]:box[2]].copy()
            for mask,color in [(app,[255,70,255]),(y==1,[70,255,120])]:
                local=mask[box[1]:box[3],box[0]:box[2]];crop[local&~ndimage.binary_erosion(local)]=color
            im=Image.fromarray(crop);im.thumbnail((290,250));tile=Image.new('RGB',(300,300),'#101820');tile.paste(im,((300-im.width)//2,45+(250-im.height)//2));draw=ImageDraw.Draw(tile);draw.text((5,4),f"Sol{e['sol']} region{o['region']} NAV missed",fill='white',font=font);draw.text((5,23),f"appearance {'found' if ap>=minimum else 'missed'} / {item['appearance_mean_probability']:.2f}",fill='#ff99ff',font=font);tiles.append(tile)
        rows.append(row)
    out=R/'artifacts/v11-diagnosis';out.mkdir(exist_ok=True,parents=True)
    for start in range(0,len(tiles),12):
        sheet=Image.new('RGB',(1200,900),'#101820')
        for i,tile in enumerate(tiles[start:start+12]):sheet.paste(tile,((i%4)*300,(i//4)*300))
        sheet.save(out/f'missed-substantial-{start//12}.jpg',quality=95)
    cm=np.asarray(sealed['auxiliary_confusion_matrix']);report={'candidate_sha256':sealed['candidate_sha256'],'test_is_consumed':True,'diagnostic_only':True,'model_or_policy_changed':False,'appearance_threshold':'binary argmax, equivalent to0.5, no search','substantial_total':len(substantial),'navigation_found':sum(o['navigation_hit'] for o in substantial),'appearance_found':sum(o['appearance_hit'] for o in substantial),'navigation_misses_appearance_found':sum(not o['navigation_hit'] and o['appearance_hit'] for o in substantial),'appearance_all_objects_found':sum(o['appearance_hit'] for o in allobjects),'appearance_all_objects':len(allobjects),'appearance_precision':float(cm[1,1]/max(1,cm[:,1].sum())),'appearance_pixel_recall':float(cm[1,1]/max(1,cm[1].sum())),'scenes':rows,'warning':'Appearance positives include nonblocking and raised-looking terrain; this is not extra obstacle recall. Post-test diagnosis cannot justify tuning or a release claim.'}
    (R/'data/annotations/operational-v11-posttest-diagnosis.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k!='scenes'}),flush=True)
if __name__=='__main__':main()
