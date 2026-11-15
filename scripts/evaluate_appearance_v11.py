"""One opening of sealed fresh episodes with a policy fixed before predictions."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
_dll=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
import onnxruntime as ort
from appearance_guard_v11 import guard_obstacles
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from backend.surface_inference import predict_outputs,navigation_masks
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf8')
def up(mask):return np.asarray(Image.fromarray(mask).resize((1024,1024),Image.Resampling.NEAREST))
def counts(mask,truth):
    return {'ground_fp':int((mask&(truth==0)).sum()),'hardware_fp':int((mask&(truth==2)).sum()),'obstacle_tp':int((mask&(truth==1)).sum()),'ground_pixels':int((truth==0).sum()),'hardware_pixels':int((truth==2).sum()),'obstacle_pixels':int((truth==1).sum())}
def summary(rows,key):
    total={k:sum(r[key][k] for r in rows) for k in rows[0][key]}
    total.update(ground_fp_fraction=total['ground_fp']/max(1,total['ground_pixels']),hardware_fp_fraction=total['hardware_fp']/max(1,total['hardware_pixels']),obstacle_pixel_recall=total['obstacle_tp']/max(1,total['obstacle_pixels']))
    return total
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--split',choices=['test_v11','consumed_control','validation'],default='test_v11');a=ap.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    seal=a.output/'test-opening.json';m=json.loads(a.manifest.read_text('utf8'));entries=[e for e in m['entries'] if e['split']==a.split]
    protocol=R/'data/annotations/operational-v11-protocol.json';policy=json.loads(protocol.read_text('utf8'));assert sha(protocol)==m['protocol_sha256']
    legacy={}
    if a.split=='consumed_control':
        legacy_path=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training/manifest-operational-v9.json'
        legacy={e['source_id']:e for e in json.loads(legacy_path.read_text('utf8'))['entries']}
    if a.split=='test_v11':
        assert not seal.exists(),'Fresh episodes were already opened. Do not rerun or retune on this set.'
        assert all(e['new_source'] for e in entries) and len(entries)==8
        assert sum(len(e['objects']) for e in entries)>=30 and sum(sum(o['class_id']==1 for o in e['objects']) for e in entries)>=20
    sealed={'manifest_sha256':sha(a.manifest),'candidate_sha256':sha(a.candidate),'baseline_sha256':sha(R/'data/prepared/navigation.onnx'),'protocol_sha256':sha(protocol),'split':a.split,'opened_at_unix':time.time(),'policy':policy['filter_policy_fixed_before_test'],'trainer_sha256':sha(R/'scripts/train_appearance_v11.py')}
    assert sealed['baseline_sha256']==policy['production_baseline_sha256'];dump(seal,sealed)
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    base=ort.InferenceSession(str(R/'data/prepared/navigation.onnx'),sess_options=opts,providers=['CPUExecutionProvider']);aux=ort.InferenceSession(str(a.candidate),sess_options=opts,providers=['CPUExecutionProvider']);rows=[];cm=np.zeros((2,2),np.int64)
    for e in entries:
        assert sha(e['image'])==e['source_sha256'] and sha(e['appearance_label'])==e['appearance_label_sha256']
        im=Image.open(e['image']).convert('RGB');p=predict_outputs(base,im,'whole')[0][0];r=predict_outputs(aux,im,'whole')[0][0];old,hardware=navigation_masks(p,im.size,.7,.8,3);guarded,suppressed=guard_obstacles(old,p,r)
        old,guarded=up(old),up(guarded);assert not (guarded&~old).any()
        truthpath=e['navigation_metric_label'] if e['new_source'] else e['label']
        if a.split=='consumed_control':
            original=legacy[e['source_id']];assert sha(original['label'])==original['label_sha256'];truthpath=original['label']
        truth=np.asarray(Image.open(truthpath));binary=np.asarray(Image.open(e['appearance_label']));pred=up(r.argmax(0).astype('uint8'));v=binary<2;cm+=np.bincount(2*binary[v]+pred[v],minlength=4).reshape(2,2)
        objects=[]
        for o in e.get('objects',[]):
            x0,y0,x1,y1=o['bbox'];region=(np.asarray(Image.open(e['label']))[y0:y1,x0:x1]==o['class_id']);size=int(region.sum());minimum=max(3,int(.1*size))
            # Reviewed interior overlap, not whole-object precision or size.
            hits=[int((mask[y0:y1,x0:x1]&region).sum()) for mask in (old,guarded)]
            objects.append({**o,'baseline_hit':hits[0]>=minimum,'guarded_hit':hits[1]>=minimum,'baseline_overlap':hits[0],'guarded_overlap':hits[1],'minimum_overlap_pixels':minimum})
        row={'source_id':e['source_id'],'sol':e['sol'],'baseline':counts(old,truth),'guarded':counts(guarded,truth),'suppressed_pixels':int((old&~guarded).sum()),'subset':True,'objects':objects};rows.append(row)
        rgb=np.asarray(im);rgb=(255*(rgb/255.)**(.5 if np.median(rgb)<45 else 1)).astype('uint8')
        rgb[guarded]=(rgb[guarded]*.6+[100,20,10]).astype('uint8');removed=old&~guarded;rgb[removed]=(rgb[removed]*.5+[0,100,100]).astype('uint8');Image.fromarray(rgb).save(a.output/f"sol{e['sol']}-guard.jpg",quality=96)
        print('SCENE',e['sol'],row['baseline']['ground_fp'],row['guarded']['ground_fp'],flush=True)
    b,g=summary(rows,'baseline'),summary(rows,'guarded');objects=[o for row in rows for o in row['objects']];substantial=[o for o in objects if o['class_id']==1];found=[o for o in substantial if o['baseline_hit']];lost=[o for o in found if not o['guarded_hit']]
    reduction=(b['ground_fp']-g['ground_fp'])/max(1,b['ground_fp']);gates={'minimum_appearance_objects':len(objects)>=30,'minimum_substantial_objects':len(substantial)>=20,'all_baseline_found_substantial_preserved':not lost,'ground_fp_no_increase':g['ground_fp']<=b['ground_fp'],'ground_fp_relative_decrease_5pct':b['ground_fp']>=1000 and reduction>=.05,'hardware_fp_no_increase':g['hardware_fp']<=b['hardware_fp'],'obstacle_recall_within_002':g['obstacle_pixel_recall']>=b['obstacle_pixel_recall']-.02,'navigation_subset':True}
    report={**sealed,'baseline':b,'guarded':g,'relative_ground_fp_reduction':reduction,'appearance_objects':len(objects),'substantial_objects':len(substantial),'baseline_found_substantial':len(found),'guarded_found_substantial':sum(o['guarded_hit'] for o in substantial),'lost_substantial':lost,'auxiliary_confusion_matrix':cm.tolist(),'auxiliary_rock_iou':float(cm[1,1]/max(1,cm[1].sum()+cm[:,1].sum()-cm[1,1])),'metric_gates':gates,'metric_gates_pass':all(gates.values()),'scenes':rows,'warning':'Sparse visual-interior metrics, not exhaustive image accuracy, metric clearance or driving safety. Fresh set is consumed after this run. Memory and manual review remain separate promotion gates.'}
    dump(a.output/'evaluation.json',report);print('COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ('scenes','policy','lost_substantial')}),flush=True)
if __name__=='__main__':main()
