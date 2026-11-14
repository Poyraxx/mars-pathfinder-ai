"""Evaluate ONNX candidates on native publisher pixels and small rock objects.

Default is validation. Run a sealed test only after the candidate is frozen.
Object recall is labelled pixel coverage >=25%, not a certified detection metric.
Small apparent size is NOT a measured distance. Unknown pixels are excluded.
"""
import argparse
import hashlib
import json
import os
import time
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
_dll=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
import onnxruntime as ort
from train_surface import scores
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.surface_inference import predict_probabilities


def evaluate(path, entries, rock_threshold=None, policy='whole'):
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
    size=session.get_inputs()[0].shape[-1];confusion=np.zeros((4,4),np.int64)
    objects={name:{'detected':0,'total':0} for name in ['up-to-16px','17-32px','33px+']}
    confident_missed=rock_total=confident_fp=nonrock_total=0;latency=[];rows=[]
    threshold_hist=np.zeros((101,4,3),np.int64)
    for index,entry in enumerate(entries):
        source=Image.open(entry['image']).convert('RGB');target=np.asarray(Image.open(entry['label']))
        assert source.size==target.shape[::-1]
        tick=time.perf_counter();probability,passes=predict_probabilities(session,source,policy);latency.append((time.perf_counter()-tick)*1000)
        decision=probability.argmax(0).astype('uint8')
        if rock_threshold is not None:decision=np.where(probability[3]>=rock_threshold,3,probability[:3].argmax(0)).astype('uint8')
        decision_score=np.take_along_axis(probability,decision[None],axis=0)[0]
        prediction=np.asarray(Image.fromarray(decision).resize(source.size,Image.Resampling.NEAREST))
        confidence=np.asarray(Image.fromarray(decision_score).resize(source.size,Image.Resampling.NEAREST))
        valid=target<4;truth_rock=target==3
        rock_probability=np.asarray(Image.fromarray(probability[3]).resize(source.size,Image.Resampling.NEAREST))
        nonrock=np.asarray(Image.fromarray(probability[:3].argmax(0).astype('uint8')).resize(source.size,Image.Resampling.NEAREST))
        bins=np.minimum((rock_probability[valid]*100).astype(int),100)
        threshold_hist+=np.bincount(12*bins+3*target[valid].astype(int)+nonrock[valid],minlength=101*12).reshape(101,4,3)
        confusion+=np.bincount(4*target[valid]+prediction[valid],minlength=16).reshape(4,4)
        confident_missed+=int((truth_rock&(prediction!=3)&(confidence>=.7)).sum());rock_total+=int(truth_rock.sum())
        confident_fp+=int((valid&~truth_rock&(prediction==3)&(confidence>=.7)).sum());nonrock_total+=int((valid&~truth_rock).sum())
        components,count=ndimage.label(truth_rock);detected=total=0
        for cid,sl in enumerate(ndimage.find_objects(components),1):
            if sl is None:continue
            object_mask=components[sl]==cid;area=object_mask.sum()
            if area<8:continue
            extent=max(sl[0].stop-sl[0].start,sl[1].stop-sl[1].start)
            bucket='up-to-16px' if extent<=16 else '17-32px' if extent<=32 else '33px+'
            coverage=float((prediction[sl][object_mask]==3).mean());found=coverage>=.25
            objects[bucket]['total']+=1;objects[bucket]['detected']+=int(found);total+=1;detected+=int(found)
        rows.append({'source_id':entry.get('source_id',Path(entry['image']).stem),'group':entry['group'],'rock_objects':total,'detected_objects':detected})
        if index%50==0:print(path.parent.name,index,'/',len(entries),flush=True)
    for row in objects.values():row['recall']=row['detected']/row['total'] if row['total'] else None
    diagnostics=[]
    for cutoff in range(10,91):
        matrix=np.zeros((4,4),np.int64);matrix[:,:3]=threshold_hist[:cutoff].sum(0);matrix[:,3]=threshold_hist[cutoff:].sum((0,2))
        metric=scores(matrix)
        diagnostics.append({'threshold':cutoff/100,'miou':metric['miou'],'rock_iou':metric['class_iou']['big_rock'],
                            'rock_precision':metric['precision']['big_rock'],'rock_recall':metric['recall']['big_rock']})
    return {'metrics_native_pixels':scores(confusion),'object_recall_25pct_coverage':objects,
            'inference_policy':policy,'inference_passes':passes,
            'rock_decision_threshold':rock_threshold,'threshold_diagnostics_not_for_test_tuning':diagnostics,
            'confident_false_traversable_rock_fraction':confident_missed/max(1,rock_total),
            'confident_false_rock_nonrock_fraction':confident_fp/max(1,nonrock_total),
            'inference_ms_median':float(np.median(latency)),'images':len(entries),'input_size':size,
            'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'per_image_objects':rows}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--split',default='validation')
    ap.add_argument('--models',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--rock-threshold',type=float);ap.add_argument('--policy',choices=['whole','native-tiled-v1'],default='whole');args=ap.parse_args()
    assert args.rock_threshold is None or 0<args.rock_threshold<1
    entries=[e for e in json.loads(args.manifest.read_text('utf8'))['entries'] if e['split']==args.split]
    assert entries, 'Empty evaluation split'
    report={'split':args.split,'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            'metric_note':'Same native labelled pixels for every model; size bins are apparent pixels, not distance. Objects need >=8 annotated pixels and >=25% predicted-rock coverage. Not a complete all-stone detection benchmark.',
            'models':{}}
    for path in args.models:
        key=path.parent.name;assert key not in report['models']
        result=evaluate(path,entries,args.rock_threshold,args.policy)
        if args.split!='validation':result.pop('threshold_diagnostics_not_for_test_tuning')
        report['models'][key]=result
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),'utf8')
    print('Report:',args.output,flush=True)


if __name__=='__main__':main()
