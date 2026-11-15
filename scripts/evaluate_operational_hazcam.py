"""Native-pixel evaluation of the independent navigation head.

Model/policy hashes must be frozen before opening a test split. Sparse reviewed
regions are not a complete expert benchmark or a proof of zero missed hazards.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage
_dll=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
import onnxruntime as ort
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.surface_inference import predict_outputs,navigation_masks
from train_operational_hazcam import metrics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--split',choices=['validation','test','test_v6','test_v7','test_v8','test_v9'],default='validation');ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--obstacle-threshold',type=float,default=.7);ap.add_argument('--hardware-threshold',type=float,default=.8)
    ap.add_argument('--margin',type=int,default=3);ap.add_argument('--render',type=Path);args=ap.parse_args()
    path=args.candidate/'surface.onnx';model_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    policy={'inference_policy':'whole','navigation_obstacle_threshold':args.obstacle_threshold,'hardware_threshold':args.hardware_threshold,'hardware_margin_native_px':args.margin}
    if args.split.startswith('test'):
        frozen=json.loads((args.candidate/'frozen-selection.json').read_text('utf8'))
        assert frozen['model_sha256']==model_sha and frozen['policy']==policy, 'Test requires a previously frozen model and policy'
        if args.split in ('test_v8','test_v9'):
            assert frozen.get('manifest_sha256')==hashlib.sha256(args.manifest.read_bytes()).hexdigest(), 'New test requires its exact frozen reviewed manifest'
    entries=[e for e in json.loads(args.manifest.read_text('utf8'))['entries'] if e['split']==args.split]
    assert entries
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
    head_index=0 if len(session.get_outputs())==1 else 1
    classes=session.get_outputs()[head_index].shape[1]
    confusion=np.zeros((classes,classes),np.int64);totals=np.zeros(classes,np.int64);runtime_hw_fp=runtime_ground_fp=runtime_obstacle_tp=runtime_obstacle_as_hardware=0
    objects_total=objects_found=0;rows=[];latencies=[]
    for entry in entries:
        source=Image.open(entry['image']).convert('RGB');truth=np.asarray(Image.open(entry['label']))
        # Source and label inputs are pinned by the reviewed manifest.
        assert hashlib.sha256(Path(entry['image']).read_bytes()).hexdigest()==entry['source_sha256']
        assert hashlib.sha256(Path(entry['label']).read_bytes()).hexdigest()==entry['label_sha256']
        tick=time.perf_counter();outputs,passes=predict_outputs(session,source,policy['inference_policy']);latencies.append((time.perf_counter()-tick)*1000)
        assert outputs[head_index].shape[0]==classes
        probability=outputs[head_index];raw=probability.argmax(0).astype('uint8')
        raw=np.asarray(Image.fromarray(raw).resize(source.size,Image.Resampling.NEAREST));valid=truth<classes
        cm=np.bincount(classes*truth[valid]+raw[valid],minlength=classes*classes).reshape(classes,classes);confusion+=cm;totals+=cm.sum(1)
        obstacle,hardware=navigation_masks(probability,source.size,args.obstacle_threshold,args.hardware_threshold,args.margin)
        obstacle=np.asarray(Image.fromarray(obstacle).resize(source.size,Image.Resampling.NEAREST));hardware=np.asarray(Image.fromarray(hardware).resize(source.size,Image.Resampling.NEAREST))
        runtime_hw_fp+=int((obstacle&(truth==2)).sum());runtime_ground_fp+=int((obstacle&(truth==0)).sum())
        runtime_obstacle_tp+=int((obstacle&(truth==1)).sum());runtime_obstacle_as_hardware+=int((hardware&(truth==1)).sum())
        labels,count=ndimage.label(truth==1);found=total=0;object_rows=[]
        for cid,sl in enumerate(ndimage.find_objects(labels),1):
            if sl is None:continue
            mask=labels[sl]==cid
            if mask.sum()<50:continue
            coverage=float(obstacle[sl][mask].mean());detected=coverage>=.25
            total+=1;found+=int(detected)
            object_rows.append({'component_id':cid,'reviewed_pixels':int(mask.sum()),'coverage':coverage,'found':bool(detected),
                                'bbox':[sl[1].start,sl[0].start,sl[1].stop-1,sl[0].stop-1]})
        objects_total+=total;objects_found+=found
        rows.append({'source_id':entry['source_id'],'sol':entry['sol'],'raw_metrics':metrics(cm),'reviewed_objects':total,'found_objects':found,'objects':object_rows})
        if args.render:
            args.render.mkdir(exist_ok=True,parents=True);image=np.asarray(source).astype('float32')
            image[obstacle]=image[obstacle]*.5+[127,30,20];image[hardware]=image[hardware]*.5+[25,90,127]
            preview=Image.new('RGB',(source.width,source.height+32),'#10171e');preview.paste(Image.fromarray(image.clip(0,255).astype('uint8')),(0,32))
            ImageDraw.Draw(preview).text((8,8),f"MODEL PREDICTION / {args.split.upper()} / Sol {entry['sol']} / red: obstacle candidate / blue: rover / NO inferred depth",fill='white')
            preview.save(args.render/f"prediction-sol-{entry['sol']}.jpg",quality=94)
    report={'split':args.split,'images':len(entries),'model_sha256':model_sha,'policy':policy,
            'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),'raw_head_metrics':metrics(confusion),
            'runtime_policy':{'reviewed_obstacle_pixel_recall':runtime_obstacle_tp/max(1,totals[1]),
                              'hardware_as_obstacle_fraction':runtime_hw_fp/max(1,totals[2]),
                              'ground_small_stone_as_obstacle_fraction':runtime_ground_fp/max(1,totals[0]),
                              'obstacle_as_self_exclusion_fraction':runtime_obstacle_as_hardware/max(1,totals[1]),
                              'object_recall_25pct_coverage':{'found':objects_found,'total':objects_total,'recall':objects_found/max(1,objects_total)}},
            'inference_ms_median':float(np.median(latencies)),'per_scene':rows,
            'limitations':'Sparse visual labels on a few separate PDS episodes; not independent expert truth, exhaustive full-image precision, flight qualification, or proof of zero misses. Physical route blockage additionally needs measured stereo size/relief.'}
    args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(report,indent=2),'utf8');print(json.dumps({k:v for k,v in report.items() if k not in ('per_scene',)}),flush=True)


if __name__=='__main__':main()
