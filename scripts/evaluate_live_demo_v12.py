"""Paired actual ONNX conformance; every source was used for demo fitting."""
import argparse,json,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from train_live_demo_v12 import DemoNavigation,normalize,evaluate_demo,sha,dump
import numpy as np
import torch
import onnxruntime as ort
from PIL import Image,ImageDraw

class Runtime(torch.nn.Module):
    def __init__(self,path):
        super().__init__();options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        assert len(self.session.get_inputs())==len(self.session.get_outputs())==1
        assert self.session.get_inputs()[0].shape==[1,3,512,512]
        assert self.session.get_outputs()[0].name=='navigation_logits'
        self.input=self.session.get_inputs()[0].name
    def forward(self,x):
        out=self.session.run(None,{self.input:x.cpu().numpy()})[0]
        assert out.shape==(1,4,512,512) and np.isfinite(out).all()
        return torch.from_numpy(out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--baseline-model',type=Path,default=R/'data/prepared/navigation.onnx');ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--init',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=True,parents=True);torch.set_num_threads(2)
    entries=json.loads(a.manifest.read_text('utf8'))['entries'];start=time.perf_counter()
    baseline=Runtime(a.baseline_model);bp=a.output/'baseline-onnx-conformance.json'
    if bp.exists():
        before=json.loads(bp.read_text('utf8'));assert before['model_sha256']==sha(a.baseline_model) and before['manifest_sha256']==sha(a.manifest)
    else:
        before=evaluate_demo(baseline,entries,'cpu');before.update(model_sha256=sha(a.baseline_model),manifest_sha256=sha(a.manifest));dump(bp,before)
    print('BASELINE',before['ground_fp'],before['hardware_fp'],flush=True)
    candidate=Runtime(a.candidate);after=evaluate_demo(candidate,entries,'cpu');dump(a.output/'candidate-onnx-conformance.json',after);print('CANDIDATE',after['ground_fp'],after['hardware_fp'],flush=True)
    model=DemoNavigation(a.init);model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True));model=model.eval().to('cuda' if torch.cuda.is_available() else 'cpu');device=next(model.parameters()).device
    agreed=total=0;max_error=0.;min_frame_agreement=1.
    with torch.no_grad():
        for e in entries:
            x=normalize(Image.open(e['image']).convert('RGB'))[None];out=candidate(x).numpy();ref=model(x.to(device)).cpu().numpy();n=int((out.argmax(1)==ref.argmax(1)).sum());agreed+=n;total+=512**2;min_frame_agreement=min(min_frame_agreement,n/512**2);max_error=max(max_error,float(np.abs(out-ref).max()))
    lost=sorted(set(before['manual_regions_found'])-set(after['manual_regions_found']))
    reduction=lambda key:1-after[key]/before[key] if before[key] else (1. if after[key]==0 else -1.)
    gates={'baseline_manual_regions_retained':not lost,'rock_recall_no_more_than_0.02_drop':after['rock_pixel_recall']>=before['rock_pixel_recall']-.02,'ground_fp_reduction_at_least_20pct':reduction('ground_fp')>=.2,'hardware_fp_reduction_at_least_30pct':reduction('hardware_fp')>=.3,'ground_as_hardware_no_increase':after['ground_as_hardware']<=before['ground_as_hardware'],'export_argmax_agreement_at_least_0.999':agreed/total>=.999}
    report={'population':'All181 frames were used for training/selection. Partial annotation conformance, NOT held-out accuracy.','candidate_sha256':sha(a.candidate),'baseline_sha256':sha(R/'data/prepared/navigation.onnx'),'manifest_sha256':sha(a.manifest),'gates':gates,'all_segmentation_gates_passed':all(gates.values()),'lost_baseline_manual_regions':lost,'ground_fp_relative_reduction':reduction('ground_fp'),'hardware_fp_relative_reduction':reduction('hardware_fp'),'baseline_manual_regions_found':len(before['manual_regions_found']),'candidate_manual_regions_found':len(after['manual_regions_found']),'positive_annotation_region_proxies':after['positive_annotation_regions'],'onnx_argmax_agreement':agreed/total,'minimum_frame_argmax_agreement':min_frame_agreement,'onnx_max_abs_logit_difference':max_error,'parity_frames':len(entries),'reviewed_false_alarm_pixels_remaining':after['ground_fp']+after['hardware_fp']+after['sky_fp'],'zero_reviewed_false_alarms_achieved':after['ground_fp']+after['hardware_fp']+after['sky_fp']==0,'elapsed_s':time.perf_counter()-start,'route_and_memory_gates':'Evaluated separately; this report alone does not authorize promotion.'};dump(a.output/'paired-onnx-report.json',report)
    report['baseline_sha256']=sha(a.baseline_model);dump(a.output/'paired-onnx-report.json',report)
    # Render worst remaining reviewed false alarms, preserved small-rock reference,
    # varied tool poses and original low-resolution Navcam sources for visual QA.
    order=sorted(after['frames'],key=lambda f:f['ground_fp']+f['hardware_fp'],reverse=True)
    indices=list(dict.fromkeys([f['index'] for f in order[:12]]+[3,5,105,123,129,131,132,138,141,142,162,173,180]))
    colors=np.asarray([[0,0,0],[255,55,55],[50,140,255],[245,220,30]],np.uint8)
    for page in range((len(indices)+3)//4):
        canvas=Image.new('RGB',(1536,4*550),(12,18,23));draw=ImageDraw.Draw(canvas)
        for row,i in enumerate(indices[page*4:page*4+4]):
            e=entries[i];im=Image.open(e['image']).convert('RGB');base=im.resize((512,512));draw.text((6,row*550+5),f'{i:03} source | original ONNX appearance classes | candidate ONNX appearance classes',fill='white');canvas.paste(base,(0,row*550+25))
            for column,net in [(1,baseline),(2,candidate)]:
                logits=net(normalize(im)[None]);p=logits.softmax(1)[0].numpy();pred=p.argmax(0);visible=((pred==1)&(p[1]>=.7))|((pred==2)&(p[2]>=.8));arr=np.asarray(base).copy();arr[visible]=(.55*arr[visible]+.45*colors[pred[visible]]).astype('uint8');canvas.paste(Image.fromarray(arr),(column*512,row*550+25))
        canvas.save(a.output/f'prediction-qa-{page:02}.jpg',quality=92)
    print('COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()
