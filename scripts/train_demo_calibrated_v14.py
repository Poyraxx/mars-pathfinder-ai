"""Conservative demo fine-tuning: frozen encoder, calibrated CE and teacher anchor.

Teacher outputs regularize unlabelled pixels, never become annotation truth.
Production still runs the exported four-class ONNX with real stereo geometry.
"""
import argparse,json,random,time,runpy
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from train_live_demo_v12 import DemoNavigation,normalize,evaluate_demo,objective,compact,sha,dump,torch,F,nn
from torch.utils.data import DataLoader
from train_demo_constraints_v13 import Data,exclusions
from train_rock_refinement import seed_worker
from backend import data,local_navigation as local,perception,camera_visibility
from backend.surface_inference import navigation_masks
R=Path(__file__).resolve().parents[1]

@torch.no_grad()
def real_routes(model,device):
    """Run unchanged real-image pytest assertions through production geometry."""
    model.eval();cache={};tests=runpy.run_path(str(R/'tests/test_local_navigation.py'))
    def infer(frame_id):
        if frame_id in cache:return cache[frame_id]
        frame=next(f for f in data.frame_catalog() if f['id']==frame_id)
        source,reference=perception.inference_source(frame)
        p=model(normalize(source)[None].to(device)).float().softmax(1)[0].cpu().numpy()
        obs,hardware=navigation_masks(p,source.size,.7,.8,3)
        mask=np.zeros(p.shape[1:],np.uint8);conf=p.max(0);mask[obs]=3;conf[obs]=p[1][obs]
        excluded=hardware|(p.argmax(0)>=2);mask[excluded]=255;conf[excluded]=0
        mask=np.asarray(Image.fromarray(mask).resize(reference.size,Image.Resampling.NEAREST));conf=np.asarray(Image.fromarray(conf.astype('float32')).resize(reference.size,Image.Resampling.NEAREST))
        cache[frame_id]=(frame,mask,conf,b'',{'frame_id':frame_id,'navigation_model_sha256':'offline-training-probe'})
        return cache[frame_id]
    names=['test_real_hazcam_boulder_view_requires_body_clearance_or_reports_no_route','test_real_hazcam_open_sand_route_keeps_body_clearance_from_the_measured_rock_cluster','test_real_hazcam_blocked_view_reports_no_route_instead_of_a_curve']
    rows=[]
    try:
        local._terrain_cached.cache_clear();camera_visibility._cached.cache_clear()
        with patch.object(local,'infer',infer),patch.object(perception,'infer_navigation',infer):
            for name in names:
                try:tests[name]();rows.append({'test':name,'passed':True})
                except AssertionError as exc:rows.append({'test':name,'passed':False,'failure':str(exc)})
            for prefix in ['FLA_401492677','FLA_404772660','FLA_402284021']:
                frame=next(f for f in data.frame_catalog() if f['id'].startswith(prefix));result=local.compute(frame['id']);row=next(x for x in rows if ('boulder' if prefix=='FLA_401492677' else 'open_sand' if prefix=='FLA_404772660' else 'blocked') in x['test']);row.update(frame_id=frame['id'],status=result['status'],path_xy=result['path_xy'],ribbons=len(result['ribbons']),obstacles=len(result['obstacles']))
    finally:
        local._terrain_cached.cache_clear();camera_visibility._cached.cache_clear()
    return {'all_passed':all(x['passed'] for x in rows),'tests':rows,'scope':'Unchanged three real-image route assertions, offline Torch inference through exact production mask/geometry policy. Final ONNX/full suite remains required.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--constraints',type=Path,required=True);ap.add_argument('--init',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--epochs',type=int,default=6);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    random.seed(2028);np.random.seed(2028);torch.manual_seed(2028);torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;device='cuda' if torch.cuda.is_available() else 'cpu'
    manifest=json.loads(a.manifest.read_text());assert manifest['version']==14 and manifest['status']=='final_reviewed';entries=manifest['entries'];cs=json.loads(a.constraints.read_text());assert cs['base_manifest_sha256']==manifest['base_manifest_sha256'] and len(entries)==len(cs['entries'])==181
    for e,c in zip(entries,cs['entries']):
        assert e['index']==c['index'] and e['image_sha256']==c['image_sha256']
        for key in ['image','label','weight']:assert sha(e[key])==e[key+'_sha256']
        for key in ['not_class1','not_class2']:assert sha(c[key])==c[key+'_sha256']
    model=DemoNavigation(a.init).to(device);teacher=deepcopy(model).eval()
    for p in model.encoder.parameters():p.requires_grad_(False)
    for p in teacher.parameters():p.requires_grad_(False)
    baseline=sha(R/'data/prepared/navigation.onnx');terrain=sha(R/'data/prepared/surface.onnx')
    settings={'epochs':a.epochs,'samples_per_epoch':724,'seed':2028,'input_size':512,'batch':2,'accumulation':4,'head_lr':3e-6,'encoder_lr':0.,'encoder_frozen':True,'batch_norm_running_stats':'frozen','unknown_distillation':'KL original teacher/student T=2, weight4.0, unknown pixels and teacher-consistent reviewed pixels; not label substitution','manifest_sha256':sha(a.manifest),'constraints_sha256':sha(a.constraints),'init_sha256':sha(a.init),'trainer_sha256':sha(__file__),'loss_policy':'weighted CE + class exclusions + teacher anchor, no class-wide rock-recall or Dice bias','original_navigation_sha256':baseline,'global_surface_sha256':terrain,'selection':'Improve corrected-label objective; preserve ALL original-found manual regions, recall drop<=0.005; Navcam false hardware<=0.0001; pass unchanged real boulder/open/blocked route assertions. Actual ONNX and full suite required before promotion.','population':'All181 demo sources used for fitting/selection. Sparse annotations, no independent test or zero-error claim.'};dump(a.output/'settings.json',settings)
    before=evaluate_demo(model,entries,device);neg_before=exclusions(model,entries,cs['entries'],device);probe=real_routes(model,device);assert probe['all_passed'],probe;dump(a.output/'initial-real-routes.json',probe);dump(a.output/'initial-demo-conformance.json',before);dump(a.output/'initial-exclusion-conformance.json',neg_before);keep=set(before['manual_regions_found'])
    def value(s,n):return objective(s)-.5*n['navcam_hardware_fraction']-.25*n['clear_flat_rock_fraction']
    best=value(before,neg_before);best_epoch=0;unconstrained=best;history=[];torch.save(model.state_dict(),a.output/'best-navigation.pt');print('INITIAL',json.dumps({'demo':compact(before),'real_routes':probe,'exclusions':{k:v for k,v in neg_before.items() if k!='frames'}}),flush=True)
    loader=DataLoader(Data(entries,cs['entries']),batch_size=2,shuffle=True,num_workers=2,persistent_workers=True,pin_memory=device=='cuda',worker_init_fn=seed_worker)
    opt=torch.optim.AdamW([{'params':model.head.parameters(),'lr':3e-6}],weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=3e-7);scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda');weights=torch.tensor([1,2,1.5,1],device=device);start=time.perf_counter()
    for epoch in range(a.epochs):
        model.train();model.encoder.eval()
        for layer in model.modules():
            if isinstance(layer,nn.BatchNorm2d):layer.eval()
        opt.zero_grad();losses=[]
        for i,(im,y,w,n1,n2) in enumerate(loader):
            im,y,w,n1,n2=[x.to(device) for x in [im,y,w,n1,n2]];valid=y!=255
            with torch.no_grad(),torch.amp.autocast(device_type=device,enabled=device=='cuda'):teacher_logits=teacher(im).float()
            with torch.amp.autocast(device_type=device,enabled=device=='cuda'):
                logits=model(im).float();p=logits.softmax(1);ce=F.cross_entropy(logits,y,weight=weights,ignore_index=255,reduction='none');ce=(ce*w).sum()/w.sum().clamp_min(1);target=F.one_hot(y.clamp(0,3),4).permute(0,3,1,2).float()*valid[:,None];pr=p*valid[:,None];dice=1-((2*(pr*target).sum((0,2,3))+1)/(pr.sum((0,2,3))+target.sum((0,2,3))+1))[1:3].mean();zero=logits.reshape(-1)[0]*0
                def mean(prob,mask):return prob[mask].mean() if mask.any() else zero
                gfp=mean(p[:,1],y==0);hfp=mean(p[:,1],y==2);ghw=mean(p[:,2],y==0);miss=mean(1-p[:,1],y==1);hmiss=mean(1-p[:,2],y==2)
                no_rock=mean(-torch.log1p(-p[:,1].clamp(max=1-1e-6)),n1);no_hw=mean(-torch.log1p(-p[:,2].clamp(max=1-1e-6)),n2)
                kd=mean(F.kl_div(F.log_softmax(logits/2,1),F.softmax(teacher_logits/2,1),reduction='none').sum(1)*4,(~valid)|(teacher_logits.argmax(1)==y))
                loss=(ce+no_rock+2*no_hw+4*kd)/4
            scaler.scale(loss).backward()
            if (i+1)%4==0 or i+1==len(loader):scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update();opt.zero_grad()
            losses.append(float(loss.detach())*4)
            if (i+1)%120==0:print('BATCH',epoch+1,i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
        score=evaluate_demo(model,entries,device);negative=exclusions(model,entries,cs['entries'],device);routes=real_routes(model,device);lost=sorted(keep-set(score['manual_regions_found']));eligible=not lost and score['rock_pixel_recall']>=before['rock_pixel_recall']-.005 and routes['all_passed'] and negative['navcam_hardware_fraction']<=.0001;v=value(score,negative);selected=eligible and v>best
        if v>unconstrained:unconstrained=v;torch.save(model.state_dict(),a.output/'best-unconstrained.pt')
        if selected:best=v;best_epoch=epoch+1;torch.save(model.state_dict(),a.output/'best-navigation.pt');dump(a.output/'best-real-routes.json',routes)
        torch.save(model.state_dict(),a.output/'last-navigation.pt');row={'epoch':epoch+1,'loss':float(np.mean(losses)),'demo':compact(score),'exclusions':{k:v for k,v in negative.items() if k!='frames'},'real_routes':routes,'lost_original_manual_regions':lost,'eligible':eligible,'selected':selected,'score':v,'elapsed_s':time.perf_counter()-start};history.append(row);dump(a.output/'history.json',history);print('EPOCH',json.dumps(row),flush=True);scheduler.step()
    model.load_state_dict(torch.load(a.output/'best-navigation.pt',map_location=device,weights_only=True));final=evaluate_demo(model,entries,device);neg=exclusions(model,entries,cs['entries'],device);routes=real_routes(model,device);dump(a.output/'selected-demo-conformance.json',final);dump(a.output/'selected-exclusion-conformance.json',neg);dump(a.output/'selected-real-routes.json',routes);model=model.cpu().eval();torch.onnx.export(model,torch.randn(1,3,512,512),str(a.output/'navigation.onnx'),opset_version=17,input_names=['image'],output_names=['navigation_logits'],dynamo=False,external_data=False)
    assert sha(R/'data/prepared/navigation.onnx')==baseline and sha(R/'data/prepared/surface.onnx')==terrain
    report={'best_epoch':best_epoch,'epochs_completed':a.epochs,'candidate_sha256':sha(a.output/'navigation.onnx'),'selected':compact(final),'baseline':compact(before),'real_routes':routes,'candidate_only':True,'population':'Demo fit/conformance only, no held-out/generalization claim; no zero-error promise.'};dump(a.output/'candidate-report.json',report);print('COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()
