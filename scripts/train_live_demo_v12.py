"""Actual ONNX navigation fine-tune on all reviewed demo frames.

All frames are fitting/conformance data by explicit user request. No frame-ID
lookup, manual-mask inference override, generated camera pixels or test claim.
"""
import argparse,hashlib,json,os,random,time
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image,ImageEnhance
_dll=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from operational_model import OperationalSurface
from train_rock_refinement import seed_worker
import sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from backend.surface_inference import navigation_masks

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf8')
def normalize(im):
    x=np.asarray(im.resize((512,512),Image.Resampling.BILINEAR),np.float32)/255
    return torch.from_numpy(((x-[.485,.456,.406])/[.229,.224,.225]).transpose(2,0,1).astype('float32'))
class DemoNavigation(nn.Module):
    def __init__(self,checkpoint):
        super().__init__();old=OperationalSurface(spatial_refinement=True,separate_navigation_encoder=True);old.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));self.encoder=deepcopy(old.navigation_encoder);self.head=deepcopy(old.navigation_head)
    def forward(self,x):
        h=self.encoder(pixel_values=x,output_hidden_states=True,return_dict=True).hidden_states
        return F.interpolate(self.head(h),size=x.shape[-2:],mode='bilinear',align_corners=False)
class DemoData(Dataset):
    def __init__(self,entries):self.entries=entries
    def __len__(self):return 1086
    def __getitem__(self,index):
        e=self.entries[index%len(self.entries)];im=Image.open(e['image']).convert('RGB');y=Image.open(e['label']);w=Image.open(e['weight'])
        if e['regions'] and random.random()<.4:
            x0,y0,x1,y1=random.choice(e['regions'])['bbox'];side=int(min(im.width,max(x1-x0,y1-y0,random.choice([384,640,1024]))));left=int(np.clip((x0+x1)/2-side/2,0,im.width-side));top=int(np.clip((y0+y1)/2-side/2,0,im.height-side));box=(left,top,left+side,top+side);im=im.crop(box);y=y.crop(box);w=w.crop(box)
        if random.random()<.25:im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT);y=y.transpose(Image.Transpose.FLIP_LEFT_RIGHT);w=w.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        im=ImageEnhance.Brightness(im).enhance(random.uniform(.9,1.1));im=ImageEnhance.Contrast(im).enhance(random.uniform(.9,1.1))
        return normalize(im),torch.from_numpy(np.asarray(y.resize((512,512),Image.Resampling.NEAREST),np.int64).copy()),torch.from_numpy(np.asarray(w.resize((512,512),Image.Resampling.NEAREST),np.float32).copy()/100)
def up(a,size):return np.asarray(Image.fromarray(a).resize(size,Image.Resampling.NEAREST))
@torch.no_grad()
def evaluate_demo(model,entries,device):
    model.eval();cm=np.zeros((4,4),np.int64);totals={k:0 for k in ['ground_fp','hardware_fp','sky_fp','ground_as_hardware','rock_tp','ground_pixels','hardware_pixels','sky_pixels','rock_pixels']};manual_found=[];regions_found=[];rows=[]
    for e in entries:
        im=Image.open(e['image']).convert('RGB');truth=np.asarray(Image.open(e['label']));logits=model(normalize(im)[None].to(device));p=logits.softmax(1)[0].cpu().numpy();obs,hardware=navigation_masks(p,im.size,.7,.8,3);obs,hardware=up(obs,im.size),up(hardware,im.size);pred=up(p.argmax(0).astype('uint8'),im.size);v=truth<4;cm+=np.bincount(4*truth[v]+pred[v],minlength=16).reshape(4,4)
        row={'index':e['index'],'frame_id':e['frame_id'],'ground_fp':int((obs&(truth==0)).sum()),'hardware_fp':int((obs&(truth==2)).sum()),'sky_fp':int((obs&(truth==3)).sum()),'ground_as_hardware':int((hardware&(truth==0)).sum()),'rock_tp':int((obs&(truth==1)).sum()),'ground_pixels':int((truth==0).sum()),'hardware_pixels':int((truth==2).sum()),'sky_pixels':int((truth==3).sum()),'rock_pixels':int((truth==1).sum())}
        for k in totals:totals[k]+=row[k]
        for o in e['objects']:
            x0,y0,x1,y1=o['bbox'];core=truth[y0:y1,x0:x1]==1;minimum=max(3,int(.1*core.sum()));hit=int((obs[y0:y1,x0:x1]&core).sum())>=minimum;key=f"{e['index']}:{o['key']}"
            if hit:
                regions_found.append(key)
                if o.get('kind')=='manual_visual_candidate':manual_found.append(key)
        rows.append(row)
    diag=np.diag(cm);iou=diag/np.maximum(1,cm.sum(0)+cm.sum(1)-diag);report={**totals,'ground_fp_fraction':totals['ground_fp']/max(1,totals['ground_pixels']),'hardware_fp_fraction':totals['hardware_fp']/max(1,totals['hardware_pixels']),'ground_as_hardware_fraction':totals['ground_as_hardware']/max(1,totals['ground_pixels']),'rock_pixel_recall':totals['rock_tp']/max(1,totals['rock_pixels']),'iou':iou.tolist(),'confusion_matrix':cm.tolist(),'manual_regions_found':sorted(manual_found),'all_regions_found':sorted(regions_found),'positive_annotation_regions':sum(len(e['objects']) for e in entries),'frames_with_reviewed_false_alarm':sum(r['ground_fp']+r['hardware_fp']+r['sky_fp']>0 for r in rows),'frames_evaluated':len(entries),'population':'demo fit/conformance, partial reviewed/weak labels, not an independent test','frames':rows}
    return report
def compact(score):return {k:v for k,v in score.items() if k not in ['frames','all_regions_found','manual_regions_found']}
def objective(s):return .45*s['iou'][1]+.2*s['iou'][2]+.15*s['iou'][0]+.2*s['rock_pixel_recall']-2*s['ground_fp_fraction']-2*s['hardware_fp_fraction']-s['ground_as_hardware_fraction']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--init',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';torch.backends.cudnn.benchmark=True
    manifest=json.loads(a.manifest.read_text('utf8'));assert manifest['status']=='final_reviewed' and len(manifest['entries'])==181;entries=manifest['entries']
    for e in entries:
        for key,h in [('image','image_sha256'),('label','label_sha256'),('weight','weight_sha256')]:assert sha(e[key])==e[h]
    baseline_sha=sha(R/'data/prepared/navigation.onnx');terrain_sha=sha(R/'data/prepared/surface.onnx');init_sha=sha(a.init);protocol=R/'data/annotations/live-demo-v12-protocol.json';settings={'device':device,'epochs':12,'samples':1086,'seed':2026,'size':512,'head_lr':1e-4,'encoder_lr':1e-5,'manifest_sha256':sha(a.manifest),'protocol_sha256':sha(protocol),'trainer_sha256':sha(Path(__file__)),'initial_checkpoint_sha256':init_sha,'original_navigation_sha256':baseline_sha,'terrain_sha256':terrain_sha,'population':'All181 demo frames fit and select the model. No held-out validation/test claim.','runtime':'Actual four-class ONNX; no frame-ID lookup or annotation substitution.'};dump(a.output/'settings.json',settings);print('SETTINGS',json.dumps(settings),flush=True)
    model=DemoNavigation(a.init).to(device);before=evaluate_demo(model,entries,device);dump(a.output/'initial-demo-conformance.json',before);baseline_manual=set(before['manual_regions_found']);best=objective(before);best_epoch=0;best_unconstrained=best;torch.save(model.state_dict(),a.output/'best-navigation.pt');torch.save(model.state_dict(),a.output/'best-unconstrained.pt');print('INITIAL',json.dumps(compact(before)),flush=True)
    loader=DataLoader(DemoData(entries),batch_size=2,shuffle=True,num_workers=2,persistent_workers=True,pin_memory=device=='cuda',worker_init_fn=seed_worker);opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-5},{'params':model.head.parameters(),'lr':1e-4}],weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,12,eta_min=5e-7);scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda');weights=torch.tensor([1,3,1.5,1],device=device);history=[];start=time.perf_counter()
    for epoch in range(12):
        model.train();opt.zero_grad();running=0;count=0
        for i,(im,y,pixel_weight) in enumerate(loader):
            im,y,pixel_weight=im.to(device,non_blocking=True),y.to(device,non_blocking=True),pixel_weight.to(device,non_blocking=True);valid=y!=255
            if not valid.any():continue
            with torch.amp.autocast(device_type=device,enabled=device=='cuda'):
                logits=model(im);p=logits.softmax(1);ce=F.cross_entropy(logits,y,weight=weights,ignore_index=255,reduction='none');ce=(ce*pixel_weight).sum()/pixel_weight.sum().clamp_min(1)
                target=F.one_hot(y.clamp(0,3),4).permute(0,3,1,2).float()*valid[:,None];pr=p.float()*valid[:,None];dice=1-((2*(pr*target).sum((0,2,3))+1)/(pr.sum((0,2,3))+target.sum((0,2,3))+1))[1:3].mean()
                zero=logits.reshape(-1)[0]*0;ground=y==0;rock=y==1;gear=y==2
                gfp=p[:,1][ground].mean() if ground.any() else zero;hfp=p[:,1][gear].mean() if gear.any() else zero;ghw=p[:,2][ground].mean() if ground.any() else zero;miss=(1-p[:,1][rock]).mean() if rock.any() else zero;hmiss=(1-p[:,2][gear]).mean() if gear.any() else zero
                loss=(ce+.35*dice+.75*gfp+.75*hfp+.5*ghw+miss+.5*hmiss)/4
            scaler.scale(loss).backward()
            if (i+1)%4==0 or i+1==len(loader):
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update();opt.zero_grad()
            running+=float(loss.detach())*4;count+=1
            if (i+1)%136==0:print('BATCH',epoch+1,i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
        score=evaluate_demo(model,entries,device);value=objective(score);lost=sorted(baseline_manual-set(score['manual_regions_found']));eligible=not lost and score['rock_pixel_recall']>=before['rock_pixel_recall']-.02;improved=eligible and value>best
        if value>best_unconstrained:best_unconstrained=value;torch.save(model.state_dict(),a.output/'best-unconstrained.pt');dump(a.output/'best-unconstrained-conformance.json',score)
        if improved:best=value;best_epoch=epoch+1;torch.save(model.state_dict(),a.output/'best-navigation.pt');dump(a.output/'best-demo-conformance.json',score)
        torch.save(model.state_dict(),a.output/'last-navigation.pt');row={'epoch':epoch+1,'loss':running/max(1,count),'demo':compact(score),'score':value,'eligible':eligible,'lost_baseline_manual_regions':lost,'selected':improved,'elapsed_s':time.perf_counter()-start};history.append(row);dump(a.output/'history.json',history);print('EPOCH',json.dumps(row),flush=True);scheduler.step()
    model.load_state_dict(torch.load(a.output/'best-navigation.pt',map_location=device,weights_only=True));final=evaluate_demo(model,entries,device);dump(a.output/'selected-demo-conformance.json',final);model=model.cpu().eval();torch.onnx.export(model,torch.randn(1,3,512,512),str(a.output/'navigation.onnx'),opset_version=17,input_names=['image'],output_names=['navigation_logits'],dynamo=False,external_data=False)
    assert sha(R/'data/prepared/navigation.onnx')==baseline_sha and sha(R/'data/prepared/surface.onnx')==terrain_sha and sha(a.init)==init_sha
    report={'best_epoch':best_epoch,'epochs_completed':len(history),'elapsed_s':time.perf_counter()-start,'candidate_sha256':sha(a.output/'navigation.onnx'),'original_navigation_unchanged':True,'global_terrain_unchanged':True,'original_checkpoint_unchanged':True,'population':'Demo fit/conformance only; no heldout test','selected':compact(final),'baseline':compact(before),'candidate_only':True};dump(a.output/'candidate-report.json',report);print('COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()
