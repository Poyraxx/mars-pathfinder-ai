"""Correct observed v23 appearance failures with partial class-exclusion losses."""
import argparse,json,os,random,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageEnhance
from train_live_demo_v12 import DemoNavigation,normalize,evaluate_demo,objective,compact,sha,dump,torch,F,nn
from train_rock_refinement import seed_worker
from torch.utils.data import Dataset,DataLoader
from backend.surface_inference import navigation_masks
from scipy import ndimage

class Data(Dataset):
    def __init__(self,entries,constraints):self.entries=entries;self.constraints=constraints
    def __len__(self):return 724
    def __getitem__(self,index):
        e=self.entries[index%181];c=self.constraints[e['index']];items=[Image.open(e['image']).convert('RGB')]+[Image.open(e[k]) for k in ['label','weight']]+[Image.open(c[k]) for k in ['not_class1','not_class2']]
        if random.random()<.25:items=[im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for im in items]
        items[0]=ImageEnhance.Brightness(items[0]).enhance(random.uniform(.95,1.05));items[0]=ImageEnhance.Contrast(items[0]).enhance(random.uniform(.95,1.05))
        y,w,n1,n2=[np.asarray(im.resize((512,512),Image.Resampling.NEAREST)).copy() for im in items[1:]]
        return normalize(items[0]),torch.from_numpy(y.astype('int64')),torch.from_numpy(w.astype('float32')/100),torch.from_numpy(n1.astype(bool)),torch.from_numpy(n2.astype(bool))

@torch.no_grad()
def exclusions(model,entries,cs,device):
    model.eval();totals={1:[0,0],2:[0,0]};rows=[]
    for e,c in zip(entries,cs):
        if not sum(c['negative_pixel_counts'].values()):continue
        im=Image.open(e['image']).convert('RGB');p=model(normalize(im)[None].to(device)).softmax(1)[0].cpu().numpy();obs,hw=navigation_masks(p,im.size,.7,.8,3);row={'index':e['index']}
        for cls,pred in [(1,obs),(2,hw)]:
            pred=np.asarray(Image.fromarray(pred).resize(im.size,Image.Resampling.NEAREST));mask=np.asarray(Image.open(c[f'not_class{cls}']),bool);fp=int((pred&mask).sum());n=int(mask.sum());totals[cls][0]+=fp;totals[cls][1]+=n;row[f'class{cls}_forbidden_pixels']=fp;row[f'class{cls}_reviewed_negative_pixels']=n
        rows.append(row)
    return {'clear_flat_rock_fraction':totals[1][0]/max(1,totals[1][1]),'navcam_hardware_fraction':totals[2][0]/max(1,totals[2][1]),'max_navcam_frame_hardware_fraction':max(r['class2_forbidden_pixels']/max(1,r['class2_reviewed_negative_pixels']) for r in rows),'totals':{str(k):{'false_pixels':v[0],'negative_pixels':v[1]} for k,v in totals.items()},'frames':rows,'label_type':'Manually reviewed class exclusion, not ground/safety truth'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--constraints',type=Path,required=True);ap.add_argument('--init',type=Path,required=True);ap.add_argument('--stage1',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    random.seed(2027);np.random.seed(2027);torch.manual_seed(2027);torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;device='cuda' if torch.cuda.is_available() else 'cpu'
    entries=json.loads(a.manifest.read_text('utf8'))['entries'];cs=json.loads(a.constraints.read_text('utf8'));assert cs['base_manifest_sha256']==sha(a.manifest)
    for e,c in zip(entries,cs['entries']):
        assert e['index']==c['index'] and e['image_sha256']==c['image_sha256']
        for k in ['not_class1','not_class2']:assert sha(c[k])==c[k+'_sha256']
    model=DemoNavigation(a.init);model.load_state_dict(torch.load(a.stage1,map_location='cpu',weights_only=True));model=model.to(device)
    before=evaluate_demo(model,entries,device);negative_before=exclusions(model,entries,cs['entries'],device);keep=set(before['manual_regions_found']);dump(a.output/'initial-demo-conformance.json',before);dump(a.output/'initial-exclusion-conformance.json',negative_before)
    settings={'epochs':6,'samples_per_epoch':724,'seed':2027,'input_size':512,'batch_size':2,'accumulation':4,'head_lr':3e-5,'encoder_lr':3e-6,'manifest_sha256':sha(a.manifest),'constraints_sha256':sha(a.constraints),'stage1_checkpoint_sha256':sha(a.stage1),'original_navigation_sha256':sha(Path(__file__).resolve().parents[1]/'data/prepared/navigation.onnx'),'trainer_sha256':sha(__file__),'population':'All181 demo sources plus manually source-reviewed complement labels; no held-out accuracy','selection':'Keep every stage1-found manual region and at most0.005 stage1 rock pixel recall loss, maximize original demo objective minus0.5 Navcam false hardware fraction minus0.25 clear-flat false rock fraction.'};dump(a.output/'settings.json',settings);print('INITIAL',json.dumps(negative_before),flush=True)
    def value(s,n):return objective(s)-.5*n['navcam_hardware_fraction']-.25*n['clear_flat_rock_fraction']
    best=value(before,negative_before);best_epoch=0;history=[];torch.save(model.state_dict(),a.output/'best-navigation.pt')
    loader=DataLoader(Data(entries,cs['entries']),batch_size=2,shuffle=True,num_workers=2,persistent_workers=True,pin_memory=True,worker_init_fn=seed_worker);opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':3e-6},{'params':model.head.parameters(),'lr':3e-5}],weight_decay=1e-4);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6,eta_min=3e-7);scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda');weights=torch.tensor([1,3,1.5,1],device=device);start=time.perf_counter()
    for epoch in range(6):
        model.train();opt.zero_grad();losses=[]
        for i,(im,y,w,n1,n2) in enumerate(loader):
            im,y,w,n1,n2=[x.to(device) for x in [im,y,w,n1,n2]];valid=y!=255
            with torch.amp.autocast(device_type=device,enabled=device=='cuda'):
                logits=model(im);p=logits.float().softmax(1);ce=F.cross_entropy(logits.float(),y,weight=weights,ignore_index=255,reduction='none');ce=(ce*w).sum()/w.sum().clamp_min(1);target=F.one_hot(y.clamp(0,3),4).permute(0,3,1,2).float()*valid[:,None];pr=p*valid[:,None];dice=1-((2*(pr*target).sum((0,2,3))+1)/(pr.sum((0,2,3))+target.sum((0,2,3))+1))[1:3].mean();zero=logits.reshape(-1)[0]*0
                def mean(prob,mask):return prob[mask].mean() if mask.any() else zero
                gfp=mean(p[:,1],y==0);hfp=mean(p[:,1],y==2);ghw=mean(p[:,2],y==0);miss=mean(1-p[:,1],y==1);hmiss=mean(1-p[:,2],y==2)
                no_rock=mean(-torch.log1p(-p[:,1].clamp(max=1-1e-6)),n1);no_hw=mean(-torch.log1p(-p[:,2].clamp(max=1-1e-6)),n2)
                loss=(ce+.35*dice+.75*gfp+.75*hfp+.5*ghw+miss+.5*hmiss+2*no_rock+4*no_hw)/4
            scaler.scale(loss).backward()
            if (i+1)%4==0 or i+1==len(loader):scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update();opt.zero_grad()
            losses.append(float(loss.detach())*4)
        score=evaluate_demo(model,entries,device);negative=exclusions(model,entries,cs['entries'],device);lost=sorted(keep-set(score['manual_regions_found']));eligible=not lost and score['rock_pixel_recall']>=before['rock_pixel_recall']-.005;v=value(score,negative);selected=eligible and v>best
        if selected:best=v;best_epoch=epoch+1;torch.save(model.state_dict(),a.output/'best-navigation.pt')
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),'demo':compact(score),'exclusions':{k:v for k,v in negative.items() if k!='frames'},'lost_stage1_manual_regions':lost,'eligible':eligible,'selected':selected,'score':v,'elapsed_s':time.perf_counter()-start};history.append(row);dump(a.output/'history.json',history);print('EPOCH',json.dumps(row),flush=True);schedule.step()
    model.load_state_dict(torch.load(a.output/'best-navigation.pt',map_location=device,weights_only=True));final=evaluate_demo(model,entries,device);negative=exclusions(model,entries,cs['entries'],device);dump(a.output/'selected-demo-conformance.json',final);dump(a.output/'selected-exclusion-conformance.json',negative);model=model.cpu().eval();torch.onnx.export(model,torch.randn(1,3,512,512),str(a.output/'navigation.onnx'),opset_version=17,input_names=['image'],output_names=['navigation_logits'],dynamo=False,external_data=False)
    dump(a.output/'candidate-report.json',{'best_epoch':best_epoch,'epochs_completed':6,'candidate_sha256':sha(a.output/'navigation.onnx'),'selected':compact(final),'baseline':compact(before),'exclusions':negative,'candidate_only':True,'population':'Demo conformance only, no independent test'});print('COMPLETE',best_epoch,sha(a.output/'navigation.onnx'),flush=True)
if __name__=='__main__':main()
