"""Six prespecified epochs, validation-only selection, no fresh test access."""
import argparse,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageEnhance
_dll=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from appearance_model_v11 import RockAppearance,load_navigation
from train_rock_refinement import seed_worker
from train_operational_hazcam import state_sha

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor(im):
    a=np.asarray(im.resize((512,512),Image.Resampling.BILINEAR),np.float32)/255
    return torch.from_numpy(((a-[.485,.456,.406])/[.229,.224,.225]).transpose(2,0,1).astype('float32'))
class Reviewed(Dataset):
    def __init__(self,entries,samples):self.entries,self.samples=entries,samples
    def __len__(self):return self.samples
    def __getitem__(self,index):
        e=self.entries[index%len(self.entries)];im=Image.open(e['image']).convert('RGB');mask=Image.open(e['appearance_label'])
        if e['regions'] and random.random()<.6:
            r=random.choice(e['regions']);x0,y0,x1,y1=r['bbox'];side=int(min(1024,max(x1-x0,y1-y0,random.choice([384,640,1024]))))
            left=int(np.clip((x0+x1)/2-side/2,0,1024-side));top=int(np.clip((y0+y1)/2-side/2,0,1024-side))
            box=(left,top,left+side,top+side);im=im.crop(box);mask=mask.crop(box)
        if random.random()<.5:im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT);mask=mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        im=ImageEnhance.Brightness(im).enhance(random.uniform(.65,1.35));im=ImageEnhance.Contrast(im).enhance(random.uniform(.8,1.25))
        im=Image.fromarray((255*(np.asarray(im)/255.)**random.uniform(.8,1.25)).astype('uint8'))
        label=np.asarray(mask.resize((512,512),Image.Resampling.NEAREST),np.int64).copy()
        return tensor(im),torch.from_numpy(label)
@torch.no_grad()
def evaluate(model,entries,device):
    model.eval();cm=np.zeros((2,2),np.int64)
    for e in entries:
        pred=model(tensor(Image.open(e['image']).convert('RGB'))[None].to(device)).argmax(1)[0].cpu().numpy().astype('uint8')
        pred=np.asarray(Image.fromarray(pred).resize((1024,1024),Image.Resampling.NEAREST));truth=np.asarray(Image.open(e['appearance_label']));v=truth<2
        cm+=np.bincount(2*truth[v]+pred[v],minlength=4).reshape(2,2)
    tp=cm[1,1];return {'rock_iou':float(tp/max(1,cm[1].sum()+cm[:,1].sum()-tp)),'precision':float(tp/max(1,cm[:,1].sum())),'recall':float(tp/max(1,cm[1].sum())),'confusion_matrix':cm.tolist()}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--init',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    manifest=json.loads(a.manifest.read_text('utf8'));train=[e for e in manifest['entries'] if e['split']=='train'];val=[e for e in manifest['entries'] if e['split']=='validation']
    assert len(train)==84 and len(val)==8
    for e in train+val:assert sha(e['image'])==e['source_sha256'] and sha(e['appearance_label'])==e['appearance_label_sha256']
    old=load_navigation(a.init);frozen=state_sha(old);model=RockAppearance(old).to(device);encoder_sha=state_sha(model.encoder)
    settings={'manifest_sha256':sha(a.manifest),'init_sha256':sha(a.init),'old_complete_state_sha256':frozen,'encoder_sha256':encoder_sha,'epochs':6,'samples':1008,'size':512,'head_lr':2e-5,'selection':'binary masked validation rock IoU, initial checkpoint eligible','test_evaluated':False,'seed':2026,'train':len(train),'validation':len(val),'loss':'CE with sqrt batch inverse frequencies capped[.5,3]; .25 positive Dice; ignored255','crops':'384/640/1024, no128-to512 apparent-size amplification'}
    (a.output/'settings.json').write_text(json.dumps(settings,indent=2),'utf8');print(json.dumps(settings),flush=True)
    print('DEVICE',device,torch.cuda.get_device_name() if device=='cuda' else 'CPU',flush=True)
    loader=DataLoader(Reviewed(train,1008),batch_size=2,shuffle=True,num_workers=2,persistent_workers=True,pin_memory=device=='cuda',worker_init_fn=seed_worker)
    opt=torch.optim.AdamW(model.head.parameters(),lr=2e-5,weight_decay=1e-4);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6,eta_min=1e-6);scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
    before=evaluate(model,val,device);best=before['rock_iou'];bestepoch=0;torch.save(model.state_dict(),a.output/'best-appearance.pt');history=[];start=time.perf_counter()
    (a.output/'initial-validation.json').write_text(json.dumps(before,indent=2),'utf8');print('INITIAL',json.dumps(before),flush=True)
    for epoch in range(6):
        model.train();opt.zero_grad();running=0;count=0
        for i,(im,target) in enumerate(loader):
            im,target=im.to(device,non_blocking=True),target.to(device,non_blocking=True);valid=target!=255
            if not valid.any():continue
            with torch.amp.autocast(device_type=device,enabled=device=='cuda'):
                logits=model(im);hist=torch.stack([(target==c).sum() for c in range(2)]).float();weight=torch.sqrt(hist.sum()/hist.clamp_min(1)).clamp(.5,3);weight/=weight.mean()
                ce=F.cross_entropy(logits,target,weight=weight,ignore_index=255);prob=logits.softmax(1)[:,1];positive=target==1
                dice=1-(2*(prob*positive).sum()+1)/((prob*valid).sum()+positive.sum()+1);loss=(ce+.25*dice)/4
            scaler.scale(loss).backward()
            if (i+1)%4==0 or i+1==len(loader):
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.head.parameters(),5);scaler.step(opt);scaler.update();opt.zero_grad()
            running+=float(loss.detach())*4;count+=1
            if (i+1)%126==0:print('BATCH',epoch+1,i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
        score=evaluate(model,val,device);improved=score['rock_iou']>best
        if improved:best=score['rock_iou'];bestepoch=epoch+1;torch.save(model.state_dict(),a.output/'best-appearance.pt')
        row={'epoch':epoch+1,'loss':running/max(1,count),'validation':score,'improved':improved,'elapsed_s':time.perf_counter()-start};history.append(row);(a.output/'history.json').write_text(json.dumps(history,indent=2),'utf8');print('EPOCH',json.dumps(row),flush=True);schedule.step()
    assert state_sha(old)==frozen and state_sha(model.encoder)==encoder_sha
    model.load_state_dict(torch.load(a.output/'best-appearance.pt',map_location=device,weights_only=True));model=model.cpu().eval()
    torch.onnx.export(model,torch.randn(1,3,512,512),str(a.output/'appearance.onnx'),opset_version=17,input_names=['image'],output_names=['appearance_logits'],dynamo=False,external_data=False)
    result={'best_epoch':bestepoch,'best_validation_iou':best,'initial_validation_iou':before['rock_iou'],'old_navigation_state_unchanged':state_sha(old)==frozen,'encoder_unchanged':state_sha(model.encoder)==encoder_sha,'candidate_sha256':sha(a.output/'appearance.onnx'),'test_evaluated':False,'elapsed_s':time.perf_counter()-start}
    (a.output/'candidate-report.json').write_text(json.dumps(result,indent=2),'utf8');print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
