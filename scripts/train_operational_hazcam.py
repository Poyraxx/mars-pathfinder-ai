"""Train reviewed large-obstacle/ground/rover labels; leave terrain encoder fixed.

Candidates stay outside production. Full native validation drives selection;
heldout test scenes are never opened here.
"""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageEnhance
_dll=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from operational_model import OperationalSurface
from train_rock_refinement import make_model,seed_worker

NAMES=['ground_or_small_stone','substantial_obstacle','rover_hardware','sky_or_distant_unknown']


def state_sha(model):
    digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ReviewedDataset(Dataset):
    def __init__(self,entries,size=512,samples=None):self.entries,self.size,self.samples=entries,size,samples
    def __len__(self):return self.samples or len(self.entries)
    def __getitem__(self,index):
        entry=self.entries[index%len(self.entries)];source=Image.open(entry['image']).convert('RGB');label=Image.open(entry['label']).convert('L')
        assert source.size==label.size
        if self.samples:
            regions=entry['regions'];selected=random.choices(regions,weights=[2 if r['class_id'] in (1,2) else 1 for r in regions])[0] if random.random()<.6 else None
            if selected:
                x0,y0,x1,y1=selected['bbox'];side=min(min(source.size),max((x1-x0)*1.4,(y1-y0)*1.4,random.choice([448,640,800,1024])))
                cx=(x0+x1)/2+random.uniform(-.12,.12)*side;cy=(y0+y1)/2+random.uniform(-.12,.12)*side
                left=int(np.clip(cx-side/2,0,source.width-side));top=int(np.clip(cy-side/2,0,source.height-side))
                box=(left,top,int(left+side),int(top+side));source=source.crop(box);label=label.crop(box)
            if random.random()<.5:source=source.transpose(Image.Transpose.FLIP_LEFT_RIGHT);label=label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            source=ImageEnhance.Brightness(source).enhance(random.uniform(.65,1.35));source=ImageEnhance.Contrast(source).enhance(random.uniform(.75,1.3))
        source=source.resize((self.size,self.size),Image.Resampling.BILINEAR);label=label.resize((self.size,self.size),Image.Resampling.NEAREST)
        pixels=np.asarray(source,dtype='float32')/255.
        if self.samples:pixels=np.power(pixels,random.uniform(.8,1.25))
        pixels=((pixels-[.485,.456,.406])/[.229,.224,.225]).transpose(2,0,1).astype('float32')
        target=np.asarray(label,dtype='int64').copy();assert np.isin(target,[0,1,2,3,255]).all()
        return torch.from_numpy(pixels),torch.from_numpy(target)


def metrics(cm):
    n=len(cm)
    tp=np.diag(cm);union=cm.sum(0)+cm.sum(1)-tp
    iou=np.divide(tp,union,out=np.zeros(n),where=union>0)
    precision=np.divide(tp,cm.sum(0),out=np.zeros(n),where=cm.sum(0)>0)
    recall=np.divide(tp,cm.sum(1),out=np.zeros(n),where=cm.sum(1)>0)
    return {'miou':float(iou.mean()),'iou':dict(zip(NAMES,iou.tolist())),'precision':dict(zip(NAMES,precision.tolist())),
            'recall':dict(zip(NAMES,recall.tolist())),'hardware_as_obstacle_fraction':float(cm[2,1]/max(1,cm[2].sum())),
            'ground_as_obstacle_fraction':float(cm[0,1]/max(1,cm[0].sum())),'confusion_matrix':cm.tolist()}


@torch.no_grad()
def evaluate(model,entries,size,device):
    model.eval();cm=np.zeros((4,4),np.int64)
    for entry in entries:
        source=Image.open(entry['image']).convert('RGB');truth=np.asarray(Image.open(entry['label']))
        pixels=np.asarray(source.resize((size,size),Image.Resampling.BILINEAR),dtype='float32')/255
        x=((pixels-[.485,.456,.406])/[.229,.224,.225]).transpose(2,0,1)[None].astype('float32')
        pred=model(torch.from_numpy(x).to(device))[1].argmax(1)[0].cpu().numpy().astype('uint8')
        pred=np.asarray(Image.fromarray(pred).resize(source.size,Image.Resampling.NEAREST));valid=truth<4
        cm+=np.bincount(4*truth[valid]+pred[valid],minlength=16).reshape(4,4)
    return metrics(cm)


def objective(score):
    return (.5*score['iou']['substantial_obstacle']+.3*score['recall']['substantial_obstacle']+.2*score['iou']['rover_hardware']
            -2*score['hardware_as_obstacle_fraction']-.5*score['ground_as_obstacle_fraction'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--init',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--size',type=int,default=640);ap.add_argument('--epochs',type=int,default=24)
    ap.add_argument('--samples',type=int,default=440);ap.add_argument('--lr',type=float,default=.0005)
    ap.add_argument('--resume',type=Path);ap.add_argument('--spatial-refinement',action='store_true')
    ap.add_argument('--train-navigation-encoder',action='store_true')
    ap.add_argument('--ground-error-weight',type=float,default=.75)
    ap.add_argument('--obstacle-error-weight',type=float,default=1.)
    args=ap.parse_args()
    args.output.mkdir(exist_ok=True,parents=True);random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.backends.cudnn.benchmark=True
    manifest=json.loads(args.manifest.read_text('utf8'));train=[e for e in manifest['entries'] if e['split']=='train'];validation=[e for e in manifest['entries'] if e['split']=='validation']
    assert train and validation
    assert not ({e['group'] for e in train}&{e['group'] for e in manifest['entries'] if e['split']!='train'})
    settings={**vars(args),'manifest':str(args.manifest),'init':str(args.init),'output':str(args.output),'device':str(device),
              'encoder_frozen':not args.train_navigation_encoder,'terrain_encoder_frozen':True,'terrain_head_frozen':True,
              'selection_split':'validation','test_evaluated':False,'navigation_classes':NAMES,
              'resume':str(args.resume) if args.resume else None,
              'resume_sha256':hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None,
              'loss':f'weighted CE +0.5 Dice + hardware class error +{args.ground_error_weight} ground class error +{args.obstacle_error_weight} obstacle recall error',
              'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),'init_sha256':hashlib.sha256(args.init.read_bytes()).hexdigest()}
    (args.output/'settings.json').write_text(json.dumps(settings,indent=2),'utf8');print(json.dumps(settings),flush=True)
    base=make_model('segformer-b0');base.load_state_dict(torch.load(args.init,map_location='cpu',weights_only=True));frozen_sha=state_sha(base)
    model=OperationalSurface(base,spatial_refinement=args.spatial_refinement,separate_navigation_encoder=args.train_navigation_encoder)
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=True)
        if args.spatial_refinement and 'navigation_head.classifier.weight' in state:
            for suffix in ['weight','bias']:state['navigation_head.classifier.1.'+suffix]=state.pop('navigation_head.classifier.'+suffix)
        result=model.load_state_dict(state,strict=False)
        assert not result.unexpected_keys and all('navigation_head.classifier.0.' in k or k.startswith('navigation_encoder.') for k in result.missing_keys), 'Unexpected checkpoint mismatch'
        assert state_sha(model.base)==frozen_sha, 'Resume checkpoint changes the frozen terrain model'
    model=model.to(device)
    loader=DataLoader(ReviewedDataset(train,args.size,args.samples),batch_size=2,shuffle=True,num_workers=2,pin_memory=device.type=='cuda',persistent_workers=True,worker_init_fn=seed_worker)
    hist=np.zeros(4,np.int64)
    for entry in train:
        values=np.asarray(Image.open(entry['label']));hist+=np.bincount(values[values<4],minlength=4)
    weights=np.clip(np.sqrt(hist.sum()/np.maximum(hist,1)),1,6);weights/=weights.mean()
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device),ignore_index=255)
    parameters=[{'params':model.navigation_head.parameters(),'lr':args.lr}]
    if model.navigation_encoder is not None:parameters.append({'params':model.navigation_encoder.parameters(),'lr':args.lr*.1})
    optimizer=torch.optim.AdamW(parameters,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=args.lr*.05)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');history=[];start=time.perf_counter();best=-1e9;best_epoch=0;best_score=None
    before=evaluate(model,validation,args.size,device);(args.output/'initial-validation.json').write_text(json.dumps(before,indent=2),'utf8')
    best=objective(before);best_score=before;torch.save(model.state_dict(),args.output/'best-operational.pt')
    for epoch in range(args.epochs):
        model.train();running=0.;count=0;optimizer.zero_grad()
        for index,(image,label) in enumerate(loader):
            image,label=image.to(device,non_blocking=True),label.to(device,non_blocking=True)
            if not (label!=255).any():continue
            with torch.amp.autocast(device_type=device.type,enabled=device.type=='cuda'):
                logits=model(image)[1];valid=(label!=255).unsqueeze(1);target=F.one_hot(label.clamp(0,3),4).permute(0,3,1,2).float()*valid
                probability=logits.softmax(1)*valid
                dice=1-((2*(probability*target).sum((0,2,3))+1)/(probability.sum((0,2,3))+target.sum((0,2,3))+1)).mean()
                gear=label==2;gear_penalty=(1-probability[:,2][gear]).mean() if gear.any() else logits.sum()*0
                ground=label==0;ground_penalty=(1-probability[:,0][ground]).mean() if ground.any() else logits.sum()*0
                obstacle=label==1;obstacle_miss=(1-probability[:,1][obstacle]).mean() if obstacle.any() else logits.sum()*0
                loss=(criterion(logits,label)+.5*dice+gear_penalty+args.ground_error_weight*ground_penalty+args.obstacle_error_weight*obstacle_miss)/4
            scaler.scale(loss).backward()
            if (index+1)%4==0 or index+1==len(loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad()
            running+=float(loss.detach())*4;count+=1
        score=evaluate(model,validation,args.size,device);value=objective(score)
        improved=value>best
        if improved:best=value;best_score=score;best_epoch=epoch+1;torch.save(model.state_dict(),args.output/'best-operational.pt')
        row={'epoch':epoch+1,'loss':running/max(1,count),'validation':score,'selection_score':value,'improved':improved,'elapsed_s':time.perf_counter()-start}
        history.append(row);(args.output/'history.json').write_text(json.dumps(history,indent=2),'utf8');print('EPOCH',json.dumps(row),flush=True);scheduler.step()
    model.load_state_dict(torch.load(args.output/'best-operational.pt',map_location=device,weights_only=True));model=model.cpu().eval()
    torch.onnx.export(model,torch.randn(1,3,args.size,args.size),str(args.output/'surface.onnx'),opset_version=17,input_names=['image'],output_names=['logits','navigation_logits'],dynamo=False,external_data=False)
    report={'candidate_only':True,'test_evaluated':False,'input_size':args.size,'best_epoch':best_epoch,'best_validation':best_score,'epochs_completed':len(history),
            'elapsed_s':time.perf_counter()-start,'terrain_weights_unchanged':state_sha(model.base)==frozen_sha,
            'frozen_terrain_state_sha256':frozen_sha,
            'model_sha256':hashlib.sha256((args.output/'surface.onnx').read_bytes()).hexdigest()}
    (args.output/'candidate-report.json').write_text(json.dumps(report,indent=2),'utf8');print('COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
