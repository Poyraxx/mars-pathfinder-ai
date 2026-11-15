"""Candidate-only rock refinement: native crops, rock-balanced batches, untouched test.

No candidate is copied into data/prepared. Model selection uses validation only.
Run with the project's CUDA venv and an explicit frozen manifest/checkpoint.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
_dll = os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name == 'nt' and os.environ.get('MARS_DLL_DIR') else None
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from train_surface import CLASSES, scores, evaluate, MarsDataset

PRETRAINED = 'nvidia/segformer-b0-finetuned-ade-512-512'
REVISION = '489d5cd81a0b59fab9b7ea758d3548ebe99677da'


class SegformerSurface(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        kwargs = {'num_labels':4, 'id2label':dict(enumerate(CLASSES)), 'label2id':{v:k for k,v in enumerate(CLASSES)}}
        if pretrained:
            self.network = SegformerForSemanticSegmentation.from_pretrained(PRETRAINED, revision=REVISION,
                use_safetensors=True, ignore_mismatched_sizes=True, **kwargs)
        else:
            config = SegformerConfig.from_pretrained(PRETRAINED, revision=REVISION, **kwargs)
            self.network = SegformerForSemanticSegmentation(config)
    @property
    def backbone(self): return self.network.segformer
    @property
    def classifier(self): return self.network.decode_head
    def forward(self, image):
        logits = self.network(pixel_values=image).logits
        return {'out':F.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)}


def make_model(architecture, pretrained=False):
    if architecture == 'segformer-b0': return SegformerSurface(pretrained)
    return lraspp_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=4)


class RockDataset(Dataset):
    def __init__(self, entries, size=512):
        self.entries, self.size = entries, size
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, index):
        entry = self.entries[index]
        image=Image.open(entry['image']).convert('RGB')
        label=Image.open(entry['label']).convert('L')
        if image.size != label.size:
            raise ValueError('Source image and label are not pixel-aligned')
        width,height=image.size
        boxes=entry.get('rock_boxes',[])
        if boxes and random.random()<.70:
            box=random.choices(boxes,weights=[1/max(8,b[4])**.35 for b in boxes])[0]
            x,y,w,h,_=box
            crop=min(width,height,max(max(w,h)*1.8,random.choice([320,448,640,800])))
            cx=x+w/2+random.uniform(-.18,.18)*crop
            cy=y+h/2+random.uniform(-.18,.18)*crop
            left=int(np.clip(cx-crop/2,0,width-crop));top=int(np.clip(cy-crop/2,0,height-crop))
            rect=(left,top,int(left+crop),int(top+crop))
            image=image.crop(rect);label=label.crop(rect)
        elif random.random()<.30:
            crop=int(min(width,height)*random.uniform(.55,1.))
            left=random.randint(0,width-crop);top=random.randint(0,height-crop)
            rect=(left,top,left+crop,top+crop);image=image.crop(rect);label=label.crop(rect)
        image=image.resize((self.size,self.size),Image.Resampling.BILINEAR)
        label=label.resize((self.size,self.size),Image.Resampling.NEAREST)
        if random.random()<.5:
            image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT);label=label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image=ImageEnhance.Brightness(image).enhance(random.uniform(.75,1.25))
        image=ImageEnhance.Contrast(image).enhance(random.uniform(.8,1.3))
        pixels=np.asarray(image,dtype=np.float32)/255.
        pixels=np.power(pixels,random.uniform(.8,1.25))
        pixels=(pixels-[.485,.456,.406])/[.229,.224,.225]
        target=np.asarray(label,dtype=np.int64).copy()
        if not np.isin(target,[0,1,2,3,255]).all():
            raise ValueError('Invalid semantic class in source label')
        return torch.from_numpy(pixels.transpose(2,0,1).astype(np.float32)),torch.from_numpy(target)


def seed_worker(_):
    torch.set_num_threads(1)
    seed=torch.initial_seed()%2**32;np.random.seed(seed);random.seed(seed)


def objective(s,aux=None):
    rock=s['class_iou']['big_rock'] or 0.
    value=.5*rock+.3*s['recall']['big_rock']+.2*s['miou']-.3*max(0.,.35-s['precision']['big_rock'])
    if aux:
        matrix=np.asarray(aux['confusion_matrix']);neg=matrix[:3].sum();fp=matrix[:3,3].sum()/max(1,neg)
        value+=.4*(.5*aux['recall']['big_rock']+.3*(aux['class_iou']['big_rock'] or 0.)+.2*(1-fp))
    return value


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--init',type=Path)
    ap.add_argument('--architecture',choices=['lraspp','segformer-b0'],default='lraspp')
    ap.add_argument('--aux-validation',type=Path,help='Separate sparse camera-domain validation; never its test entries')
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--size',type=int,default=512)
    ap.add_argument('--batch',type=int,default=2);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--minutes',type=float,default=120.)
    ap.add_argument('--lr',type=float,default=2e-4);ap.add_argument('--rock-focus',type=float,default=.5);ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest=json.loads(args.manifest.read_text('utf8'));entries=manifest['entries']
    train=[e for e in entries if e['split']=='train'];validation=[e for e in entries if e['split']=='validation']
    groups={s:{e['group'] for e in entries if e['split']==s} for s in ('train','validation','test')}
    assert not (groups['train']&groups['validation'] or groups['train']&groups['test'] or groups['validation']&groups['test'])
    assert train and validation
    settings={**vars(args),'manifest':str(args.manifest),'init':str(args.init),'output':str(args.output),'device':str(device),
              'selection_split':'validation','test_evaluated':False,'train_images':len(train),'validation_images':len(validation),
              'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),'init_sha256':hashlib.sha256(args.init.read_bytes()).hexdigest() if args.init else None,
              'pretrained_repository':PRETRAINED if args.architecture=='segformer-b0' else None,
              'pretrained_revision':REVISION if args.architecture=='segformer-b0' else None,
              'aux_validation':str(args.aux_validation) if args.aux_validation else None}
    (args.output/'settings.json').write_text(json.dumps(settings,indent=2),'utf8')
    print(json.dumps(settings),flush=True)
    hist=np.zeros(4,np.int64)
    for entry in train:
        label=np.asarray(Image.open(entry['label']));hist+=np.bincount(label[label<4].ravel(),minlength=4)
    weights=np.clip(np.sqrt(hist.sum()/np.maximum(hist,1)),1,12);weights/=weights.mean()
    sample_weights=[float(e.get('sampling_weight',2. if e.get('rock_boxes') else 1.)) for e in train]
    sampler=WeightedRandomSampler(sample_weights,len(train),replacement=True,generator=torch.Generator().manual_seed(args.seed))
    options={'batch_size':args.batch,'num_workers':args.workers,'pin_memory':device.type=='cuda','worker_init_fn':seed_worker}
    if args.workers:options['persistent_workers']=True
    train_loader=DataLoader(RockDataset(train,args.size),sampler=sampler,**options)
    val_loader=DataLoader(MarsDataset(validation,size=args.size),shuffle=False,**options)
    aux_loader=None
    if args.aux_validation:
        aux=[e for e in json.loads(args.aux_validation.read_text('utf8'))['entries'] if e['split']=='validation']
        assert aux and not ({e['group'] for e in aux}&groups['train'])
        aux_loader=DataLoader(MarsDataset(aux,size=args.size),shuffle=False,**options)
    if not args.init and args.architecture=='lraspp': raise ValueError('LRASPP requires a frozen checkpoint')
    model=make_model(args.architecture,pretrained=not args.init).to(device)
    if args.init:model.load_state_dict(torch.load(args.init,map_location=device,weights_only=True))
    baseline=evaluate(model,val_loader,device)
    baseline_aux=evaluate(model,aux_loader,device) if aux_loader else None
    if baseline_aux:(args.output/'baseline-aux-validation.json').write_text(json.dumps(baseline_aux,indent=2),'utf8')
    (args.output/'baseline-validation.json').write_text(json.dumps(baseline,indent=2),'utf8')
    print('BASELINE_VALIDATION',json.dumps(baseline),flush=True)
    optimizer=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':args.lr*.1},{'params':model.classifier.parameters(),'lr':args.lr}],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=args.lr*.005)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    ce=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device),ignore_index=255)
    best=objective(baseline,baseline_aux);best_epoch=0;best_scores=baseline;best_aux=baseline_aux;torch.save(model.state_dict(),args.output/'best-surface.pt')
    history=[];start=time.perf_counter()
    for epoch in range(args.epochs):
        for p in model.backbone.parameters():p.requires_grad=epoch>=1
        model.train()
        if epoch==0:model.backbone.eval()
        optimizer.zero_grad();running=0.;count=0
        for i,(image,label) in enumerate(train_loader):
            image,label=image.to(device,non_blocking=True),label.to(device,non_blocking=True)
            if not (label!=255).any():continue
            with torch.amp.autocast(device_type=device.type,enabled=device.type=='cuda'):
                logits=model(image)['out'];valid=(label!=255).unsqueeze(1)
                target=F.one_hot(label.clamp(0,3),4).permute(0,3,1,2).float()*valid
                probability=logits.softmax(1)*valid
                dice=1-((2*(probability*target).sum((0,2,3))+1)/(probability.sum((0,2,3))+target.sum((0,2,3))+1)).mean()
                rock=probability[:,3];truth=target[:,3]
                tp=(rock*truth).sum();fp=(rock*(1-truth)).sum();fn=((1-rock)*truth).sum()
                tversky=(1-(tp+1)/(tp+.3*fp+.7*fn+1))**.75
                loss=(ce(logits,label)+.5*dice+args.rock_focus*tversky)/4
            scaler.scale(loss).backward()
            if (i+1)%4==0 or i+1==len(train_loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad()
            running+=float(loss.detach())*4;count+=1
            if i%25==0:print(f'epoch={epoch+1} batch={i}/{len(train_loader)} loss={float(loss.detach())*4:.4f} elapsed_s={time.perf_counter()-start:.1f}',flush=True)
        validation_scores=evaluate(model,val_loader,device)
        aux_scores=evaluate(model,aux_loader,device) if aux_loader else None
        metric=objective(validation_scores,aux_scores)
        quality_gate=True
        if aux_scores:
            cm=np.asarray(aux_scores['confusion_matrix']);negative_fp=cm[:3,3].sum()/max(1,cm[:3].sum())
            quality_gate=validation_scores['miou']>=baseline['miou']-.03 and (validation_scores['class_iou']['big_rock'] or 0.)>=(baseline['class_iou']['big_rock'] or 0.)-.03 and negative_fp<=.05
        improved=quality_gate and metric>best
        if improved:best=metric;best_epoch=epoch+1;best_scores=validation_scores;best_aux=aux_scores;torch.save(model.state_dict(),args.output/'best-surface.pt')
        row={'epoch':epoch+1,'loss':running/max(count,1),'validation':validation_scores,'aux_validation':aux_scores,'quality_gate':quality_gate,'selection_score':metric,'improved':improved,'elapsed_s':time.perf_counter()-start}
        history.append(row);(args.output/'history.json').write_text(json.dumps(history,indent=2),'utf8');print('EPOCH',json.dumps(row),flush=True)
        scheduler.step()
        if time.perf_counter()-start>args.minutes*60:break
    model.load_state_dict(torch.load(args.output/'best-surface.pt',map_location=device,weights_only=True));model=model.cpu().eval()
    class Export(nn.Module):
        def __init__(self,network):super().__init__();self.network=network
        def forward(self,image):return self.network(image)['out']
    torch.onnx.export(Export(model).eval(),torch.randn(1,3,args.size,args.size),str(args.output/'surface.onnx'),opset_version=17,input_names=['image'],output_names=['logits'],dynamo=False,external_data=False)
    report={'candidate_only':True,'test_evaluated':False,'evaluation_split':'validation','input_size':args.size,'baseline_validation':baseline,
            'best_selection_score':best,'best_epoch':best_epoch,'best_validation':best_scores,'architecture':args.architecture,
            'best_aux_validation':best_aux,
            'epochs_completed':len(history),'elapsed_s':time.perf_counter()-start,
            'model_sha256':hashlib.sha256((args.output/'surface.onnx').read_bytes()).hexdigest()}
    (args.output/'candidate-report.json').write_text(json.dumps(report,indent=2),'utf8');print('COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
