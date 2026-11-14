"""Fine-tune a small, genuine AI4Mars semantic model and verify ONNX export.

No canned metrics, synthetic labels or test-set threshold tuning.
"""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
_dll_handle = os.add_dll_directory(os.environ["MARS_DLL_DIR"]) if os.name == "nt" and os.environ.get("MARS_DLL_DIR") else None
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

ROOT=Path(__file__).resolve().parents[1]
TRAINING=Path(os.environ.get("LOCALAPPDATA","."))/"mars-pathfinder"/"training"
OUT=ROOT/"data"/"prepared"
CLASSES=["soil","bedrock","sand","big_rock"]


class MarsDataset(Dataset):
    def __init__(self,entries,size=384,augment=False): self.entries,self.size,self.augment=entries,size,augment
    def __len__(self):return len(self.entries)
    def __getitem__(self,i):
        e=self.entries[i]
        image=Image.open(e["image"]).convert("RGB").resize((self.size,self.size),Image.Resampling.BILINEAR)
        label=Image.open(e["label"]).convert("L").resize((self.size,self.size),Image.Resampling.NEAREST)
        if self.augment:
            if random.random()<.5: image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT);label=label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image=ImageEnhance.Brightness(image).enhance(random.uniform(.85,1.15))
            image=ImageEnhance.Contrast(image).enhance(random.uniform(.9,1.1))
        pixels=np.asarray(image,dtype=np.float32)/255
        pixels=(pixels-[.485,.456,.406])/[.229,.224,.225]
        target=np.asarray(label,dtype=np.int64).copy()
        if not np.isin(target,[0,1,2,3,255]).all():raise ValueError("Unexpected class IDs in publisher NAV labels")
        return torch.from_numpy(pixels.transpose(2,0,1).astype(np.float32)),torch.from_numpy(target)


def scores(confusion):
    intersection=np.diag(confusion)
    union=confusion.sum(0)+confusion.sum(1)-intersection
    iou=np.divide(intersection,union,out=np.full(4,np.nan),where=union>0)
    precision=np.divide(intersection,confusion.sum(0),out=np.zeros(4,dtype=float),where=confusion.sum(0)>0)
    recall=np.divide(intersection,confusion.sum(1),out=np.zeros(4,dtype=float),where=confusion.sum(1)>0)
    return {"miou":float(np.nanmean(iou)),"class_iou":{c:(float(v) if np.isfinite(v) else None) for c,v in zip(CLASSES,iou)},
            "precision":dict(zip(CLASSES,precision.tolist())),"recall":dict(zip(CLASSES,recall.tolist())),
            "f1":dict(zip(CLASSES,np.divide(2*precision*recall,precision+recall,out=np.zeros(4),where=precision+recall>0).tolist())),
            "confusion_matrix":confusion.tolist()}


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); confusion=np.zeros((4,4),dtype=np.int64)
    for image,target in loader:
        pred=model(image.to(device))["out"].argmax(1).cpu().numpy()
        target=target.numpy();valid=target!=255
        confusion+=np.bincount(4*target[valid]+pred[valid],minlength=16).reshape(4,4)
    return scores(confusion)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--epochs",type=int,default=10);parser.add_argument("--batch",type=int,default=2);parser.add_argument("--minutes",type=float,default=240);parser.add_argument("--export-only",action="store_true")
    args=parser.parse_args()
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.set_num_threads(4)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print({"torch":torch.__version__,"device":str(device),"gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None},flush=True)
    manifest=json.loads((TRAINING/"manifest.json").read_text("utf-8"))
    groups={split:{e["group"] for e in manifest["entries"] if e["split"]==split} for split in ["train","validation","test"]}
    assert not(groups["train"]&groups["validation"] or groups["train"]&groups["test"] or groups["validation"]&groups["test"]),"Split leakage"
    splits={s:[e for e in manifest["entries"] if e["split"]==s] for s in groups}
    hist=np.zeros(4,dtype=np.int64)
    for e in splits["train"]:
        label=np.asarray(Image.open(e["label"]))
        hist+=np.bincount(label[label<4].ravel(),minlength=4)
    weights=np.clip(np.sqrt(hist.sum()/np.maximum(hist,1)),1,10);weights/=weights.mean()
    print({"class_pixel_counts":hist.tolist(),"loss_weights":weights.tolist(),"split_counts":{s:len(v) for s,v in splits.items()}},flush=True)
    loaders={s:DataLoader(MarsDataset(v,augment=s=="train"),batch_size=args.batch,shuffle=s=="train",num_workers=0) for s,v in splits.items()}
    model=lraspp_mobilenet_v3_large(weights=None,weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2,num_classes=4).to(device)
    optimizer=torch.optim.AdamW([{"params":model.backbone.parameters(),"lr":1e-4},{"params":model.classifier.parameters(),"lr":1e-3}],weight_decay=1e-4)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    ce=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device),ignore_index=255)
    began=time.perf_counter();best=-1.;history=[]
    checkpoint=TRAINING/"best-surface.pt"
    for epoch in range(0 if args.export_only else args.epochs):
        for p in model.backbone.parameters():p.requires_grad=epoch>=2
        model.train()
        if epoch<2:model.backbone.eval()
        running=0.;batches=0;optimizer.zero_grad()
        for i,(image,label) in enumerate(loaders["train"]):
            image,label=image.to(device),label.to(device)
            if not (label!=255).any():continue
            with torch.amp.autocast(device_type=device.type,enabled=device.type=="cuda"):
                output=model(image)["out"]
                valid=(label!=255).unsqueeze(1)
                safe=label.clamp(0,3)
                target=F.one_hot(safe,4).permute(0,3,1,2).float()*valid
                probs=output.softmax(1)*valid
                dice=1-((2*(probs*target).sum((0,2,3))+1)/(probs.sum((0,2,3))+target.sum((0,2,3))+1)).mean()
                loss=(ce(output,label)+dice)/4
            scaler.scale(loss).backward()
            if (i+1)%4==0 or i+1==len(loaders["train"]):
                scaler.step(optimizer);scaler.update();optimizer.zero_grad()
            running+=float(loss.detach())*4;batches+=1
            if i%50==0:print(f"epoch{epoch+1} batch{i}/{len(loaders['train'])} loss={float(loss.detach())*4:.4f}",flush=True)
        val=evaluate(model,loaders["validation"],device)
        row={"epoch":epoch+1,"train_loss":running/max(batches,1),"validation":val,"elapsed_s":time.perf_counter()-began};history.append(row)
        (TRAINING/'history.json').write_text(json.dumps(history,indent=2),'utf-8')
        print(json.dumps(row),flush=True)
        if val["miou"]>best:
            best=val["miou"];torch.save(model.state_dict(),checkpoint)
        if time.perf_counter()-began>args.minutes*60:break
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    progress=json.loads((TRAINING/'run.json').read_text('utf-8')) if args.export_only and (TRAINING/'run.json').exists() else {"epochs_completed":len(history),"elapsed_training_s":time.perf_counter()-began}
    test=evaluate(model,loaders["test"],device)
    model=model.cpu().eval()
    class Export(nn.Module):
        def __init__(self,m):super().__init__();self.m=m
        def forward(self,x):return self.m(x)["out"]
    OUT.mkdir(parents=True,exist_ok=True)
    export_model=Export(model).eval()
    torch.onnx.export(export_model,torch.randn(1,3,320,320),str(OUT/"surface.onnx"),opset_version=17,input_names=["image"],output_names=["logits"],dynamo=False,external_data=False)
    model.eval()  # Keep BatchNorm in eval after the exporter restores wrapper state.
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(OUT/"surface.onnx"),sess_options=options,providers=["CPUExecutionProvider"])
    ds=MarsDataset(splits["test"],size=320)
    confusion=np.zeros((4,4),dtype=np.int64);latencies=[];agreement=[]
    for i in range(len(ds)):
        x,y=ds[i];x=x.numpy()[None]
        tick=time.perf_counter();logits=session.run(None,{"image":x})[0];latencies.append((time.perf_counter()-tick)*1000)
        prediction=logits.argmax(1)[0];target=y.numpy();valid=target!=255
        confusion+=np.bincount(4*target[valid]+prediction[valid],minlength=16).reshape(4,4)
        if i<3:
            with torch.no_grad():reference=model(torch.from_numpy(x))["out"].numpy()
            agreement.append(float((reference.argmax(1)==logits.argmax(1)).mean()))
    if min(agreement)<.995:raise RuntimeError("ONNX parity check failed")
    card={"ready":True,"name":"LR-ASPP · MobileNetV3","status":"Fine-tuned on real AI4Mars v0.6 MSL/Navcam subset",
          "classes":CLASSES,"input_size":320,"threshold":.7,"threshold_note":"Conservative display threshold; not calibrated as occupancy probability",
          "training_size":384,"epochs_completed":progress['epochs_completed'],"training_device":str(device),"elapsed_training_s":round(progress['elapsed_training_s'],1),
          "optimizer":"AdamW","learning_rates":{"backbone":1e-4,"head":1e-3},"loss":"weighted CE + Dice","batch_size":args.batch,"gradient_accumulation":4,
          "split_counts":{s:len(v) for s,v in splits.items()},"split_method":manifest["split_method"],"seed":42,
          "dataset":"AI4Mars merged v0.6","dataset_url":manifest["source"],"dataset_license":"CC-BY-4.0",
          "metrics":scores(confusion),"pytorch_test_384":test,"onnx_parity_pixel_agreement":agreement,
          "local_cpu_latency_ms":{"p50":float(np.median(latencies)),"p95":float(np.percentile(latencies,95)),"threads":1,"note":"Local CPU measurement; not Render Free benchmark"},
          "model_sha256":hashlib.file_digest((OUT/"surface.onnx").open("rb"),"sha256").hexdigest(),
          "limitations":[manifest["note"],"Small preliminary subset; no generalization or human-safety guarantee","Only4 NAV classes; no small-stone class","RAD archive frames have different photometric processing from training EDR images","Camera-orbital registration is measured separately by image correlation; this card covers segmentation only"]}
    (OUT/"model-card.json").write_text(json.dumps(card,indent=2),"utf-8")
    if history:(OUT/"training-history.json").write_text(json.dumps(history,indent=2),"utf-8")
    print(json.dumps(card,indent=2),flush=True)


if __name__=="__main__":main()
