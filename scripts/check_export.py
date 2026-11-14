import json,os
from pathlib import Path
_dll_handle=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
import torch
import numpy as np
import onnxruntime as ort
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from train_surface import TRAINING,MarsDataset,OUT

torch.set_num_threads(1)
m=lraspp_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=4)
m.load_state_dict(torch.load(TRAINING/'best-surface.pt',map_location='cpu',weights_only=True));m.eval()
entries=json.loads((TRAINING/'manifest.json').read_text('utf-8'))['entries']
ds=MarsDataset([e for e in entries if e['split']=='test'],size=320)
op=ort.SessionOptions();op.intra_op_num_threads=1
session=ort.InferenceSession(str(OUT/'surface.onnx'),sess_options=op,providers=['CPUExecutionProvider'])
for i in range(3):
    x,y=ds[i]
    with torch.no_grad():ref=m(x[None])['out'].numpy()
    output=session.run(None,{'image':x.numpy()[None]})[0]
    print({'frame':i,'max_abs':float(np.max(np.abs(ref-output))),'mean_abs':float(np.mean(np.abs(ref-output))),'pixel_agreement':float((ref.argmax(1)==output.argmax(1)).mean()),'reference_range':[float(ref.min()),float(ref.max())],'onnx_range':[float(output.min()),float(output.max())]},flush=True)
