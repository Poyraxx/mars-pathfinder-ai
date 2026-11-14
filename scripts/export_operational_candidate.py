"""Export identical trained weights at another resolution for validation only."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
_dll=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
import torch
from operational_model import OperationalSurface


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--size',type=int,required=True)
    ap.add_argument('--navigation-only',action='store_true');args=ap.parse_args()
    assert args.size%32==0 and 256<=args.size<=640
    args.output.mkdir(exist_ok=True,parents=True);torch.set_num_threads(2)
    settings=json.loads((args.source/'settings.json').read_text('utf8'))
    settings.update(export_size=args.size,training_size=settings['size'],export_source=str(args.source))
    model=OperationalSurface(navigation_classes=len(settings['navigation_classes']),spatial_refinement=settings.get('spatial_refinement',False),separate_navigation_encoder=settings.get('train_navigation_encoder',False)).eval()
    checkpoint=args.source/'best-operational.pt';model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    settings['navigation_only']=args.navigation_only
    shutil.copyfile(checkpoint,args.output/'best-operational.pt')
    (args.output/'settings.json').write_text(json.dumps(settings,indent=2),'utf8')
    path=args.output/'surface.onnx'
    class NavigationOnly(torch.nn.Module):
        def __init__(self,combined):
            super().__init__();self.encoder=combined.navigation_encoder or combined.base.network.segformer;self.head=combined.navigation_head
        def forward(self,image):
            hidden=self.encoder(pixel_values=image,output_hidden_states=True,return_dict=True).hidden_states
            return torch.nn.functional.interpolate(self.head(hidden),size=image.shape[-2:],mode='bilinear',align_corners=False)
    exported=NavigationOnly(model).eval() if args.navigation_only else model
    torch.onnx.export(exported,torch.randn(1,3,args.size,args.size),str(path),opset_version=17,input_names=['image'],output_names=['navigation_logits'] if args.navigation_only else ['logits','navigation_logits'],dynamo=False,external_data=False)
    report={'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'input_size':args.size,'navigation_only':args.navigation_only,'weights_changed':False,'candidate_only':True,'test_evaluated':False}
    (args.output/'export-report.json').write_text(json.dumps(report,indent=2),'utf8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
