"""Research-only four-class compatibility export of a five-class navigation head.

Additional-rock appearance4 joins ground0 for the HISTORICAL operational task;
the raw five-class export is retained separately for appearance evaluation.
This operation does not assert metric size or replace any production file.
"""
import argparse,hashlib,json,os
from pathlib import Path
_dll=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
import torch
from operational_model import OperationalSurface

class CompatibleNavigation(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.encoder=model.navigation_encoder;self.head=model.navigation_head
    def forward(self,image):
        hidden=self.encoder(pixel_values=image,output_hidden_states=True,return_dict=True).hidden_states
        logits=torch.nn.functional.interpolate(self.head(hidden),size=image.shape[-2:],mode='bilinear',align_corners=False)
        return torch.cat([torch.logsumexp(logits[:,[0,4]],dim=1,keepdim=True),logits[:,1:4]],dim=1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    settings=json.loads((args.source/'settings.json').read_text('utf8'));assert len(settings['navigation_classes'])==5
    model=OperationalSurface(navigation_classes=5,spatial_refinement=True,separate_navigation_encoder=True).eval()
    checkpoint=args.source/'best-operational.pt';model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));torch.set_num_threads(2)
    args.output.mkdir(exist_ok=True,parents=True);path=args.output/'surface.onnx'
    torch.onnx.export(CompatibleNavigation(model).eval(),torch.randn(1,3,512,512),str(path),opset_version=17,input_names=['image'],output_names=['navigation_logits'],dynamo=False,external_data=False)
    report={'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'candidate_only':True,'input_size':512,'compatibility':'logsumexp ground0+additional_rock4; original obstacle1/hardware2/sky3 preserved as channels','production_file_replaced':False}
    (args.output/'export-report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
