"""Consumed-test diagnostic: same binary head before/after fine-tuning.

This comparison is descriptive and cannot select another checkpoint or policy.
The navigation model has a different taxonomy and is not a fair binary baseline.
"""
import os,json,hashlib
from pathlib import Path
_dll=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
import torch
from appearance_model_v11 import load_navigation,RockAppearance
from train_appearance_v11 import evaluate
R=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training';C=T/'v22-v11-appearance'
def main():
    torch.set_num_threads(2);sealed=json.loads((C/'fresh-test-v11/evaluation.json').read_text('utf8'));assert hashlib.sha256((C/'appearance.onnx').read_bytes()).hexdigest()==sealed['candidate_sha256']
    entries=[e for e in json.loads((T/'manifest-operational-v11.json').read_text('utf8'))['entries'] if e['split']=='test_v11']
    model=RockAppearance(load_navigation(T/'v11-navigation-512/best-operational.pt'));before=evaluate(model,entries,'cpu');model.load_state_dict(torch.load(C/'best-appearance.pt',map_location='cpu',weights_only=True));after=evaluate(model,entries,'cpu')
    report={'test_is_consumed':True,'posttest_diagnostic_only':True,'checkpoint_or_policy_selected_here':False,'same_binary_taxonomy_preprocessing_argmax_labels':True,'initial_binary_head':before,'fine_tuned_binary_head':after,'candidate_sha256':sealed['candidate_sha256'],'interpretation':'Binary visible-rock appearance supervision, not obstacle clearance. This initialized two-logit head is not the deployed four-class navigation taxonomy. Sparse-interior metrics only.'}
    (R/'data/annotations/operational-v11-initial-comparison.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
