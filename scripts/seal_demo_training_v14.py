"""Persist training diagnostics and an explicit original-model retention decision."""
import argparse,hashlib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1]
ORIGINAL='6dcf488232cc3d46cfe7250c1bbfe1311437c9f3f65555cd6858708602ec82b0'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,indent=2),encoding='utf8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,nargs='+',required=True);ap.add_argument('--keep-original',action='store_true');a=ap.parse_args();out=R/'data/research/live-demo-v14';out.mkdir(parents=True,exist_ok=True);rows=[]
    for run in a.runs:
        dest=out/run.name;dest.mkdir(exist_ok=True);report=json.loads((run/'candidate-report.json').read_text());history=json.loads((run/'history.json').read_text());settings=json.loads((run/'settings.json').read_text());assert len(history)==report['epochs_completed']==settings['epochs'];assert sha(run/'navigation.onnx')==report['candidate_sha256'];assert settings['original_navigation_sha256']==ORIGINAL
        names=['settings.json','history.json','candidate-report.json','initial-demo-conformance.json','initial-exclusion-conformance.json','initial-real-routes.json','selected-demo-conformance.json','selected-exclusion-conformance.json','selected-real-routes.json']
        for name in names:shutil.copyfile(run/name,dest/name)
        paired=run/'onnx-evaluation/paired-onnx-report.json'
        if paired.exists():
            for name in ['paired-onnx-report.json','baseline-onnx-conformance.json','candidate-onnx-conformance.json']:shutil.copyfile(paired.parent/name,dest/name)
        row={'run':run.name,'selected_epoch':report['best_epoch'],'epochs_completed':len(history),'selected_onnx_sha256':report['candidate_sha256'],'selected_is_original_model':report['candidate_sha256']==ORIGINAL,'selected_torch_route_tests_passed':report['real_routes']['all_passed'],'epochs_failing_real_routes':[h['epoch'] for h in history if not h['real_routes']['all_passed']],'epochs_losing_original_manual_regions':[h['epoch'] for h in history if h['lost_original_manual_regions']],'selected_checkpoint_sha256':sha(run/'best-navigation.pt'),'last_checkpoint_sha256':sha(run/'last-navigation.pt'),'manifest_sha256':settings['manifest_sha256'],'selected_ground_false_pixels':report['selected']['ground_fp'],'selected_hardware_false_pixels':report['selected']['hardware_fp'],'selected_ground_as_hardware_pixels':report['selected']['ground_as_hardware'],'selected_rock_pixel_recall':report['selected']['rock_pixel_recall'],'paired_onnx_report_available':paired.exists()}
        if paired.exists():row['paired_onnx_gates']=json.loads(paired.read_text())['gates']
        rows.append(row)
    result={'version':14,'population':'All181 demo sources used for fitting and selection. Partial labels, not held-out accuracy; no whole-image zero-error claim.','runs':rows,'production_navigation_sha256':sha(R/'data/prepared/navigation.onnx'),'global_surface_sha256':sha(R/'data/prepared/surface.onnx'),'installed_new_model':False,'original_tests_not_weakened':True,'labels_archive_sha256':sha(R/'data/annotations/live-demo-v14-masks.zip'),'reason':'Training diagnostics only. No candidate is authorized for promotion by this script.'}
    if a.keep_original:
        assert result['production_navigation_sha256']==ORIGINAL
        assert all(r['selected_is_original_model'] or r.get('paired_onnx_gates') and not all(r['paired_onnx_gates'].values()) for r in rows),'A potentially passing candidate requires remaining release checks, not automatic rejection'
        result['decision']='retain_original';result['reason']='No newly trained candidate passed all existing segmentation release gates. Original weights remain installed. Source annotations, rejected checkpoints and detailed real-route failures are preserved for further work.'
    dump(out/'training-result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
