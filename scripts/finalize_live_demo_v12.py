"""Record actual release evidence; install only after every fixed gate passes."""
import argparse,hashlib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1]

def load(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--base-run',type=Path);ap.add_argument('--install',action='store_true');a=ap.parse_args();run=a.run
    paired=load(run/'paired-onnx-report.json');profile=load(run/'cpu-runtime.json');routes=load(run/'route-comparison.json');review=load(run/'release-visual-review.json');integrity=load(R/'data/annotations/live-demo-v12-integrity.json');trained=load(run/'candidate-report.json');settings=load(run/'settings.json')
    digest=sha(run/'navigation.onnx');assert digest==paired['candidate_sha256']==profile['model_sha256']==routes['candidate_sha256']==trained['candidate_sha256']==review['candidate_sha256']
    assert paired['baseline_sha256']==integrity['original_production_sha_expected'];assert paired['manifest_sha256']==integrity['manifest_sha256']==settings['manifest_sha256']
    original_settings=load(a.base_run/'settings.json') if a.base_run else settings
    assert original_settings['protocol_sha256']==sha(R/'data/annotations/live-demo-v12-protocol.json')
    settings['terrain_sha256']=original_settings['terrain_sha256']
    gates={**paired['gates'],'all181_source_and_mask_integrity':integrity['passed'] and integrity['unique_demo_frames']==181,'cpu_peak_below450_mib':profile['peak_working_set_mib']<=450,'runtime_does_not_import_training_stack':not profile['training_stack_imported'],'all_default_routes_checked':routes['all181_unique_frames_computed'] and routes['all_four_default_goals_identical'],'prediction_overlays_reviewed':review['prediction_overlays_reviewed'],'route_changes_reviewed':review['route_changes_reviewed'],'no_unresolved_route_regression':review['no_unresolved_route_regression'],'global_terrain_original':sha(R/'data/prepared/surface.onnx')==settings['terrain_sha256']}
    gates['source_overlay_semantics_acceptable']=review['source_overlay_semantics_acceptable']
    if a.base_run:
        assert settings['stage1_checkpoint_sha256']==sha(a.base_run/'best-navigation.pt')
        negative=load(run/'exclusion-onnx-conformance.json');negative_initial=load(a.base_run/'exclusion-onnx-conformance.json');assert negative['model_sha256']==digest and negative['constraints_sha256']==settings['constraints_sha256'];stage1=load(a.base_run/'candidate-onnx-conformance.json');selected=load(run/'candidate-onnx-conformance.json')
        gates.update(stage1_manual_regions_retained=not(set(stage1['manual_regions_found'])-set(selected['manual_regions_found'])),navcam_false_hardware_mean_at_most0_001=negative['navcam_hardware_fraction']<=.001,navcam_false_hardware_worst_frame_at_most0_01=negative['max_navcam_frame_hardware_fraction']<=.01,clear_flat_false_rock_at_most0_01=negative['clear_flat_rock_fraction']<=.01,clear_flat_false_rock_reduction_at_least80pct=negative['clear_flat_rock_fraction']<=.2*negative_initial['clear_flat_rock_fraction'])
    result={'population':paired['population'],'candidate_sha256':digest,'baseline_sha256':paired['baseline_sha256'],'selected_epoch':trained['best_epoch'],'epochs_completed':trained['epochs_completed'],'stage1_epochs':original_settings['epochs'] if a.base_run else None,'gates':gates,'all_gates_passed':all(gates.values()),'installed':False,'zero_reviewed_false_alarms_achieved':paired['zero_reviewed_false_alarms_achieved'],'reviewed_false_alarm_pixels_remaining':paired['reviewed_false_alarm_pixels_remaining'],'mean_per_frame_label_coverage':integrity['mean_per_frame_label_coverage'],'no_full_image_zero_error_or_generalization_claim':True,'manual_masks_not_used_at_runtime':True}
    dest=R/('data/research/live-demo-v13' if a.base_run else 'data/research/live-demo-v12');dest.mkdir(exist_ok=True,parents=True)
    for name in ['settings.json','history.json','candidate-report.json','paired-onnx-report.json','baseline-onnx-conformance.json','candidate-onnx-conformance.json','cpu-runtime.json','candidate-default-routes.json','route-comparison.json','release-visual-review.json']:
        shutil.copy2(run/name,dest/name)
    if a.base_run:
        shutil.copy2(run/'exclusion-onnx-conformance.json',dest/'exclusion-onnx-conformance.json');shutil.copy2(a.base_run/'exclusion-onnx-conformance.json',dest/'stage1-exclusion-onnx-conformance.json')
    if a.install:
        assert all(gates.values()),'Failed release gate: keep the original model'
        assert sha(R/'data/prepared/navigation.onnx')==paired['baseline_sha256'],'Installed model changed during evaluation'
        old=load(R/'data/prepared/navigation-model-card.json');dump(dest/'original-navigation-model-card.json',old)
        metrics=load(run/'candidate-onnx-conformance.json');card={'ready':True,'name':'SegFormer-B0 local navigation (v23, reviewed live-demo fit)','task':old['task'],'classes':old['classes'],'input_size':512,'model_sha256':digest,'inference_policy':'whole','navigation_obstacle_threshold':.7,'hardware_threshold':.8,'hardware_margin_native_px':3,'training':{'init':'Original v11 navigation checkpoint; separate navigation encoder and head only','frames':181,'hazcam_frames':142,'navcam_frames':39,'epochs_completed':trained['epochs_completed'],'selected_epoch':trained['best_epoch'],'selection':'Fixed protocol data/annotations/live-demo-v12-protocol.json','labels':'Reviewed partial interiors plus weighted stereo weak supervision; mean per-frame coverage19.70%; unknown255 ignored','population':'Every demo frame used for fitting and selection; no held-out test'},'demo_conformance':{k:v for k,v in metrics.items() if k not in ['frames','all_regions_found','manual_regions_found']},'generalization':'Not measured for this new candidate; demo fitting scores are not independent accuracy','peak_working_set_mib':profile['peak_working_set_mib'],'export_parity':{'frames':paired['parity_frames'],'argmax_agreement':paired['onnx_argmax_agreement'],'max_abs_logit_difference':paired['onnx_max_abs_logit_difference']},'how_used':old['how_used'],'limitations':'Partial project-reviewed labels and weak geometric supervision. Remaining reviewed false alarms are reported. No zero full-image error, exhaustive rock detection, certified traversability or flight qualification claim. No frame-ID or annotation lookup at inference.','license':old['license'],'previous_model_sha256':paired['baseline_sha256'],'results':'data/research/live-demo-v12/release-result.json'}
        if a.base_run:
            card['name']='SegFormer-B0 local navigation (v24, source-reviewed demo refinement)';card['training']['stage1_epochs_completed']=original_settings['epochs'];card['training']['selection']='data/annotations/live-demo-v12-protocol.json plus live-demo-v13-protocol.json';card['training']['labels']+='; source-reviewed class-exclusion labels also suppress false hardware without labelling unknown terrain free';card['class_exclusion_conformance']={k:v for k,v in negative.items() if k!='frames'};card['results']='data/research/live-demo-v13/release-result.json'
        shutil.copy2(run/'navigation.onnx',R/'data/prepared/navigation.onnx');dump(R/'data/prepared/navigation-model-card.json',card);assert sha(R/'data/prepared/navigation.onnx')==digest;result['installed']=True
    dump(dest/'release-result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
