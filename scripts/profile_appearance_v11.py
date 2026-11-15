"""Separate-process CPU budget: global model, three local routes and auxiliary."""
import argparse,json,os,sys,hashlib,time
from pathlib import Path
_dll=os.add_dll_directory(os.environ['MARS_DLL_DIR']) if os.name=='nt' and os.environ.get('MARS_DLL_DIR') else None
from profile_navigation_runtime import peak_memory_bytes
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    from backend import data,local_navigation,perception
    from backend.surface_inference import predict_outputs,navigation_masks
    from appearance_guard_v11 import guard_obstacles
    import onnxruntime as ort
    from PIL import Image
    data.load_regions();options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;options.enable_cpu_mem_arena=False
    aux=ort.InferenceSession(str(a.candidate),sess_options=options,providers=['CPUExecutionProvider']);rows=[]
    for prefix in ['FLA_401492677','FLA_404772660','FLA_402284021']:
        frame=next(f for f in data.frame_catalog() if f['id'].startswith(prefix));start=time.perf_counter();route=local_navigation.compute(frame['id'])
        # Resolve through the same real native input used by the runtime.
        im,_=perception.inference_source(frame);p=predict_outputs(perception.get_navigation_session(),im,'whole')[0][0];r=predict_outputs(aux,im,'whole')[0][0];old,_=navigation_masks(p,im.size,.7,.8,3);guard,_=guard_obstacles(old,p,r)
        rows.append({'frame_id':frame['id'],'local_route_status':route['status'],'aux_elapsed_total_s':time.perf_counter()-start,'mask_subset':not (guard&~old).any().item()})
    perception.infer(rows[-1]['frame_id']);assert not any(k in sys.modules for k in ['torch','transformers'])
    peak=peak_memory_bytes()/1024**2;report={'candidate_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),'peak_working_set_mib':peak,'within450mib':peak<=450,'provider':'CPUExecutionProvider','platform':sys.platform,'aux_cpu_arena':False,'training_stack_imported':False,'routes':rows,'limitation':'One Windows process and three real routes plus global classifier; host allocator and warmed server caches may differ. No Render memory guarantee.'}
    a.output.write_text(json.dumps(report,indent=2),'utf8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
