"""Fresh CPU computation of every default demo route for paired regression QA."""
import argparse,gzip,hashlib,json,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.model:os.environ['MARS_NAVIGATION_MODEL_PATH']=str(a.model)
    from backend import data,main,replay
    from scripts.profile_navigation_runtime import peak_memory_bytes
    data.load_regions();rows=[];details={};start=time.perf_counter()
    for s in replay.scenarios():
        # Importing the API does not execute its lifespan or load build caches.
        result=main._scenario_local(s['id'],None);details[s['id']]=result;frames=[]
        for fid,r in result['frames'].items():frames.append({'frame_id':fid,'status':r['status'],'failure_kind':r.get('failure_kind'),'ribbon_segments':len(r.get('ribbons',[])),'path_points':len(r.get('path_xy',[])),'reason':r.get('reason')})
        rows.append({'scenario':s['id'],'goal':result['goal'],'frames':frames});print('SCENARIO',s['id'],len(frames),round(time.perf_counter()-start,1),flush=True)
    a.output.parent.mkdir(exist_ok=True,parents=True)
    a.output.with_suffix('.details.json.gz').write_bytes(gzip.compress(json.dumps(details,separators=(',',':')).encode(),mtime=0))
    assert not any(k in sys.modules for k in ['torch','transformers'])
    path=a.model or data.DATA_DIR/'navigation.onnx';report={'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'fresh_computation_not_build_cache':True,'scenarios':rows,'unique_frames':len({f['frame_id'] for s in rows for f in s['frames']}),'peak_working_set_mib':peak_memory_bytes()/1024**2,'elapsed_s':time.perf_counter()-start,'training_stack_imported':False};a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(report,indent=2),encoding='utf8');print('COMPLETE',json.dumps({k:v for k,v in report.items() if k!='scenarios'}),flush=True)
if __name__=='__main__':main()
