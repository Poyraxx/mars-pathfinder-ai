"""Prove marker clipping leaves the actual cached planner/world memory unchanged."""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np
from backend import data,precompute
from backend.camera_visibility import visibility

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--before',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();sealed=precompute.load();assert sealed is not None,'Missing or stale deployment computation';before=json.loads(gzip.decompress(a.before.read_bytes()));after=sealed['local'];assert before.keys()==after.keys();catalog={f['id']:f for f in data.frame_catalog()};count_before=count_after=0;checked=0;model_hash=hashlib.sha256((data.DATA_DIR/'navigation.onnx').read_bytes()).hexdigest();card=json.loads((data.DATA_DIR/'navigation-model-card.json').read_text('utf8'));assert model_hash==card['model_sha256'];unique=set()
    for sid,b in before.items():
        c=after[sid];assert b['goal']==c['goal'] and b['frames'].keys()==c['frames'].keys()
        for fid,old in b['frames'].items():
            new=c['frames'][fid];unique.add(fid)
            for key in ['status','failure_kind','path_xy','segments','ribbons','tracking','geometry']:
                assert old.get(key)==new.get(key),(sid,fid,key)
            if new.get('segmentation'):assert new['segmentation']['navigation_model_sha256']==model_hash
            if not any(new.get(k) or old.get(k) for k in ['obstacles','tracked_obstacles']):continue
            excluded,_=visibility(catalog[fid]);assert excluded is not None
            for key in ['obstacles','tracked_obstacles']:
                count_before+=len(old.get(key,[]));count_after+=len(new.get(key,[]))
                for x,y in new.get(key,[]):
                    # Returned points are rounded to4decimals after clipping.
                    # Check the nearest native mask sample, allowing roundoff at
                    # the boundary by using the same normalized image cells.
                    r=min(excluded.shape[0]-1,int(y*excluded.shape[0]));col=min(excluded.shape[1]-1,int(x*excluded.shape[1]));assert not excluded[r,col],(sid,fid,key,x,y)
                checked+=1
    assert len(unique)==181 and count_after<count_before;assert not any(k in sys.modules for k in ['torch','transformers'])
    report={'passed':True,'actual_navigation_sha256':model_hash,'current_build_input_hash_verified':True,'build_input_sha256':sealed['input_sha256'],'unique_frames':len(unique),'scenarios':len(after),'planner_status_path_ribbon_geometry_and_world_tracking_unchanged':True,'marker_samples_before':count_before,'marker_samples_after':count_after,'hidden_on_self_or_unknown_pixels':count_before-count_after,'no_returned_marker_on_excluded_pixels':True,'marker_lists_checked':checked,'training_stack_imported':False,'scope':'Image-marker occlusion only; no ground clearance or unseen route fabricated'};a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
