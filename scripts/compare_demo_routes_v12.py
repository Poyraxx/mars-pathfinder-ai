"""Compare fresh computations at identical default goals; flag changes for review."""
import argparse,json
from collections import Counter
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    b=json.loads(a.baseline.read_text('utf8'));c=json.loads(a.candidate.read_text('utf8'));assert b['fresh_computation_not_build_cache'] and c['fresh_computation_not_build_cache'];assert b['unique_frames']==c['unique_frames']==181
    before={s['scenario']:s for s in b['scenarios']};after={s['scenario']:s for s in c['scenarios']};assert before.keys()==after.keys();changes=[];rows=[]
    for sid,bs in before.items():
        cs=after[sid];assert bs['goal']==cs['goal'];bf={f['frame_id']:f for f in bs['frames']};cf={f['frame_id']:f for f in cs['frames']};assert bf.keys()==cf.keys()
        for fid,x in bf.items():
            y=cf[fid]
            if (x['status'],x['failure_kind'],x['ribbon_segments']>0)!=(y['status'],y['failure_kind'],y['ribbon_segments']>0):changes.append({'scenario':sid,'frame_id':fid,'before':x,'after':y})
        rows.append({'scenario':sid,'frames':len(bf),'baseline_status_counts':dict(Counter(f['status'] for f in bf.values())),'candidate_status_counts':dict(Counter(f['status'] for f in cf.values()))})
    report={'baseline_sha256':b['model_sha256'],'candidate_sha256':c['model_sha256'],'all181_unique_frames_computed':True,'all_four_default_goals_identical':True,'scenarios':rows,'changed_frame_scenario_entries':changes,'status_changes_need_review':len(changes),'baseline_peak_mib':b['peak_working_set_mib'],'candidate_peak_mib':c['peak_working_set_mib'],'note':'New rock detections can correctly remove unsafe routes. Inspect changes; route status alone is not obstacle ground truth.'};a.output.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
