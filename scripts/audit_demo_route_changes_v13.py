"""Explain paired default-route changes using retained original API geometry."""
import argparse,gzip,json
from pathlib import Path
from collections import Counter

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline-api',type=Path,required=True);ap.add_argument('--baseline-snapshot',type=Path,required=True);ap.add_argument('--candidate-details',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    old=json.loads(gzip.decompress(a.baseline_api.read_bytes()));details=json.loads(gzip.decompress(a.candidate_details.read_bytes()));current=details['hazcam-extended-traverse'];baseline=json.loads(a.baseline_snapshot.read_text('utf8'));snapshot=next(s for s in baseline['scenarios'] if s['scenario']=='hazcam-extended-traverse');assert old['goal']==current['goal']==snapshot['goal'];assert {r['frame_id']:r['status'] for r in snapshot['frames']}=={fid:r['status'] for fid,r in old['frames'].items()}
    rows=[]
    for fid,b in old['frames'].items():
        c=current['frames'][fid]
        if b['status']==c['status']:continue
        rows.append({'frame_id':fid,'before':b['status'],'after':c['status'],'reason':c.get('reason'),'old_geometry':b.get('geometry'),'new_geometry':c.get('geometry'),'old_turn_deg':b.get('required_turn_deg'),'new_turn_deg':c.get('required_turn_deg')})
    result={'original_api_matches_fresh_original_snapshot':True,'same_default_goal':True,'transitions':dict(Counter(r['before']+' -> '+r['after'] for r in rows)),'changes':rows,'note':'Status changes reflect updated learned obstacle/self masks under unchanged stereo clearance and target bearing rules. A safe U-turn is not assumed when measured swept-body clearance is absent. Orange remembered marks that occlude onto hardware are a separate image-overlay defect, fixed without dropping the world-space track.'};a.output.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result['transitions']),flush=True)
if __name__=='__main__':main()
