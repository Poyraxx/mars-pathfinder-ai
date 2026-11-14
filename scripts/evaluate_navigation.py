"""Measured A/B/C comparison on real DEM; unknown obstacle truth remains null."""
import json
import time
import numpy as np
from backend import data
from backend.navigation import cost_grid,plan_legs,metrics


def main():
    data.load_regions();region=data.regions['gale'];rng=np.random.default_rng(42)
    valid=np.argwhere(region.analysis['valid'] & (region.analysis['slope']<15))
    trials=[]
    for i in range(10):
        start,goal=[tuple(p) for p in valid[rng.choice(len(valid),2,replace=False)]]
        outcomes=[]
        for label,mode in [('A','distance'),('B','terrain'),('C','surface')]:
            tick=time.perf_counter()
            try:
                path,cost=plan_legs(cost_grid(region.analysis,mode,20),[start,goal],region.meta['cell_m'])
                result={"system":label,"status":"ready","cost":cost,**metrics(path,region.elevation,region.analysis['slope'],region.meta['cell_m'],.5)}
            except (ValueError,TimeoutError) as exc:result={"system":label,"status":str(exc)}
            result['planning_ms']=(time.perf_counter()-tick)*1000
            result['independent_obstacle_collisions']=None
            result['coverage_fraction']=float(region.analysis['valid'].mean())
            outcomes.append(result)
        trials.append({'trial':i+1,'start_cell':list(map(int,start)),'goal_cell':list(map(int,goal)),'outcomes':outcomes})
    report={'source':region.meta['source_id'],'seed':42,'trials':trials,
            'surface_updates_accepted':0,'note':'C equals B until independently validated surface updates exist. No improvement or collision-reduction claim is made. Independent obstacle ground truth is unavailable.'}
    out=data.DATA_DIR/'navigation-evaluation.json';out.write_text(json.dumps(report,indent=2),'utf-8')
    print(json.dumps({'trials':len(trials),'path_cases':sum(r['status']=='ready' for t in trials for r in t['outcomes']),'surface_updates_accepted':0,'output':str(out)},indent=2))


if __name__=='__main__':main()
