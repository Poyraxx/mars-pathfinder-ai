"""Camera projection and navigation invariants, using actual archived fixtures."""
import numpy as np

from backend import data, replay
from backend.camera_model import project, site_points_to_pixels


def test_cahvore_perspective_limit_and_behind_camera_rejection():
    # Analytic camera fixture only, not claimed as a Mars observation.
    m=dict(type='CAHVORE',C=[0,0,0],A=[1,0,0],H=[0,100,0],V=[0,0,100],
           O=[1,0,0],R=[0,0,0],E=[0,0,0],T=1,P=1)
    uv=project(np.array([[1,0,0],[1,.2,.3],[-1,0,0]]),m)
    assert np.allclose(uv[:2],[[0,0],[20,30]],atol=1e-7)
    assert np.isnan(uv[2]).all()


def test_pds_camera_models_reproduce_original_stereo_pixel_indices():
    fs=[f for f in data.frame_catalog() if f.get('camera_model')]
    assert len(fs)>=4
    for f in fs:
        xyz=np.load(data.DATA_DIR/f['xyz_path'])
        rr,cc=np.nonzero(np.isfinite(xyz).all(-1)&(xyz!=0).any(-1))
        rr,cc=rr[::11],cc[::11]
        m=f['camera_model'];scale=f['native_shape'][0]//f['shape'][0]
        expected=np.column_stack((cc*scale+scale//2,rr*scale+scale//2))
        actual=site_points_to_pixels(xyz[rr,cc],m)
        error=np.linalg.norm(actual-expected,axis=1)
        assert np.isfinite(error).all()
        assert np.percentile(error,95)<.4


def test_remaining_path_does_not_return_to_departure():
    meta={'west_m':0,'north_m':0,'cell_m':1}
    cells=[(0,c) for c in range(21)]
    remaining=replay._remaining_cells(cells,np.array([15.2,-.5]),meta)
    assert remaining[0][1]>=15
    assert remaining[-1]==cells[-1]
    assert (0,0) not in remaining


def test_runtime_rejects_stale_registration_proof():
    from copy import deepcopy
    from backend.perception import localize_obstacles, registration_evidence_matches
    frame=deepcopy(next(f for f in data.frame_catalog() if f['registration'].get('verified')))
    assert registration_evidence_matches(frame)
    frame['registration']['evidence_sha256']='0'*64
    result=localize_obstacles(frame,np.zeros(frame['shape'],dtype=np.uint8),np.zeros(frame['shape']),{})
    assert not result['accepted'] and 'stale' in result['reason']


def test_forward_footage_has_calibrated_route_in_every_actual_frame():
    data.load_regions()
    result=replay.compute('hazcam-rocknest-approach')
    assert len(result['steps'])==4
    assert any(e['type']=='off_route' and e['deviation_m']>2 for e in result['events'])
    assert result['steps'][-1]['remaining_m']<result['steps'][0]['remaining_m']
    for step in result['steps']:
        p=step['camera_route']
        assert p['segments'] and p['visible_route_m']>0
        assert 'CAHVORE' in p['method']
        for segment in p['segments']:
            assert all(0<=q[0]<=1 and 0<=q[1]<=1 for q in segment)


def test_missed_destination_replans_at_last_frame_and_never_mirrors_route():
    data.load_regions()
    result=replay.compute('hazcam-return-waypoint')
    assert any(e['type']=='u_turn' for e in result['events'])
    last=result['steps'][-1]
    assert last['event']=='u_turn' and last['deviation_m']<1
    assert last['deviation_before_replan_m']>2
    assert abs(last['turn_to_route_deg'])>120
    assert not last['camera_route']['segments']


def test_extended_traverse_uses_many_distinct_real_acquisitions_and_poses():
    data.load_regions()
    result=replay.compute('hazcam-extended-traverse')
    steps=result['steps']
    assert len(steps)>=100 and len({s['frame_id'] for s in steps})==len(steps)
    assert [s['time'] for s in steps]==sorted(s['time'] for s in steps)
    frames={f['id']:f for f in data.frame_catalog()}
    positions=np.array([frames[s['frame_id']]['places']['corrected_rover_origin_global_ned_m'][:2] for s in steps])
    assert np.count_nonzero(np.linalg.norm(np.diff(positions,axis=0),axis=1)>.25)>=30
    assert np.linalg.norm(positions[-1]-positions[0])>100
    assert all(frames[s['frame_id']]['camera_model']['validation']['accepted'] for s in steps)
    assert all(s['route'] for s in steps)
    assert any(s['camera_route']['segments'] for s in steps)
    assert steps[-1]['remaining_m']<steps[0]['remaining_m']
