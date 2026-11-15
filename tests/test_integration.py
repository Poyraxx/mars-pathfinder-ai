import json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.main import app,events,missions
from backend import data
from backend.perception import localize_obstacles


def test_unverified_camera_never_changes_cost_map():
    frame={"registration":{"verified":False}}
    result=localize_obstacles(frame,np.full((8,8),3),np.ones((8,8)),{})
    assert result["accepted"] is False
    assert "not been verified" in result["reason"]


def test_missing_registration_bound_still_rejects_even_if_flag_true():
    result=localize_obstacles({"registration":{"verified":True}},np.full((8,8),3),np.ones((8,8)),{})
    assert result["accepted"] is False


def _plane_with_block(error=0.02):
    n,e=np.meshgrid(np.arange(240)*.025,np.arange(240)*.025,indexing='ij')
    down=np.zeros_like(n)
    block=(np.abs(n-3)<.25)&(np.abs(e-3)<.25)
    down[block]=-.3
    xyz=np.stack([n,e,down],-1).astype(np.float32)
    xye=np.full(xyz.shape,error/np.sqrt(3),dtype=np.float32)
    return xyz,xye,block


def test_stereo_hazards_find_measured_step_only():
    from backend.perception import stereo_hazards
    xyz,xye,block=_plane_with_block()
    found=stereo_hazards(xyz,xye,.15)
    assert found['components']==1
    assert found['mask'][block].mean()>.9 and not found['mask'][~block].any()
    assert stereo_hazards(xyz,xye,.4)['mask'].sum()==0


def test_stereo_hazards_ignore_points_with_large_stereo_error():
    from backend.perception import stereo_hazards
    xyz,xye,_=_plane_with_block(error=.5)
    assert stereo_hazards(xyz,xye,.15)['mask'].sum()==0


def test_learned_big_rock_class_never_writes_obstacles():
    path=data.DATA_DIR/'frames.json'
    if not path.exists():pytest.skip('Real PDS catalogue not prepared')
    frame=next((f for f in json.loads(path.read_text('utf-8')) if f['registration'].get('verified')),None)
    if frame is None:pytest.skip('No registered frame')
    xyz=np.load(data.DATA_DIR/frame['xyz_path'])
    everything_big_rock=np.full(xyz.shape[:2],3)
    meta=json.loads((data.DATA_DIR/'gale'/'metadata.json').read_text('utf-8'))
    tall=localize_obstacles(frame,everything_big_rock,np.ones(xyz.shape[:2]),meta,step_m=1.0)
    assert tall['accepted'] is True and len(tall['points_rc'])==0
    assert tall['obstacle_source']=='stereo_geometry' and tall['model_big_rock_pixels_not_used']>0


def test_api_region_validation_and_actual_planner():
    with TestClient(app) as client:
        assert client.get('/healthz').status_code==200
        catalog=client.get('/api/catalog').json()
        assert len(catalog['layers'])==3
        assert client.get('/api/tiles/viking/50/0/0').status_code==404
        assert client.get('/api/tiles/viking/8/0/0').status_code==404
        assert client.get('/api/tiles/mola/8/0/0').status_code==404
        assert client.get('/api/tiles/themis/10/0/0').status_code==404
        assert client.get('/api/terrain/0/0/999').status_code==404
        if not catalog['regions']:pytest.skip('Run prepare_pds + assemble_data to install real region')
        region=catalog['regions'][0]
        bad=client.post('/api/missions',json={"start":{"lat":0,"lon":0},"goal":{"lat":0,"lon":1}})
        assert bad.status_code==422
        body={"start":region['default_start'],"goal":region['default_goal'],"region_id":region['id']}
        created=client.post('/api/missions',json=body)
        assert created.status_code==201
        mid=created.json()['id']
        from backend.main import route_result
        result=route_result(missions[mid],'terrain')
        assert result['status']=='ready'
        assert result['distance_m']>0
        assert result['max_slope_deg']<=20
        assert client.get(f'/api/missions/{mid}/report').json()['events'][0]['type']=='mission.created'
        stale=client.post(f'/api/missions/{mid}/plan',json={"expected_map_version":999})
        assert stale.status_code==409


def test_pds_corrected_affine_reproduces_localized_anchor():
    path=data.DATA_DIR/'frames.json'
    if not path.exists():pytest.skip('Real PDS catalogue not prepared')
    for frame in json.loads(path.read_text('utf-8')):
        p=frame['places']
        result=np.array(p['archive_site_to_global_ned_rotation'])@np.array(p['telemetry_rover_origin_in_site_m'])+np.array(p['archive_site_to_global_ned_translation_m'])
        assert np.allclose(result,p['corrected_rover_origin_global_ned_m'],atol=1e-6,rtol=0)


def test_verified_registration_requires_stored_passing_evidence():
    import hashlib
    path=data.DATA_DIR/'frames.json'
    if not path.exists():pytest.skip('Real PDS catalogue not prepared')
    for frame in json.loads(path.read_text('utf-8')):
        reg=frame['registration']
        if not reg.get('verified'):
            continue
        evidence_path=data.DATA_DIR/reg['evidence']
        assert evidence_path.exists()
        assert hashlib.sha256(evidence_path.read_bytes()).hexdigest()==reg['evidence_sha256']
        evidence=json.loads(evidence_path.read_text('utf-8'))
        assert evidence['verified'] is True and evidence['frame_id']==frame['id']
        rules=evidence['acceptance']
        assert len(evidence['trials'])>=4
        for trial in evidence['trials']:
            assert trial['empirical_p']<=rules['empirical_p_max']
            assert trial['best_shift_m']<=rules['best_shift_max_m']
            assert trial['null_n']>=1000
        assert 0<reg['horizontal_error_bound_m']<=2
        assert reg['horizontal_error_bound_m']>=max(t['best_shift_m'] for t in evidence['trials'])
        assert hashlib.sha256((data.DATA_DIR/frame['xyz_path']).read_bytes()).hexdigest()==evidence['inputs_sha256']['xyz_npy']


def test_compute_slot_is_reserved_before_background_task_starts(monkeypatch):
    import asyncio
    from fastapi import HTTPException
    from backend import main
    from backend.schemas import PlanRequest

    async def exercise():
        monkeypatch.setattr(main, 'compute_lock', asyncio.Lock())
        monkeypatch.setattr(main, 'route_result', lambda mission, mode: {'mode': mode})
        mid='compute-reservation-test'
        main.missions[mid]={'map_version':0, 'state':'created'}
        try:
            await main.plan(mid, PlanRequest(expected_map_version=0))
            with pytest.raises(HTTPException) as error:
                await main.plan(mid, PlanRequest(expected_map_version=0))
            assert error.value.status_code == 429
            await main.compute_lock.acquire()
            main.compute_lock.release()
            assert main.missions[mid]['state']=='ready'
        finally:
            main.missions.pop(mid,None)

    asyncio.run(exercise())
