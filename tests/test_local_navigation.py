"""Controlled synthetic 3D/semantic fixtures; not presented as NASA observations."""
from dataclasses import replace
import math

import numpy as np
import pytest

from backend import local_navigation as local


def terrain_fixture(kind='flat'):
    cfg = replace(local.DEFAULT, forward_max_m=16., half_width_m=7.)
    f, r = np.meshgrid(np.arange(.425, 15.98, .04), np.arange(-6.975, 7., .04), indexing='ij')
    h = np.zeros(f.shape)
    rock = np.zeros(f.shape, bool)
    if kind == 'rock':
        rock = (f-8.)**2+r**2 < .55**2
        # A second obstacle closes the left passage; the right side remains open.
        rock |= (f-8.)**2+(r+4.)**2 < .7**2
    elif kind == 'low_block':
        rock=(f-8.)**2+r**2<.55**2
    elif kind == 'pebble':
        rock=(f-8.025)**2+(r-.025)**2<.035**2
    elif kind == 'reference_stone':
        rock=(f-8.025)**2+(r-.025)**2<.06**2
    elif kind == 'closed':
        rock = abs(f-8.) < .25
    h[rock] = .7
    if kind=='low_block':h[rock]=.18
    if kind=='pebble':h[rock]=.008
    if kind=='reference_stone':h[rock]=.06
    xyz = np.stack((f, r, -h), axis=-1)
    xye = np.full(xyz.shape, .001)
    classes = np.where(rock, 3, 1).astype(np.uint8)
    confidence = np.full(f.shape, .99)
    return local.build_terrain(xyz, xye, classes, confidence, np.zeros(3), np.array([1., 0.]), np.array([0., 1.]), cfg)


def assert_validated(result, terrain):
    assert result['status'] == 'ready', result
    points = np.array(result['path_xy'])
    assert all(local.segment_clear(a, b, terrain) for a, b in zip(points, points[1:]))
    return points


def test_open_flat_ground_produces_direct_route_without_cosmetic_curves():
    terrain = terrain_fixture()
    result = local.plan_observed(terrain, [13., 0.])
    points = assert_validated(result, terrain)
    assert len(points) == 2  # collision-checked direct shortcut
    assert abs(points[:, 1]).max() < .11
    assert result['length_m'] == pytest.approx(np.linalg.norm(points[-1]-points[0]), abs=.002)


def test_large_rock_uses_the_open_right_side_and_preserves_clearance_after_simplification():
    terrain = terrain_fixture('rock')
    result = local.plan_observed(terrain, [13., 0.])
    points = assert_validated(result, terrain)
    assert len(points) >= 3 and points[:, 1].max() > 2.
    dense = np.concatenate([np.linspace(a, b, 100) for a, b in zip(points, points[1:])])
    assert np.linalg.norm(dense-[8., 0.], axis=1).min() > local.DEFAULT.half_width_m_body+.55


def test_closed_passage_returns_no_path_and_never_falls_back_to_a_curve():
    terrain = terrain_fixture('closed')
    result = local.plan_observed(terrain, [13., 0.])
    assert result['status'] == 'no_path' and result['path_xy'] == []


def test_harmless_pebble_is_not_inflated_into_a_rover_sized_obstacle():
    terrain=terrain_fixture('pebble');result=local.plan_observed(terrain,[13.,0.])
    points=assert_validated(result,terrain)
    assert not terrain['blocked'].any() and terrain['model_rock_confirmed_cells']==0
    assert terrain['semantic_rock_pixels']>0
    assert len(points)==2 and np.abs(points[:,1]).max()<.11


def test_broad_raised_block_is_avoided_even_below_the_quarter_metre_step_limit():
    terrain=terrain_fixture('low_block');result=local.plan_observed(terrain,[13.,0.])
    points=assert_validated(result,terrain)
    assert terrain['model_rock_confirmed_cells']>0 and np.abs(points[:,1]).max()>1.


def test_sensitive_profile_includes_reference_sized_raised_stone():
    terrain=terrain_fixture('reference_stone');result=local.plan_observed(terrain,[13.,0.])
    points=assert_validated(result,terrain)
    assert terrain['model_rock_confirmed_cells']>0 and np.abs(points[:,1]).max()>1.


def test_self_hardware_does_not_become_an_obstacle_or_interpolated_free_ground():
    cfg=replace(local.DEFAULT,forward_max_m=16.,half_width_m=7.)
    f,r=np.meshgrid(np.arange(.425,15.98,.04),np.arange(-6.975,7.,.04),indexing='ij')
    equipment=(f-5.)**2+r*r<.12**2
    xyz=np.stack((f,r,np.where(equipment,-1.,0.)),axis=-1)
    classes=np.where(equipment,255,1).astype('uint8')
    terrain=local.build_terrain(xyz,np.full(xyz.shape,.001),classes,np.full(f.shape,.99),np.zeros(3),np.array([1.,0.]),np.array([0.,1.]),cfg)
    rc=tuple(local._cell([[5.,0.]],terrain)[0])
    assert terrain['nonterrain_pixels_excluded']>0
    assert not terrain['blocked'][rc] and not terrain['observed'][rc] and not terrain['free'][rc]


def test_rock_appearance_on_a_gentle_ground_crest_needs_local_relief():
    cfg=replace(local.DEFAULT,forward_max_m=16.,half_width_m=7.)
    f,r=np.meshgrid(np.arange(.425,15.98,.04),np.arange(-6.975,7.,.04),indexing='ij')
    height=.10*np.cos((f-8.)/2.)
    texture=(f-8.)**2+r*r<.3**2
    xyz=np.stack((f,r,-height),axis=-1)
    terrain=local.build_terrain(xyz,np.full(xyz.shape,.001),np.where(texture,3,1).astype('uint8'),np.full(f.shape,.99),np.zeros(3),np.array([1.,0.]),np.array([0.,1.]),cfg)
    assert terrain['semantic_rock_pixels']>0 and terrain['model_rock_confirmed_cells']==0
    result=local.plan_observed(terrain,[13.,0.])
    points=assert_validated(result,terrain)
    assert len(points)==2 and np.abs(points[:,1]).max()<.11


def test_unknown_ground_and_unobserved_turn_are_not_traversable():
    terrain = terrain_fixture()
    terrain['free'][:] = False
    terrain['observed'][:] = False
    result = local.plan_observed(terrain, [13., 0.])
    assert result['status'] == 'insufficient_geometry' and not result['path_xy']
    result = local.plan_observed(terrain, [-13., 0.])
    assert result['status'] == 'turn_required' and not result['path_xy'] and result['turn_direction'] in ('left', 'right')


def test_destination_behind_plans_a_swing_on_measured_free_ground():
    terrain = terrain_fixture()
    result = local.plan_observed(terrain, [-10., 4.])
    assert result['status'] == 'u_turn' and result['turn_direction'] == 'right'
    points = np.array(result['path_xy'])
    assert points[-1, 1] > 1. and all(local.segment_clear(a, b, terrain) for a, b in zip(points, points[1:]))


def test_combined_diagonal_slope_is_checked_not_only_each_axis():
    cfg=replace(local.DEFAULT,forward_max_m=16.,half_width_m=7.)
    f,r=np.meshgrid(np.arange(.425,15.98,.04),np.arange(-6.975,7.,.04),indexing='ij')
    # Each axis is16.7 degrees; the actual combined slope is23 degrees.
    xyz=np.stack((f,r,-.3*(f+r)),axis=-1)
    terrain=local.build_terrain(xyz,np.full(xyz.shape,.001),np.ones(f.shape,dtype=np.uint8),np.full(f.shape,.99),np.zeros(3),np.array([1.,0.]),np.array([0.,1.]),cfg)
    result=local.plan_observed(terrain,[13.,0.])
    assert result['status']=='no_path' and not result['path_xy']


def test_shortcut_cannot_cut_a_diagonal_obstacle_corner():
    terrain = {'x0': 0., 'y0': 0., 'cell_m': 1., 'free': np.ones((3, 3), bool)}
    terrain['free'][0, 1] = False
    assert not local.segment_clear([.5, .5], [1.5, 1.5], terrain)


def preview_fixture(raised=True,error=.18):
    f,r=np.meshgrid(np.arange(.425,11.98,.05),np.arange(-4.975,5.,.05),indexing='ij')
    rock=(f-6.5)**2+r*r<.6**2
    height=np.where(rock & raised,.65,0.)
    xyz=np.stack((f,r,-height),axis=-1)
    xye=np.full(xyz.shape,.001);xye[f>4.5]=[error,.01,.015]
    segmentation=np.where(rock,3,1).astype('uint8')
    return local.build_terrain(xyz,xye,segmentation,np.full(f.shape,.99),np.zeros(3),np.array([1.,0.]),np.array([0.,1.]))


def test_distant_stereo_rock_biases_heading_without_extending_known_ground():
    terrain=preview_fixture();result=local.plan_observed(terrain,[20.,0.])
    assert len(terrain['distant_rock_candidates'])==1
    assert result['status']=='partial', result
    points=np.asarray(result['path_xy'])
    assert abs(points[-1,1])>.9  # start avoiding before reaching the distant rock
    assert points[:,0].max()<4.6  # uncertain far XYZ never became drivable
    assert all(local.segment_clear(a,b,terrain) for a,b in zip(points,points[1:]))


def test_semantic_rock_on_flat_ground_and_excessive_stereo_error_cannot_create_preview():
    assert not preview_fixture(raised=False)['distant_rock_candidates']
    assert not preview_fixture(error=.8)['distant_rock_candidates']


def calibrated_plane(tmp_path, monkeypatch):
    from backend import data
    # Analytic pinhole limit of CAHVORE, with real perspective and an independently
    # constructed stereo depth image for these tests only.
    model = dict(type='CAHVORE', C=[0., 0., -1.], A=[1., 0., 0.], H=[512., 400., 0.], V=[512., 0., 400.],
                 O=[1., 0., 0.], R=[0., 0., 0.], E=[0., 0., 0.], T=1., P=1.,
                 rover_to_site_rotation=np.eye(3).tolist(), rover_origin_in_site_m=[0., 0., 0.], native_shape=[1024, 1024])
    yy, xx = np.meshgrid(np.arange(256)*4+2, np.arange(256)*4+2, indexing='ij')
    valid = yy > 512
    f = np.divide(400., yy-512, out=np.zeros(yy.shape), where=valid)
    r = (xx-512)*f/400.
    xyz = np.stack((f, r, np.zeros(f.shape)), axis=-1)
    xyz[~valid] = 0.
    np.save(tmp_path/'xyz.npy', xyz)
    np.save(tmp_path/'xye.npy', np.full(xyz.shape, .001))
    monkeypatch.setattr(data, 'DATA_DIR', tmp_path)
    return {'camera_model': model, 'xyz_path': 'xyz.npy', 'xye_path': 'xye.npy'}


def test_metric_ground_ribbon_obeys_camera_perspective(tmp_path, monkeypatch):
    frame = calibrated_plane(tmp_path, monkeypatch)
    projection = local.project_path(frame, terrain_fixture(), [[3.1, 0.], [12.5, 0.]])
    assert len(projection['ribbons']) > 100
    quads = np.array(projection['ribbons'])
    widths = np.linalg.norm(quads[:, 1]-quads[:, 0], axis=1)
    assert widths[0] > widths[-1]*3
    assert quads[0, :, 1].mean() > quads[-1, :, 1].mean()
    assert (quads >= 0).all() and (quads <= 1).all()
    expected_near = local.DEFAULT.ribbon_width_m*400/3.1/1024
    assert widths[0] == pytest.approx(expected_near, rel=.03)


def test_closer_stereo_surface_occludes_ribbon_instead_of_painting_over_it(tmp_path, monkeypatch):
    frame = calibrated_plane(tmp_path, monkeypatch)
    terrain = terrain_fixture()
    clear = local.project_path(frame, terrain, [[3.1, 0.], [12.5, 0.]])
    xyz = np.load(tmp_path/'xyz.npy')
    # Measured test occluder on the same image rays, halfway to the floor.
    camera = np.array([0., 0., -1.])
    xyz[144:152, 123:133] = camera+.5*(xyz[144:152, 123:133]-camera)
    np.save(tmp_path/'xyz.npy', xyz)
    occluded = local.project_path(frame, terrain, [[3.1, 0.], [12.5, 0.]])
    assert len(occluded['ribbons']) < len(clear['ribbons'])
    assert len(occluded['segments']) >= 2


def test_missing_calibration_does_not_generate_monocular_depth(monkeypatch):
    monkeypatch.setattr(local.data, 'frame_catalog', lambda: [{'id': 'no-stereo'}])
    result = local.compute('no-stereo')
    assert result['status'] == 'insufficient_geometry'
    assert result['segments'] == result['ribbons'] == []


def _frame(prefix):
    from backend import data
    return next(f for f in data.frame_catalog() if f['id'].startswith(prefix))


def _terrain_for(frame):
    from backend import data
    _, mask, conf, _, _ = local.infer(frame['id'])
    xyz = np.load(data.DATA_DIR/frame['xyz_path']); xye = np.load(data.DATA_DIR/frame['xye_path'])
    origin, forward, right = local.frame_basis(frame)
    return local.build_terrain(xyz, xye, mask, conf, origin, forward, right)


def test_real_hazcam_boulder_view_requires_body_clearance_or_reports_no_route():
    # Catalogue sol 45, facing Jake Matijevic. This demo is a training scene,
    # so this checks planner geometry, not held-out model generalisation.
    frame = _frame('FLA_401492677')
    result = local.compute(frame['id'])
    terrain = _terrain_for(frame)
    assert result['obstacles']
    if result['status']=='no_path':
        # The user-selected sensitive profile also blocks the smaller measured
        # rocks beside Jake. Prove the rover's free component cannot exit its
        # footprint, instead of weakening clearance to recover the old demo.
        from scipy import ndimage
        labels,_=ndimage.label(terrain['free'])
        start=tuple(local._cell([[.05,0.]],terrain)[0])
        assert not ((labels==labels[start])&~terrain['rover_cells']).any()
        assert not result['ribbons'] and not result['segments']
        return
    assert result['status'] in ('ready', 'partial') and result['ribbons']
    points = np.array(result['path_xy'])
    assert all(local.segment_clear(a, b, terrain) for a, b in zip(points, points[1:]))
    assert result['min_obstacle_distance_m'] >= result['geometry']['required_obstacle_distance_m']-local.DEFAULT.cell_m
    assert np.abs(points[:, 1]).max() > 1.  # it steers around the rock instead of through it
    quads = np.array(result['ribbons'])
    assert (quads >= 0).all() and (quads <= 1).all()


def test_real_hazcam_open_sand_route_keeps_body_clearance_from_the_measured_rock_cluster():
    """Open sand with a stereo-confirmed rock cluster ahead-left (~4 m). The route stays a simple,
    nearly straight lane but shifts right just enough for the 2.8 m body clearance; it never
    bends toward the rock, and scoop marks/rover hardware are not obstacles."""
    result = local.compute(_frame('FLA_404772660')['id'])
    points = np.array(result['path_xy'])
    assert result['status'] in ('ready', 'partial') and len(points) >= 2 and result['ribbons']
    assert result['min_obstacle_distance_m'] >= result['geometry']['required_obstacle_distance_m']
    assert points[:, 1].min() > -.3 and np.abs(points[:, 1]).max() < 1.2
    assert len(points) <= 4


def test_real_hazcam_blocked_view_reports_no_route_instead_of_a_curve():
    result = local.compute('FLA_402284021XYZ_F0043232FHAZ00202M1')
    assert result['status'] in ('no_path', 'insufficient_geometry')
    assert result['segments'] == result['ribbons'] == [] and result['reason']


def test_local_api_validates_goal_and_does_not_change_global_map():
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend import data
    before = (data.DATA_DIR/'gale/elevation.npy').read_bytes()
    with TestClient(app) as client:
        path='/api/frames/FLA_402284021XYZ_F0043232FHAZ00202M1/local-navigation'
        response=client.get(path)
        assert response.status_code == 200 and response.json()['ribbons'] == []
        assert client.get(path+'?goal_lat=nan&goal_lon=0').status_code == 422
        assert client.get(path+'?goal_lat=0').status_code == 422
        assert client.get('/api/frames/not-a-source/local-navigation').status_code == 404
    assert (data.DATA_DIR/'gale/elevation.npy').read_bytes() == before


def test_remembered_rock_fills_only_unmeasured_cells_and_survives_a_missed_view():
    """A stereo-confirmed obstacle from an earlier frame is carried in global coordinates. It is
    added only where the current frame has no measurement, and it reaches the planner's clearance."""
    from backend import data, replay
    data.load_regions()
    steps = replay.compute('hazcam-extended-traverse')['steps']
    frames = local.compute_sequence(steps[:30], None)
    used = [f for f in frames.values() if f.get('tracking', {}).get('remembered_cells_used', 0) > 0]
    assert used, 'memory never used on the Rocknest approach'
    assert all('_confirmed_global' not in f for f in frames.values())
    single = local.compute(steps[23]['frame_id'])
    assert 'tracked_obstacles' in single and single['tracking']['remembered_cells_used'] == 0


def test_memory_never_overrides_a_cell_measured_in_the_current_frame():
    frame = _frame('FLA_404772660')
    terrain = {'blocked': np.zeros((4, 4), bool), 'rover_cells': np.zeros((4, 4), bool),
               'observed': np.array([[True]*4, [True]*4, [False]*4, [False]*4]), 'corridor_observed': np.ones((4, 4)),
               'cost': np.ones((4, 4)), 'height': np.zeros((4, 4)), 'x0': 0., 'y0': -.2, 'cell_m': .1}
    origin, forward, right = local.frame_basis(frame)
    cells = local._centres(np.array([[0, 1], [3, 1]]), terrain)
    site_xy = origin[:2]+cells[:, 0, None]*forward+cells[:, 1, None]*right
    remembered = local._site_to_global(frame, site_xy, np.zeros(2))
    updated, added = local.with_memory(frame, terrain, remembered)
    assert added.tolist() == [[3, 1]] and updated['blocked'][3, 1] and not updated['blocked'][0, 1]
