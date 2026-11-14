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
    elif kind == 'closed':
        rock = abs(f-8.) < .25
    h[rock] = .7
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


def test_unknown_ground_and_unobserved_turn_are_not_traversable():
    terrain = terrain_fixture()
    terrain['free'][:] = False
    terrain['observed'][:] = False
    result = local.plan_observed(terrain, [13., 0.])
    assert result['status'] == 'insufficient_geometry' and not result['path_xy']
    result = local.plan_observed(terrain, [-13., 0.])
    assert result['status'] == 'turn_required' and not result['path_xy']


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


def test_real_hazcam_rock_ahead_is_passed_on_a_measured_side_with_body_clearance():
    # Curiosity Front Hazcam, sol 46, facing the 'Jake Matijevic' rock.
    frame = _frame('FLA_401492677')
    result = local.compute(frame['id'])
    assert result['status'] in ('ready', 'partial') and result['ribbons'] and result['obstacles']
    terrain = _terrain_for(frame)
    points = np.array(result['path_xy'])
    assert all(local.segment_clear(a, b, terrain) for a, b in zip(points, points[1:]))
    assert result['min_obstacle_distance_m'] >= result['geometry']['required_obstacle_distance_m']-local.DEFAULT.cell_m
    assert np.abs(points[:, 1]).max() > 1.  # it steers around the rock instead of through it
    quads = np.array(result['ribbons'])
    assert (quads >= 0).all() and (quads <= 1).all()


def test_real_hazcam_open_sand_keeps_a_straight_route():
    result = local.compute(_frame('FLA_404772660')['id'])
    points = np.array(result['path_xy'])
    assert result['status'] in ('ready', 'partial') and len(points) >= 2
    assert np.abs(points[:, 1]).max() < .3 and result['ribbons']


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
