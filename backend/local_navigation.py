"""Observed-space Hazcam planning. No screen-space curves or monocular depth.

Coordinates: local x=forward, y=right (metres), h=SITE-frame up. Semantic
segmentation is vetoed by stereo geometry.

Body model: a corridor as wide as Curiosity (2.8 m). The route centreline must stay
half a body width plus a margin away from every MEASURED obstacle cell, must itself run
over measured (or small-triangle-interpolated) ground, and at least half of the swept
corridor around it must be measured. Unmeasured cells are not assumed to be obstacles;
the measured share of the corridor is reported with every route.
The global/orbital map and its registration gate are not modified here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math

import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay, QhullError

from . import data
from .camera_model import site_points_to_pixels
from .navigation import astar
from .perception import infer


@dataclass(frozen=True)
class Settings:
    cell_m: float = .10
    forward_min_m: float = 0.
    body_front_m: float = 1.6    # Curiosity's front wheels; rover origin is mid-body
    forward_max_m: float = 8.
    half_width_m: float = 5.
    max_stereo_error_m: float = .10
    max_triangle_edge_m: float = .30
    max_step_m: float = .25      # wheel-hazard research threshold (Curiosity wheels are 0.5 m)
    rock_relief_m: float = .08
    max_slope_deg: float = 20.
    semantic_threshold: float = .70
    half_width_m_body: float = 1.4    # Curiosity is 2.8 m wide (wheel-to-wheel)
    clearance_m: float = .15
    min_corridor_observed: float = .5
    ribbon_width_m: float = .5


DEFAULT = Settings()


def frame_basis(frame):
    model = frame['camera_model']
    origin = np.asarray(model['rover_origin_in_site_m'], dtype=float)
    heading = np.asarray(model['rover_to_site_rotation'], dtype=float)[:2, 0]
    heading /= np.linalg.norm(heading)
    right = np.array([-heading[1], heading[0]])
    return origin, heading, right


def _cell(points, terrain):
    p = np.atleast_2d(points)
    return np.column_stack((np.floor((p[:, 0]-terrain['x0'])/terrain['cell_m']),
                            np.floor((p[:, 1]-terrain['y0'])/terrain['cell_m']))).astype(int)


def _centres(cells, terrain):
    return np.asarray(cells)*terrain['cell_m'] + [terrain['x0']+terrain['cell_m']/2, terrain['y0']+terrain['cell_m']/2]


def build_terrain(xyz, xye, segmentation, confidence, origin, forward, right, settings=DEFAULT):
    """Rasterize actual stereo and bounded interpolating triangles, never extrapolate."""
    cfg = settings
    shape = (int(round((cfg.forward_max_m-cfg.forward_min_m)/cfg.cell_m)),
             int(round(2*cfg.half_width_m/cfg.cell_m)))
    t = {'x0': cfg.forward_min_m, 'y0': -cfg.half_width_m, 'cell_m': cfg.cell_m,
         'settings': asdict(cfg)}
    delta = xyz[..., :2]-origin[:2]
    xy = np.stack((delta@forward, delta@right), axis=-1)
    error = np.linalg.norm(xye, axis=-1)
    rc = _cell(np.nan_to_num(xy).reshape(-1, 2), t).reshape((*xyz.shape[:2], 2))
    valid = np.isfinite(xyz).all(-1) & (xyz != 0).any(-1) & np.isfinite(error) & (error <= cfg.max_stereo_error_m)
    valid &= (rc[..., 0] >= 0) & (rc[..., 0] < shape[0]) & (rc[..., 1] >= 0) & (rc[..., 1] < shape[1])
    # Stereo points on the rover's own wheels and body (visible in the Hazcam's lower corners)
    # are hardware, not terrain: drop everything inside the body box.
    valid &= ~((xy[..., 0] < cfg.body_front_m) & (np.abs(xy[..., 1]) <= cfg.half_width_m_body+.2))
    ids = np.ravel_multi_index(rc[valid].T, shape)
    n = shape[0]*shape[1]
    low, high = np.full(n, np.inf), np.full(n, -np.inf)
    height = -xyz[..., 2][valid]
    np.minimum.at(low, ids, height); np.maximum.at(high, ids, height)
    count = np.bincount(ids, minlength=n)
    # Class order is checked by the caller. Rock predictions veto a cell; semantic
    # confidence is not an occupancy probability and never overrides measured relief.
    traversable = (segmentation[valid] != 3) & (confidence[valid] >= cfg.semantic_threshold)
    rocks = (segmentation[valid] == 3) & (confidence[valid] >= cfg.semantic_threshold)
    good_count = np.bincount(ids[traversable], minlength=n)
    rock_count = np.bincount(ids[rocks], minlength=n)
    sand_count = np.bincount(ids[(segmentation[valid] == 2) & traversable], minlength=n)
    observed = count > 0
    z = low.copy(); z[~observed] = np.nan
    sem = (good_count > 0) & (rock_count == 0)
    # Interpolate only small stereo-supported surface triangles. Their bounded
    # edges cannot bridge a boulder-sized hole or invent ground behind an occluder.
    measured_rc = np.column_stack(np.unravel_index(np.flatnonzero(observed), shape))
    positions = _centres(measured_rc, t)
    if len(positions) >= 3:
        try:
            tri = Delaunay(positions)
            target_rc = np.column_stack(np.unravel_index(np.arange(n), shape))
            targets = _centres(target_rc, t)
            simplex = tri.find_simplex(targets)
            safe_idx = np.maximum(simplex, 0)
            vertices = tri.simplices[safe_idx]
            p = positions[vertices]
            edges = np.max(np.linalg.norm(p-np.roll(p, 1, axis=1), axis=2), axis=1)
            measured_z = z[observed]
            zspan = np.ptp(measured_z[vertices], axis=1)
            supported = (simplex >= 0) & (edges <= cfg.max_triangle_edge_m) & (zspan <= cfg.max_step_m)
            bary = np.einsum('nij,nj->ni', tri.transform[safe_idx, :2], targets-tri.transform[safe_idx, 2])
            weights = np.column_stack((bary, 1-bary.sum(1)))
            interpolated = np.sum(measured_z[vertices]*weights, axis=1)
            holes = ~observed & supported
            z[holes] = interpolated[holes]
            sem[holes] = np.all(sem[observed][vertices[holes]], axis=1)
            observed |= holes
        except QhullError:
            pass
    z = z.reshape(shape); observed = observed.reshape(shape); sem = sem.reshape(shape)
    high = high.reshape(shape); low = low.reshape(shape)
    # Robust ground baseline fitted from measured lower surfaces. All heights
    # still come from stereo; the plane is used only to flag elevated obstacles.
    rr, cc = np.nonzero(observed)
    pts = _centres(np.column_stack((rr, cc)), t)
    a = np.column_stack((pts, np.ones(len(pts))))
    heights = z[rr, cc]
    keep = np.ones(len(heights), dtype=bool)
    coefficients = np.zeros(3)
    if len(heights) >= 12:
        for _ in range(5):
            if keep.sum() < 6:
                break
            coefficients = np.linalg.lstsq(a[keep], heights[keep], rcond=None)[0]
            residual = heights-a@coefficients
            keep = (residual >= np.percentile(residual, 5)-.05) & (residual <= np.percentile(residual, 65))
    baseline = np.full(shape, np.nan)
    baseline[rr, cc] = a@coefficients
    # The rover stands on the ground under its own front half: that area is physically
    # occupied by the rover, so it is usable for turning in place. Its height is the
    # fitted ground plane; it is never drawn (the body hides it from the camera).
    all_rc = np.column_stack(np.unravel_index(np.arange(shape[0]*shape[1]), shape))
    all_xy = _centres(all_rc, t)
    rover_cells = ((all_xy[:, 0] < cfg.body_front_m) & (np.abs(all_xy[:, 1]) <= cfg.half_width_m_body)).reshape(shape)
    plane = (np.column_stack((all_xy, np.ones(len(all_xy))))@coefficients).reshape(shape)
    z[rover_cells] = plane[rover_cells]; baseline[rover_cells] = plane[rover_cells]
    high[rover_cells] = low[rover_cells] = plane[rover_cells]
    observed = observed | rover_cells; sem = sem | rover_cells
    relief = z-baseline
    raised = observed & ((relief >= cfg.max_step_m) | (high-low >= cfg.max_step_m))
    # Slope from a 0.2 m smoothed surface: single-cell stereo noise is not a slope.
    zf = np.where(observed, z, 0.); w = ndimage.gaussian_filter(observed.astype(float), 2.)
    zs = np.where(w > .2, ndimage.gaussian_filter(zf, 2.)/np.maximum(w, 1e-6), np.nan)
    gy, gx = np.gradient(zs, cfg.cell_m)
    slope = np.nan_to_num(np.hypot(gx, gy))
    steep = observed & (np.degrees(np.arctan(slope)) > cfg.max_slope_deg)
    # Step edges between neighbouring measured cells.
    discontinuity = np.zeros(shape, bool)
    for axis in (0, 1):
        dz = np.abs(np.diff(z, axis=axis)); bad = np.isfinite(dz) & (dz >= cfg.max_step_m)
        s0, s1 = [slice(None)]*2, [slice(None)]*2
        s0[axis], s1[axis] = slice(None, -1), slice(1, None)
        discontinuity[tuple(s0)] |= bad; discontinuity[tuple(s1)] |= bad
    # The learned big-rock class blocks only where stereo confirms relief; otherwise it adds cost.
    rock_sem = rock_count.reshape(shape) > 0
    rock_confirmed = rock_sem & observed & (np.nan_to_num(relief) >= cfg.rock_relief_m)
    blocked = raised | discontinuity | steep | rock_confirmed
    passable = observed & ~blocked
    # Distance from every cell to the nearest MEASURED obstacle (blocked cell).
    obstacle = blocked
    clearance = ndimage.distance_transform_edt(~obstacle, sampling=cfg.cell_m) if obstacle.any() else np.full(shape, np.inf)
    radius = cfg.half_width_m_body+cfg.clearance_m+cfg.cell_m/math.sqrt(2)
    # Share of the swept corridor (disc of half a body width) that stereo actually measured.
    k = int(math.ceil(cfg.half_width_m_body/cfg.cell_m))
    yy, xx = np.mgrid[-k:k+1, -k:k+1]
    disc = ((yy**2+xx**2)*cfg.cell_m**2 <= cfg.half_width_m_body**2).astype(float)
    corridor_observed = ndimage.convolve(observed.astype(float), disc/disc.sum(), mode='constant', cval=0.)
    free = (passable & (clearance >= radius) & (corridor_observed >= cfg.min_corridor_observed)) | rover_cells
    cost = 1. + 2*np.minimum(slope, 1.)**2 + 3*(sand_count.reshape(shape) > 0) + 5*rock_sem + 2*(~sem)
    cost[~free] = np.inf
    return {**t, 'height': z, 'observed': observed, 'blocked': blocked, 'free': free, 'cost': cost,
            'clearance': clearance, 'corridor_observed': corridor_observed, 'rover_cells': rover_cells, 'support_pixels': int(valid.sum()),
            'semantic_rock_pixels': int(rocks.sum()), 'geometry_obstacle_cells': int((raised | discontinuity | steep).sum()), 'model_rock_cells': int(rock_sem.sum()), 'model_rock_confirmed_cells': int(rock_confirmed.sum())}


def segment_clear(a, b, terrain):
    """Dense supercover-style validation, including diagonal corner cells."""
    length = np.linalg.norm(np.asarray(b)-a)
    samples = np.linspace(a, b, max(2, int(math.ceil(length/(terrain['cell_m']*.2)))+1))
    cells = _cell(samples, terrain)
    shape = terrain['free'].shape
    if ((cells < 0).any() or (cells[:, 0] >= shape[0]).any() or (cells[:, 1] >= shape[1]).any()
            or not terrain['free'][tuple(cells.T)].all()):
        return False
    for p, q in zip(cells, cells[1:]):
        if (p != q).all() and not (terrain['free'][p[0], q[1]] and terrain['free'][q[0], p[1]]):
            return False
    return True


def plan_observed(terrain, target_xy):
    target = np.asarray(target_xy, float)
    angle = math.degrees(math.atan2(target[1], target[0]))
    if target[0] <= 0 or abs(angle) >= 100:
        return {'status': 'turn_required', 'reason': 'Target behind the observed front terrain; a safe U-turn cannot be verified from this view.',
                'required_turn_deg': round(angle, 1), 'path_xy': []}
    free = terrain['free']
    rc = np.column_stack(np.nonzero(free))
    if not len(rc):
        obs = terrain['observed']
        enough = bool(obs.any() and (terrain['corridor_observed'][obs] >= terrain['settings']['min_corridor_observed']).any())
        return {'status': 'no_path' if enough else 'insufficient_geometry',
                'reason': 'Measured obstacles leave no 2.8 m wide corridor.' if enough else 'Too little measured ground in front of the rover; no route is asserted.', 'path_xy': []}
    points = _centres(rc, terrain)
    # The route starts at the rover's own origin (mid-body) and may turn inside the
    # rover's current footprint before leaving it over measured, obstacle-clear ground.
    start = tuple(_cell(np.array([[terrain['cell_m']/2, 0.]]), terrain)[0])
    goal_rc = _cell(target, terrain)[0]
    inside = (goal_rc >= 0).all() and (goal_rc < free.shape).all()
    partial = not inside or not terrain['observed'][tuple(goal_rc)]
    if not partial:
        if not free[tuple(goal_rc)]:
            return {'status': 'no_path', 'reason': 'The requested local target or its clearance envelope is blocked.', 'path_xy': []}
        goal = tuple(goal_rc)
    else:
        direction = target/np.linalg.norm(target)
        progress = points@direction
        outside = ~terrain['rover_cells'][tuple(rc.T)]
        if not outside.any():
            return {'status': 'no_path', 'reason': 'Measured obstacles leave no 2.8 m wide corridor beyond the rover.', 'path_xy': []}
        progress = np.where(outside, progress, -np.inf)
        frontier = progress >= progress.max()-terrain['cell_m']*1.5
        candidates = np.flatnonzero(frontier)
        perpendicular = np.abs(points[:, 0]*direction[1]-points[:, 1]*direction[0])
        goal = tuple(rc[candidates[np.argmin(perpendicular[candidates])]])
    try:
        cells, _ = astar(terrain['cost'], start, goal, terrain['cell_m'])
    except (ValueError, TimeoutError):
        return {'status': 'no_path', 'reason': 'No collision-free connection through the observed terrain.', 'path_xy': []}
    path = _centres(cells, terrain)
    # Collision-checked string pulling, not a decorative spline. Check every
    # shortcut against the inflated grid, and recheck the final segments.
    simplified = [path[0]]; i = 0
    while i < len(path)-1:
        j = len(path)-1
        while j > i+1 and not segment_clear(path[i], path[j], terrain):
            j -= 1
        simplified.append(path[j]); i = j
    if not all(segment_clear(a, b, terrain) for a, b in zip(simplified, simplified[1:])):
        return {'status': 'no_path', 'reason': 'Final shortcut collision validation failed.', 'path_xy': []}
    # Round the corners (Chaikin, two passes, endpoints fixed) and keep the smoothed
    # line only if every segment passes the same collision check again.
    smooth = [np.asarray(q) for q in simplified]
    for _ in range(2):
        if len(smooth) < 3:
            break
        nxt = [smooth[0]]
        for q0, q1 in zip(smooth, smooth[1:]):
            nxt += [.75*q0+.25*q1, .25*q0+.75*q1]
        nxt.append(smooth[-1])
        smooth = nxt
    smoothed = len(smooth) > len(simplified) and all(segment_clear(a, b, terrain) for a, b in zip(smooth, smooth[1:]))
    if smoothed:
        simplified = smooth
    length = sum(float(np.linalg.norm(b-a)) for a, b in zip(simplified, simplified[1:]))
    if length < .2:
        return {'status': 'insufficient_geometry', 'reason': 'Measured clearance leaves no useful forward segment.', 'path_xy': []}
    dense = np.concatenate([np.linspace(a, b, max(2, int(np.linalg.norm(b-a)/terrain['cell_m'])+1)) for a, b in zip(simplified, simplified[1:])])
    dc = _cell(dense, terrain)
    corridor = float(np.mean(terrain['corridor_observed'][tuple(dc.T)]))
    min_clear = float(np.min(terrain['clearance'][tuple(dc.T)]))
    return {'status': 'partial' if partial else 'ready', 'path_xy': np.asarray(simplified).tolist(), 'smoothed': bool(smoothed),
            'corridor_observed_fraction': round(corridor, 3), 'min_obstacle_distance_m': round(min_clear, 2) if np.isfinite(min_clear) else None,
            'length_m': round(length, 3), 'reason': 'Only the stereo-observed local segment is validated; unseen space is blocked.' if partial else 'Observed local route and clearance validated.',
            'start_connection_observed': False}


def _height(xy, terrain):
    rc = np.column_stack(((xy[:, 0]-terrain['x0'])/terrain['cell_m']-.5,
                          (xy[:, 1]-terrain['y0'])/terrain['cell_m']-.5))
    return ndimage.map_coordinates(terrain['height'], rc.T, order=1, mode='constant', cval=np.nan)


def project_path(frame, terrain, path_xy):
    """Project a metric-width ground ribbon with the actual PDS CAHVORE model."""
    origin, forward, right = frame_basis(frame)
    path = np.asarray(path_xy, float)
    dense = []
    for a, b in zip(path, path[1:]):
        n = max(1, math.ceil(np.linalg.norm(b-a)/.04))
        dense.extend(a+(b-a)*u for u in np.arange(n)/n)
    if not dense:
        return {'segments': [], 'ribbons': []}
    dense = np.vstack((dense, path[-1]))
    tangent = np.gradient(dense, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-9)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    half = terrain['settings']['ribbon_width_m']/2
    body = terrain['settings']['half_width_m_body']
    points = np.concatenate((dense, dense-normal*half, dense+normal*half, dense-normal*body, dense+normal*body))
    heights = _height(points, terrain)
    site = np.column_stack((origin[:2]+points[:, 0, None]*forward+points[:, 1, None]*right, -heights))
    uv = site_points_to_pixels(site, frame['camera_model'])
    h, w = frame['camera_model']['native_shape']
    good = np.isfinite(uv).all(-1) & np.isfinite(heights) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    # A route point hidden by a closer measured rock/arm is not painted over it.
    xyz = np.load(data.DATA_DIR/frame['xyz_path'])
    xye = np.load(data.DATA_DIR/frame['xye_path'])
    pixel = np.nan_to_num(uv, nan=-1.)
    scale_y, scale_x = h/xyz.shape[0], w/xyz.shape[1]
    rc = np.array([(pixel[:, 1]-scale_y//2)/scale_y, (pixel[:, 0]-scale_x//2)/scale_x])
    seen = np.column_stack([ndimage.map_coordinates(xyz[..., k], rc, order=1, mode='constant', cval=np.nan) for k in range(3)])
    camera = np.asarray(frame['camera_model']['C'])@np.asarray(frame['camera_model']['rover_to_site_rotation']).T+origin
    valid_depth = np.isfinite(xyz).all(-1) & (xyz != 0).any(-1)
    support = ndimage.map_coordinates(valid_depth.astype(float), rc, order=1, mode='constant', cval=0.) >= 1-1e-6
    error = ndimage.map_coordinates(np.linalg.norm(xye, axis=-1), rc, order=1, mode='constant', cval=np.inf)
    # Hidden only where a CLOSER measured surface lies on the same ray (rock, arm, wheel);
    # pixels without stereo (e.g. shadow) keep the route drawn on the mapped ground height.
    measured = support & np.isfinite(seen).all(-1) & (error <= terrain['settings']['max_stereo_error_m'])
    closer = np.linalg.norm(seen-camera, axis=1) < np.linalg.norm(site-camera, axis=1)-np.maximum(.12, error*2+terrain['cell_m']*math.sqrt(2))
    good &= ~(measured & closer)
    # Draw nothing whose centreline point is still under the rover body (in-place turning).
    good &= np.tile(dense[:, 0] >= terrain['settings']['body_front_m']+terrain['cell_m'], 5)
    uv /= [w, h]
    size = len(dense)
    centre, left, right_edge, body_left, body_right = np.split(uv, 5)
    valid = good[:size] & good[size:2*size] & good[2*size:3*size]
    def runs(track, ok):
        out, cur = [], []
        for i in range(size):
            if ok[i] and (not cur or np.linalg.norm(track[i]-track[i-1]) < .08):
                cur.append(track[i].round(5).tolist())
            else:
                if len(cur) >= 8: out.append(cur)
                cur = [track[i].round(5).tolist()] if ok[i] else []
        if len(cur) >= 8: out.append(cur)
        return out
    # Corridor edges are drawn along straight-ish legs only; at a pivot the offset
    # boundary sweeps an arc that reads as a tangle in the image.
    heading = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    turning = np.zeros(size, bool)
    turning[1:] |= np.abs(np.diff(heading)) > np.radians(15)
    turning = ndimage.binary_dilation(turning, iterations=int(math.ceil(body/.04)))
    corridor = runs(body_left, good[3*size:4*size] & ~turning) + runs(body_right, good[4*size:] & ~turning)
    segments, ribbons, segment = [], [], []
    for i in range(size):
        if i and segment and np.linalg.norm(centre[i]-centre[i-1]) >= .08:
            if len(segment) >= 2:
                segments.append(segment)
            segment = []
        if valid[i]:
            segment.append(centre[i].round(6).tolist())
        elif segment:
            if len(segment) >= 2:
                segments.append(segment)
            segment = []
        if i and valid[i] and valid[i-1] and np.linalg.norm(centre[i]-centre[i-1]) < .08:
            ribbons.append(np.array([left[i-1], right_edge[i-1], right_edge[i], left[i]]).round(6).tolist())
    if len(segment) >= 2:
        segments.append(segment)
    return {'segments': segments, 'ribbons': ribbons, 'ribbon_width_m': half*2, 'corridor_edges': corridor, 'body_width_m': body*2,
            'method': 'Route on measured stereo ground heights, projected by the labelled PDS CAHVORE model; hidden where a closer measured surface is on the same ray'}


def project_obstacles(frame, terrain, limit=600):
    """Image positions of measured obstacle cells (top of the measured surface)."""
    cells = np.column_stack(np.nonzero(terrain['blocked'] & ~terrain['rover_cells']))
    if not len(cells):
        return []
    if len(cells) > limit:
        cells = cells[np.linspace(0, len(cells)-1, limit).astype(int)]
    origin, forward, right = frame_basis(frame)
    xy = _centres(cells, terrain)
    site = np.column_stack((origin[:2]+xy[:, 0, None]*forward+xy[:, 1, None]*right, -terrain['height'][tuple(cells.T)]))
    uv = site_points_to_pixels(site, frame['camera_model'])
    h, w = frame['camera_model']['native_shape']
    ok = np.isfinite(uv).all(-1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return (uv[ok]/[w, h]).round(4).tolist()


@lru_cache(maxsize=48)
def _terrain_cached(frame_id, fingerprint):
    frame, mask, confidence, _, stats = infer(frame_id)
    if data.model_status().get('classes') != ['soil', 'bedrock', 'sand', 'big_rock']:
        raise ValueError('Unsupported segmentation class order')
    xyz = np.load(data.DATA_DIR/frame['xyz_path']); xye = np.load(data.DATA_DIR/frame['xye_path'])
    origin, forward, right = frame_basis(frame)
    terrain = build_terrain(xyz, xye, mask, confidence, origin, forward, right)
    return terrain, stats


def compute(frame_id, goal=None):
    frame = next((f for f in data.frame_catalog() if f['id'] == frame_id), None)
    empty = {'frame_id': frame_id, 'segments': [], 'ribbons': [], 'is_ground_projection': True}
    if frame is None:
        return {**empty, 'status': 'unavailable', 'reason': 'Unknown source frame.'}
    if not frame.get('camera_model', {}).get('validation', {}).get('accepted') or not frame.get('xye_path'):
        return {**empty, 'status': 'insufficient_geometry', 'reason': 'Validated stereo/calibration is missing; no depth is invented.'}
    files = [data.DATA_DIR/frame[k] for k in ('xyz_path', 'xye_path', 'image_path')]
    files += [data.DATA_DIR/'surface.onnx', data.DATA_DIR/'model-card.json']
    key = hashlib.sha256(json.dumps([[(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files], frame['camera_model'], asdict(DEFAULT)], sort_keys=True).encode()).hexdigest()
    terrain, inference = _terrain_cached(frame_id, key)
    origin, forward, right = frame_basis(frame)
    if goal is None:
        target = np.array([DEFAULT.forward_max_m, 0.])
    else:
        r = 3396190.
        global_xy = np.radians([goal['lat'], goal['lon']])*r
        places = frame['places']
        delta = global_xy-np.asarray(places['corrected_rover_origin_global_ned_m'])[:2]
        rot = np.asarray(places['archive_site_to_global_ned_rotation'])
        site_delta = (np.r_[delta, 0.]@rot)[:2]
        target = np.array([site_delta@forward, site_delta@right])
    result = plan_observed(terrain, target)
    projection = project_path(frame, terrain, result['path_xy']) if result['path_xy'] else {'segments': [], 'ribbons': []}
    projection['obstacles'] = project_obstacles(frame, terrain)
    if result['status'] in ('ready', 'partial') and not projection['segments']:
        result['status'] = 'out_of_view'
        result['reason'] = 'Computed ground route is occluded or outside stereo-confirmed camera pixels.'
    return {**empty, **result, **projection, 'settings': asdict(DEFAULT),
            'geometry': {'support_pixels': terrain['support_pixels'], 'observed_cells': int(terrain['observed'].sum()),
                         'obstacle_cells': int(terrain['blocked'].sum()), 'footprint_safe_cells': int(terrain['free'].sum()),
                         'required_obstacle_distance_m': round(DEFAULT.half_width_m_body+DEFAULT.clearance_m+DEFAULT.cell_m/math.sqrt(2),3),
                         'body_width_m': 2*DEFAULT.half_width_m_body,
                         'triangle_interpolation_max_edge_m': DEFAULT.max_triangle_edge_m},
            'segmentation': inference,
            'limitations': 'Research local corridor only (stereo is reliable to about 4 m). Navcam-trained segmentation is not Hazcam-validated. Unmeasured cells are not assumed to be obstacles; the measured share of the corridor is reported. Global map is unchanged.'}
