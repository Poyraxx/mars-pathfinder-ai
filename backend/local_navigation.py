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
from .perception import infer_navigation as infer


@dataclass(frozen=True)
class Settings:
    cell_m: float = .10
    forward_min_m: float = 0.
    body_front_m: float = 1.6
    forward_max_m: float = 8.
    half_width_m: float = 5.
    max_stereo_error_m: float = .10
    max_triangle_edge_m: float = .30
    max_step_m: float = .25
    rock_relief_m: float = .02
    min_rock_width_m: float = .08
    rock_ground_ring_m: float = .8
    min_rock_ground_cells: int = 12
    max_slope_deg: float = 20.
    semantic_threshold: float = .70
    half_width_m_body: float = 1.4
    clearance_m: float = .15
    min_corridor_observed: float = .5
    ribbon_width_m: float = .5
    preview_min_forward_m: float = 4.
    preview_range_m: float = 12.
    preview_max_stereo_error_m: float = .5


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


def distant_rock_candidates(xyz, xye, segmentation, confidence, xy, ground_plane, ground_uncertainty, cfg):
    """Occupied-only lookahead from actual XYZ, never monocular depth/free space.

    Less precise distant stereo cannot extend the drivable surface. A candidate
    needs semantic AND raised-surface evidence above the near-ground plane,
    after subtracting vertical uncertainty. It is only a steering cost. A slope
    change can still produce a false candidate; no safety claim is made here.
    """
    error=np.linalg.norm(xye,axis=-1)
    height=-xyz[...,2]
    baseline=xy@ground_plane[:2]+ground_plane[2]
    valid=np.isfinite(xyz).all(-1)&(xyz!=0).any(-1)&np.isfinite(xye).all(-1)
    valid &= (error>cfg.max_stereo_error_m)&(error<=cfg.preview_max_stereo_error_m)
    valid &= (xy[...,0]>=cfg.preview_min_forward_m)&(xy[...,0]<=cfg.preview_range_m)&(np.abs(xy[...,1])<=cfg.half_width_m+2.)
    valid &= (segmentation==3)&(confidence>=cfg.semantic_threshold)
    valid &= height-baseline >= cfg.rock_relief_m+2*np.abs(xye[...,2])+ground_uncertainty
    labels,count=ndimage.label(valid)
    candidates=[]
    for cid in range(1,count+1):
        pixels=labels==cid
        if pixels.sum()<3:continue
        points=xy[pixels];centre=np.median(points,axis=0)
        extent=float(np.percentile(np.linalg.norm(points-centre,axis=1),95))
        bound=float(error[pixels].max())
        width=float(np.ptp(points,axis=0).max())
        if width+2*bound<cfg.min_rock_width_m:continue
        candidates.append({'centre_xy_m':centre.round(3).tolist(),'footprint_radius_m':round(max(.10,extent),3),
                           'kind':'rock_candidate' if extent<=2. else 'large_raised_region',
                           'range_error_bound_m':round(bound,3),'support_pixels':int(pixels.sum()),
                           'source':'measured XYZ + semantic rock + uncertainty-gated relief',
                           'purpose':'steering cost only; does not establish free ground'})
    return sorted(candidates,key=lambda e:np.linalg.norm(e['centre_xy_m']))[:64]


def lookahead_risk(points, candidates, cfg):
    """Penalty for continuing the candidate heading into a distant rock envelope."""
    points=np.atleast_2d(points);length=np.linalg.norm(points,axis=1)
    direction=points/np.maximum(length[:,None],1e-6)
    risk=np.zeros(len(points))
    for candidate in candidates:
        centre=np.asarray(candidate['centre_xy_m']);along=direction@centre
        distance=np.linalg.norm(centre-direction*along[:,None],axis=1)
        envelope=cfg.half_width_m_body+cfg.clearance_m+candidate['footprint_radius_m']+candidate['range_error_bound_m']
        penalty=np.clip(1-distance/envelope,0,1)
        penalty[(along<=length)|(length<cfg.body_front_m)]=0
        risk=np.maximum(risk,penalty)
    return risk


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
    valid &= ~((xy[..., 0] < cfg.body_front_m) & (np.abs(xy[..., 1]) <= cfg.half_width_m_body))
    self_pixels=valid&(segmentation==255)
    self_ids=np.ravel_multi_index(rc[self_pixels].T,shape)
    self_cells=np.zeros(shape[0]*shape[1],bool);self_cells[self_ids]=True
    valid &= segmentation!=255
    ids = np.ravel_multi_index(rc[valid].T, shape)
    n = shape[0]*shape[1]
    low, high = np.full(n, np.inf), np.full(n, -np.inf)
    height = -xyz[..., 2][valid]
    np.minimum.at(low, ids, height); np.maximum.at(high, ids, height)
    height_error=np.zeros(n)
    np.maximum.at(height_error,ids,np.abs(xye[...,2][valid]))
    count = np.bincount(ids, minlength=n)
    traversable = (segmentation[valid] != 3) & (confidence[valid] >= cfg.semantic_threshold)
    rocks = (segmentation[valid] == 3) & (confidence[valid] >= cfg.semantic_threshold)
    good_count = np.bincount(ids[traversable], minlength=n)
    rock_count = np.bincount(ids[rocks], minlength=n)
    sand_count = np.bincount(ids[(segmentation[valid] == 2) & traversable], minlength=n)
    observed = count > 0
    z = low.copy(); z[~observed] = np.nan
    sem = (good_count > 0) & (rock_count == 0)
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
            holes = ~observed & supported & ~self_cells
            z[holes] = interpolated[holes]
            sem[holes] = np.all(sem[observed][vertices[holes]], axis=1)
            observed |= holes
        except QhullError:
            pass
    z = z.reshape(shape); observed = observed.reshape(shape); sem = sem.reshape(shape)
    high = high.reshape(shape); low = low.reshape(shape)
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
    all_rc = np.column_stack(np.unravel_index(np.arange(shape[0]*shape[1]), shape))
    all_xy = _centres(all_rc, t)
    rover_cells = ((all_xy[:, 0] < cfg.body_front_m) & (np.abs(all_xy[:, 1]) <= cfg.half_width_m_body)).reshape(shape)
    plane = (np.column_stack((all_xy, np.ones(len(all_xy))))@coefficients).reshape(shape)
    z[rover_cells] = plane[rover_cells]; baseline[rover_cells] = plane[rover_cells]
    high[rover_cells] = low[rover_cells] = plane[rover_cells]
    observed = observed | rover_cells; sem = sem | rover_cells
    relief = z-baseline
    raised = observed & ((relief >= cfg.max_step_m) | (high-low >= cfg.max_step_m))
    zf = np.where(observed, z, 0.); w = ndimage.gaussian_filter(observed.astype(float), 2.)
    zs = np.where(w > .2, ndimage.gaussian_filter(zf, 2.)/np.maximum(w, 1e-6), np.nan)
    gy, gx = np.gradient(zs, cfg.cell_m)
    slope = np.nan_to_num(np.hypot(gx, gy))
    steep = observed & (np.degrees(np.arctan(slope)) > cfg.max_slope_deg)
    discontinuity = np.zeros(shape, bool)
    for axis in (0, 1):
        dz = np.abs(np.diff(z, axis=axis)); bad = np.isfinite(dz) & (dz >= cfg.max_step_m)
        s0, s1 = [slice(None)]*2, [slice(None)]*2
        s0[axis], s1[axis] = slice(None, -1), slice(1, None)
        discontinuity[tuple(s0)] |= bad; discontinuity[tuple(s1)] |= bad
    rock_sem = rock_count.reshape(shape) > 0
    components,component_count=ndimage.label(rock_sem & observed)
    rock_confirmed=np.zeros(shape,bool)
    point_components=components[tuple(rc[valid].T)]
    measured_ground=observed & np.isfinite(low) & ~rock_sem & ~rover_cells & sem
    for cid in range(1,component_count+1):
        component=components==cid
        ring=ndimage.binary_dilation(component,iterations=max(1,int(math.ceil(cfg.rock_ground_ring_m/cfg.cell_m))))&~component&measured_ground
        ground_rc=np.column_stack(np.nonzero(ring))
        if len(ground_rc)<cfg.min_rock_ground_cells:continue
        ground_xy=_centres(ground_rc,t)
        design=np.column_stack((ground_xy,np.ones(len(ground_xy))))
        ground_height=low[tuple(ground_rc.T)]
        if np.linalg.matrix_rank(design)<3:continue
        local_plane=np.linalg.lstsq(design,ground_height,rcond=None)[0]
        fit_error=float(np.percentile(np.abs(ground_height-design@local_plane),90))
        candidate_points=rocks&(point_components==cid)
        point_relief=height-(xy[valid]@local_plane[:2]+local_plane[2])-np.abs(xye[...,2][valid])-fit_error
        elevated_points=candidate_points&(point_relief>=cfg.rock_relief_m)
        points=xy[valid][elevated_points]
        width=float(np.ptp(points,axis=0).max()) if len(points)>=2 else 0.
        if width+1e-6>=cfg.min_rock_width_m:
            rock_confirmed[tuple(rc[valid][elevated_points].T)]=True
    blocked = raised | discontinuity | steep | rock_confirmed
    passable = observed & ~blocked
    obstacle = blocked
    clearance = ndimage.distance_transform_edt(~obstacle, sampling=cfg.cell_m) if obstacle.any() else np.full(shape, np.inf)
    radius = cfg.half_width_m_body+cfg.clearance_m+cfg.cell_m/math.sqrt(2)
    k = int(math.ceil(cfg.half_width_m_body/cfg.cell_m))
    yy, xx = np.mgrid[-k:k+1, -k:k+1]
    disc = ((yy**2+xx**2)*cfg.cell_m**2 <= cfg.half_width_m_body**2).astype(float)
    corridor_observed = ndimage.convolve(observed.astype(float), disc/disc.sum(), mode='constant', cval=0.)
    free = ((passable & (corridor_observed >= cfg.min_corridor_observed)) | rover_cells) & (clearance >= radius)
    ground_uncertainty=max(.05,float(np.percentile(np.abs(heights[keep]-a[keep]@coefficients),90))) if keep.sum()>=6 else np.inf
    preview=distant_rock_candidates(xyz,xye,segmentation,confidence,xy,coefficients,ground_uncertainty,cfg)
    preview_risk=lookahead_risk(all_xy,preview,cfg).reshape(shape)
    cost = 1. + 2*np.minimum(slope, 1.)**2 + 3*(sand_count.reshape(shape) > 0) + 2*(~sem) + 6*preview_risk
    cost[~free] = np.inf
    return {**t, 'height': z, 'observed': observed, 'blocked': blocked, 'free': free, 'cost': cost,
            'physical_obstacle': raised | rock_confirmed,
            'clearance': clearance, 'corridor_observed': corridor_observed, 'rover_cells': rover_cells, 'support_pixels': int(valid.sum()),
            'measured_high':high,'height_error':height_error.reshape(shape),
            'distant_rock_candidates':preview, 'lookahead_risk':preview_risk,
            'nonterrain_pixels_excluded':int(self_pixels.sum()),
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



def plan_observed(terrain, target_xy, *, waypoint=False, preferred_turn=None, previous_path=None):
    target = np.asarray(target_xy, float)
    angle = math.degrees(math.atan2(target[1], target[0]))
    if target[0] <= 0 or abs(angle) >= 100:
        natural = 'right' if angle > 0 else 'left'
        word=preferred_turn if preferred_turn in ('left','right') and abs(angle)>=135 else natural
        choices=[word,'left' if word=='right' else 'right']
        for choice in choices:
            side=1. if choice=='right' else -1.
            directed=angle if choice==natural else angle+side*360
            for tx, ty in ((2.8, 2.8), (3.4, 3.6), (2.4, 2.0), (3.8, 1.6)):
                swing = plan_observed(terrain, [tx, side*ty])
                if swing['status'] in ('ready', 'partial') and swing.get('path_xy'):
                    return {**swing, 'status': 'u_turn', 'turn_direction': choice, 'required_turn_deg': round(directed, 1),
                            'reason': f'Destination is {abs(directed):.0f}° to the {choice}, behind the camera. Swing {choice} through measured free ground, then keep turning toward it.'}
        side=1. if word=='right' else -1.
        directed=angle if word==natural else angle+side*360
        return {'status': 'turn_required', 'turn_direction': word, 'required_turn_deg': round(directed, 1), 'path_xy': [],
                'reason': f'Destination is {abs(directed):.0f}° to the {word}. No measured room to swing out in front, so turn in place {word}; the ground behind is outside this camera view.'}
    free = terrain['free']
    rc = np.column_stack(np.nonzero(free))
    if not len(rc):
        obs = terrain['observed']
        enough = bool(obs.any() and (terrain['corridor_observed'][obs] >= terrain['settings']['min_corridor_observed']).any())
        return {'status': 'no_path' if enough else 'insufficient_geometry',
                'reason': 'Measured obstacles leave no 2.8 m wide corridor.' if enough else 'Too little measured ground in front of the rover; no route is asserted.', 'path_xy': []}
    start = tuple(_cell(np.array([[terrain['cell_m']/2, 0.]]), terrain)[0])
    labels, _ = ndimage.label(free)
    if labels[start] == 0:
        conflict=bool(terrain['blocked'][start] or terrain['clearance'][start]<terrain['settings']['half_width_m_body']+terrain['settings']['clearance_m']+terrain['cell_m']/math.sqrt(2))
        return {'status': 'no_path', 'failure_kind': 'clearance_conflict' if conflict else 'unobserved_connection',
                'reason': ('Nearby measured rocks overlap the conservative rover-clearance envelope at the current pose; no forward route is verified.'
                           if conflict else 'A connection from the rover footprint to measured ground is not established.'), 'path_xy': []}
    rc = np.column_stack(np.nonzero(labels == labels[start]))
    points = _centres(rc, terrain)
    goal_rc = _cell(target, terrain)[0]
    inside = (goal_rc >= 0).all() and (goal_rc < free.shape).all()
    reachable = inside and labels[tuple(goal_rc)] == labels[start]
    measured_target = inside and terrain['observed'][tuple(goal_rc)]
    if measured_target and not reachable and not waypoint:
        return {'status': 'no_path', 'reason': 'The requested local target or its clearance envelope is blocked.', 'path_xy': []}
    partial = not reachable or not measured_target
    if not partial:
        goal = tuple(goal_rc)
    else:
        direction = target/np.linalg.norm(target)
        progress = points@direction
        outside = ~terrain['rover_cells'][tuple(rc.T)]
        if not outside.any():
            return {'status': 'no_path', 'reason': 'Measured obstacles leave no 2.8 m wide corridor beyond the rover.', 'path_xy': []}
        progress = np.where(outside, progress, -np.inf)
        has_preview=bool(terrain.get('distant_rock_candidates'))
        frontier = progress >= progress.max()-(.75 if has_preview else terrain['cell_m']*1.5)
        candidates = np.flatnonzero(frontier)
        perpendicular = np.abs(points[:, 0]*direction[1]-points[:, 1]*direction[0])
        score=perpendicular[candidates]
        if has_preview:
            risk=terrain['lookahead_risk'][tuple(rc[candidates].T)]
            score=score+8*risk+.5*(progress.max()-progress[candidates])
        goal = tuple(rc[candidates[np.argmin(score)]])
    try:
        cells, _ = astar(terrain['cost'], start, goal, terrain['cell_m'])
    except (ValueError, TimeoutError):
        return {'status': 'no_path', 'reason': 'No collision-free connection through the observed terrain.', 'path_xy': []}
    path = _centres(cells, terrain)
    simplified = [path[0]]; i = 0
    while i < len(path)-1:
        j = len(path)-1
        while j > i+1 and not segment_clear(path[i], path[j], terrain):
            j -= 1
        simplified.append(path[j]); i = j
    if not all(segment_clear(a, b, terrain) for a, b in zip(simplified, simplified[1:])):
        return {'status': 'no_path', 'reason': 'Final shortcut collision validation failed.', 'path_xy': []}
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
    retained=False
    if previous_path is not None:
        reference=np.asarray(previous_path,float)
        if reference.ndim==2 and reference.shape[1]==2 and len(reference)>=2 and np.isfinite(reference).all():
            same_ends=np.allclose(reference[0],simplified[0],atol=1e-6) and np.allclose(reference[-1],simplified[-1],atol=1e-6)
            reference_length=float(np.linalg.norm(np.diff(reference,axis=0),axis=1).sum())
            if same_ends and reference_length<=length*1.1+.1 and all(segment_clear(a,b,terrain) for a,b in zip(reference,reference[1:])):
                simplified=reference;length=reference_length;retained=True;smoothed=False
    if length < .2:
        return {'status': 'insufficient_geometry', 'reason': 'Measured clearance leaves no useful forward segment.', 'path_xy': []}
    dense = np.concatenate([np.linspace(a, b, max(2, int(np.linalg.norm(b-a)/terrain['cell_m'])+1)) for a, b in zip(simplified, simplified[1:])])
    dc = _cell(dense, terrain)
    corridor = float(np.mean(terrain['corridor_observed'][tuple(dc.T)]))
    min_clear = float(np.min(terrain['clearance'][tuple(dc.T)]))
    return {'status': 'partial' if partial else 'ready', 'path_xy': np.asarray(simplified).tolist(), 'smoothed': bool(smoothed), 'continuity_reused':retained,
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
    measured = support & np.isfinite(seen).all(-1) & (error <= terrain['settings']['max_stereo_error_m'])
    closer = np.linalg.norm(seen-camera, axis=1) < np.linalg.norm(site-camera, axis=1)-np.maximum(.12, error*2+terrain['cell_m']*math.sqrt(2))
    good &= ~(measured & closer)
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
    heading = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    turning = np.zeros(size, bool)
    turning[1:] |= np.abs(np.diff(heading)) > np.radians(15)
    turning = ndimage.binary_dilation(turning, iterations=int(math.ceil(body/.04)))
    corridor = runs(body_left, good[3*size:4*size] & ~turning) + runs(body_right, good[4*size:] & ~turning)
    segments, ribbons, ribbon_s, segment = [], [], [], []
    along = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
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
            ribbon_s.append([round(float(along[i-1]), 3), round(float(along[i]), 3)])
    if len(segment) >= 2:
        segments.append(segment)
    return {'segments': segments, 'ribbons': ribbons, 'ribbon_s_m': ribbon_s, 'ribbon_width_m': half*2, 'corridor_edges': corridor, 'body_width_m': body*2,
            'method': 'Route on measured stereo ground heights, projected by the labelled PDS CAHVORE model; hidden where a closer measured surface is on the same ray'}


def project_obstacles(frame, terrain, limit=600, exclude=None):
    """Image positions of measured obstacle cells (top of the measured surface)."""
    mask = terrain['blocked'] & ~terrain['rover_cells']
    if exclude is not None and len(exclude):
        mask = mask.copy(); mask[tuple(np.asarray(exclude).T)] = False
    return project_cells(frame, terrain, np.column_stack(np.nonzero(mask)), limit)


def project_cells(frame, terrain, cells, limit=600):
    cells = np.asarray(cells)
    if not len(cells):
        return []
    if len(cells) > limit:
        cells = cells[np.linspace(0, len(cells)-1, limit).astype(int)]
    origin, forward, right = frame_basis(frame)
    xy = _centres(cells, terrain)
    z = terrain['height'][tuple(cells.T)]
    z = np.where(np.isfinite(z), z, np.nanmedian(terrain['height']))
    site = np.column_stack((origin[:2]+xy[:, 0, None]*forward+xy[:, 1, None]*right, -z))
    uv = site_points_to_pixels(site, frame['camera_model'])
    h, w = frame['camera_model']['native_shape']
    ok = np.isfinite(uv).all(-1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    # A remembered ground rock can project behind the moving arm. Keep its
    # world-space track, but never paint its marker on visible self hardware.
    from .camera_visibility import visibility
    excluded, _ = visibility(frame)
    if excluded is not None:
        inside = np.flatnonzero(ok)
        pixels = np.floor(uv[inside] / [w, h] * [excluded.shape[1], excluded.shape[0]]).astype(int)
        ok[inside] &= ~excluded[pixels[:, 1], pixels[:, 0]]
    return (uv[ok]/[w, h]).round(4).tolist()


def project_candidates(frame, terrain):
    """Image rings around distant (4-12 m) stereo-supported rock candidates; display + steering cost only."""
    out = []
    origin, forward, right = frame_basis(frame)
    h, w = frame['camera_model']['native_shape']
    base = float(np.nanmedian(terrain['height'])) if np.isfinite(terrain['height']).any() else 0.
    for c in terrain.get('distant_rock_candidates', []):
        cx, cy = c['centre_xy_m']; r = c['footprint_radius_m']
        ang = np.linspace(0, 2*np.pi, 17)
        ring = np.column_stack((cx+r*np.cos(ang), cy+r*np.sin(ang)))
        site = np.column_stack((origin[:2]+ring[:, 0, None]*forward+ring[:, 1, None]*right, np.full(len(ring), -base)))
        uv = site_points_to_pixels(site, frame['camera_model'])
        ok = np.isfinite(uv).all(-1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if ok.sum() >= 8:
            out.append({'ring': (uv[ok]/[w, h]).round(4).tolist(), 'range_m': round(float(np.hypot(cx, cy)), 1), 'kind': c['kind']})
    return out


@lru_cache(maxsize=160)
def _terrain_cached(frame_id, fingerprint):
    frame, mask, confidence, _, stats = infer(frame_id)
    if data.model_status().get('classes') != ['soil', 'bedrock', 'sand', 'big_rock']:
        raise ValueError('Unsupported segmentation class order')
    xyz = np.load(data.DATA_DIR/frame['xyz_path']); xye = np.load(data.DATA_DIR/frame['xye_path'])
    origin, forward, right = frame_basis(frame)
    terrain = build_terrain(xyz, xye, mask, confidence, origin, forward, right)
    return terrain, stats




MEMORY_RANGE_M = 10.
MEMORY_FRAMES = 12


def _site_to_global(frame, site_xy, heights):
    p = frame['places']
    site = np.column_stack((site_xy, -heights))
    return site @ np.asarray(p['archive_site_to_global_ned_rotation']).T + np.asarray(p['archive_site_to_global_ned_translation_m'])


def _global_to_site(frame, ned):
    p = frame['places']
    return (np.asarray(ned)-np.asarray(p['archive_site_to_global_ned_translation_m'])) @ np.asarray(p['archive_site_to_global_ned_rotation'])


def confirmed_obstacles_global(frame, terrain):
    """Stereo-confirmed obstacle cells of this frame in global NED (rover hardware excluded), 20 cm deduplicated."""
    cells = np.column_stack(np.nonzero(terrain.get('physical_obstacle', terrain['blocked']) & ~terrain['rover_cells']))
    if not len(cells):
        return np.zeros((0, 3))
    origin, forward, right = frame_basis(frame)
    xy = _centres(cells, terrain)
    site_xy = origin[:2]+xy[:, 0, None]*forward+xy[:, 1, None]*right
    z = terrain.get('measured_high',terrain['height'])[tuple(cells.T)]
    z = np.where(np.isfinite(z), z, np.nanmedian(terrain['height']))
    ned = _site_to_global(frame, site_xy, z)
    key = np.round(ned[:, :2]/.2).astype(np.int64)
    _, keep = np.unique(key, axis=0, return_index=True)
    return ned[np.sort(keep)]


def _remeasured_rocks(terrain, candidates, expected_height, cfg):
    """Retain a missed semantic detection only if current stereo remeasures its
    raised surface. A current flat-ground measurement always clears the memory.
    Historical appearance or matching image coordinates alone are insufficient.
    """
    keep=np.zeros_like(candidates)
    if 'measured_high' not in terrain or 'height_error' not in terrain:return keep
    size=max(1,int(math.ceil(cfg.rock_ground_ring_m/terrain['cell_m'])))
    support=terrain['observed']&~terrain['rover_cells']&~terrain['blocked']&~candidates
    shape=candidates.shape
    for row,col in np.column_stack(np.nonzero(candidates&terrain['observed'])):
        top=terrain['measured_high'][row,col];error=terrain['height_error'][row,col]
        if not np.isfinite(top) or abs(top-expected_height[row,col])>.05+2*error:continue
        r0,r1=max(0,row-size),min(shape[0],row+size+1)
        c0,c1=max(0,col-size),min(shape[1],col+size+1)
        points=np.column_stack(np.nonzero(support[r0:r1,c0:c1]))+[r0,c0]
        if len(points)<cfg.min_rock_ground_cells:continue
        xy=_centres(points,terrain);design=np.column_stack((xy,np.ones(len(xy))))
        ground=terrain['height'][tuple(points.T)]
        finite=np.isfinite(ground);design,ground=design[finite],ground[finite]
        if len(ground)<cfg.min_rock_ground_cells or np.linalg.matrix_rank(design)<3:continue
        plane=np.linalg.lstsq(design,ground,rcond=None)[0]
        residual=float(np.percentile(np.abs(ground-design@plane),90))
        here=np.r_[_centres([[row,col]],terrain)[0],1.]
        keep[row,col]=top-here@plane-error-residual>=cfg.rock_relief_m
    return keep


def with_memory(frame, terrain, remembered, cfg=DEFAULT):
    """Add obstacles confirmed in earlier frames (global NED) to this frame's grid.

    A rock that stereo confirmed a moment ago does not vanish because the next view misses it
    (shadow, occlusion, out of stereo range). Only physically measured obstacle cells are carried, only within
    MEMORY_RANGE_M of the current rover; the planner's clearance and free space are rebuilt."""
    none = np.zeros((0, 2), int)
    if remembered is None or not len(remembered):
        return terrain, none
    rover = np.asarray(frame['places']['corrected_rover_origin_global_ned_m'])[:2]
    near = remembered[np.linalg.norm(remembered[:, :2]-rover, axis=1) <= MEMORY_RANGE_M]
    if not len(near):
        return terrain, none
    origin, forward, right = frame_basis(frame)
    site = _global_to_site(frame, near)
    delta = site[:, :2]-origin[:2]
    local = np.column_stack((delta@forward, delta@right))
    rc = _cell(local, terrain)
    shape = terrain['blocked'].shape
    inside = (rc[:, 0] >= 0) & (rc[:, 0] < shape[0]) & (rc[:, 1] >= 0) & (rc[:, 1] < shape[1])
    if not inside.any():
        return terrain, none
    added = np.zeros(shape, bool); added[tuple(rc[inside].T)] = True
    cells_h = np.full(shape, np.nan); cells_h[tuple(rc[inside].T)] = -site[inside][:, 2]
    added &= ~terrain['blocked'] & ~terrain['rover_cells']
    remeasured=_remeasured_rocks(terrain,added,cells_h,cfg)
    added &= ~terrain['observed'] | remeasured
    if not added.any():
        return terrain, none
    height = terrain['height'].copy()
    fill = added & ~np.isfinite(height); height[fill] = cells_h[fill]
    blocked = terrain['blocked'] | added
    clearance = ndimage.distance_transform_edt(~blocked, sampling=cfg.cell_m)
    radius = cfg.half_width_m_body+cfg.clearance_m+cfg.cell_m/math.sqrt(2)
    free = ((terrain['observed'] & ~blocked & (terrain['corridor_observed'] >= cfg.min_corridor_observed)) | terrain['rover_cells']) & (clearance >= radius)
    cost = terrain['cost'].copy(); cost[~free] = np.inf
    physical=terrain.get('physical_obstacle',terrain['blocked'])|remeasured
    return {**terrain, 'blocked': blocked, 'physical_obstacle':physical,'memory_reconfirmed_cells':int(remeasured.sum()),
            'clearance': clearance, 'free': free, 'cost': cost, 'height': height}, np.column_stack(np.nonzero(added))


def compute_sequence(steps, goal):
    """Local routes for a replay in order, carrying stereo-confirmed obstacles between frames."""
    frames, memory = {}, []
    preferred_turn=None;last_step=None;last_result=None
    for index,st in enumerate(steps):
        fid = st['frame_id']
        if fid in frames:
            continue
        memory=[(at,points) for at,points in memory if index-at<=MEMORY_FRAMES]
        remembered = np.vstack([points for _,points in memory]) if memory else None
        try:
            hints={}
            if preferred_turn is not None:hints['preferred_turn']=preferred_turn
            pose=st.get('rover');previous=last_step.get('rover') if last_step else None
            if pose and previous and last_result and last_result.get('status') in ('ready','partial'):
                movement=np.linalg.norm(np.radians([pose['lat']-previous['lat'],pose['lon']-previous['lon']])*3396190.)
                turn=abs((pose['yaw_deg']-previous['yaw_deg']+180)%360-180)
                if movement<.05 and turn<2:hints['previous_path']=last_result.get('path_xy')
            frames[fid] = compute(fid, goal, st.get('route'), remembered=remembered, **hints)
        except (OSError, ValueError, KeyError) as exc:
            frames[fid] = {'frame_id': fid, 'status': 'unavailable', 'reason': f'Local stereo/model inputs unavailable ({type(exc).__name__}).',
                           'segments': [], 'ribbons': [], 'is_ground_projection': True}
            preferred_turn=None;last_step=None;last_result=None
            continue
        last_step=st;last_result=frames[fid]
        preferred_turn=last_result.get('turn_direction') if last_result.get('status') in ('u_turn','turn_required','out_of_view') else None
        confirmed = frames[fid].pop('_confirmed_global', None)
        if confirmed is not None and len(confirmed):
            memory.append((index,confirmed))
    return frames

def _route_waypoint(frame, route, origin, forward, right, ahead_m=6.):
    """First point of the remaining map route at least `ahead_m` from the rover (as lat/lon), else None."""
    r = 3396190.
    places = frame['places']
    rover = np.asarray(places['corrected_rover_origin_global_ned_m'])[:2]
    for p in route:
        xy = np.radians([p['lat'], p['lon']])*r
        if np.linalg.norm(xy-rover) >= ahead_m:
            return {'lat': p['lat'], 'lon': p['lon']}
    return None

def compute(frame_id, goal=None, route=None, remembered=None, preferred_turn=None, previous_path=None):
    frame = next((f for f in data.frame_catalog() if f['id'] == frame_id), None)
    empty = {'frame_id': frame_id, 'segments': [], 'ribbons': [], 'is_ground_projection': True}
    if frame is None:
        return {**empty, 'status': 'unavailable', 'reason': 'Unknown source frame.'}
    if not frame.get('camera_model', {}).get('validation', {}).get('accepted') or not frame.get('xye_path'):
        return {**empty, 'status': 'insufficient_geometry', 'reason': 'Validated stereo/calibration is missing; no depth is invented.'}
    files = [data.DATA_DIR/frame[k] for k in ('xyz_path', 'xye_path', 'image_path')]
    if frame.get('display_path'):
        files.append(data.DATA_DIR/frame['display_path'])
    files += [data.DATA_DIR/'surface.onnx', data.DATA_DIR/'model-card.json']
    files += [p for p in (data.DATA_DIR/'navigation.onnx',data.DATA_DIR/'navigation-model-card.json') if p.exists()]
    key = hashlib.sha256(json.dumps([[(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files], frame['camera_model'], asdict(DEFAULT)], sort_keys=True).encode()).hexdigest()
    terrain, inference = _terrain_cached(frame_id, key)
    terrain, tracked_cells = with_memory(frame, terrain, remembered)
    confirmed_here = confirmed_obstacles_global(frame, terrain)
    origin, forward, right = frame_basis(frame)
    if route:
        aim = _route_waypoint(frame, route, origin, forward, right)
        if aim is not None:
            goal = aim
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
    result = plan_observed(terrain, target, waypoint=bool(route), preferred_turn=preferred_turn, previous_path=previous_path)
    projection = project_path(frame, terrain, result['path_xy']) if result['path_xy'] else {'segments': [], 'ribbons': []}
    projection['obstacles'] = project_obstacles(frame, terrain, exclude=tracked_cells)
    projection['tracked_obstacles'] = project_cells(frame, terrain, tracked_cells)
    projection['distant_candidates'] = project_candidates(frame, terrain)
    if result['status'] in ('ready', 'partial', 'u_turn') and not projection['segments']:
        result['status'] = 'out_of_view'
        result['reason'] = 'Computed ground route is occluded or outside stereo-confirmed camera pixels.'
    return {**empty, **result, **projection, 'settings': asdict(DEFAULT),
            'geometry': {'support_pixels': terrain['support_pixels'], 'observed_cells': int(terrain['observed'].sum()),
                         'obstacle_cells': int(terrain['blocked'].sum()), 'footprint_safe_cells': int(terrain['free'].sum()),
                         'required_obstacle_distance_m': round(DEFAULT.half_width_m_body+DEFAULT.clearance_m+DEFAULT.cell_m/math.sqrt(2),3),
                         'body_width_m': 2*DEFAULT.half_width_m_body,
                         'nonterrain_pixels_excluded':terrain['nonterrain_pixels_excluded'],
                         'candidate_rock_min_measured_width_m':DEFAULT.min_rock_width_m,
                         'candidate_rock_min_relief_lower_limit_m':DEFAULT.rock_relief_m,
                         'triangle_interpolation_max_edge_m': DEFAULT.max_triangle_edge_m},
            'segmentation': inference,
            'tracking': {'remembered_cells_used': int(len(tracked_cells)), 'memory_range_m': MEMORY_RANGE_M, 'memory_frames': MEMORY_FRAMES,
                         'current_reconfirmed_cells':terrain.get('memory_reconfirmed_cells',0),
                         'note': 'Stereo-confirmed obstacles carried in PLACES coordinates. Current clear ground removes a track; current raised-surface evidence renews it. Unseen tracks expire after 12 frames.'},
            '_confirmed_global': confirmed_here,
            'lookahead':{'candidates':terrain['distant_rock_candidates'],'range_m':DEFAULT.preview_range_m,
                         'extends_drivable_surface':False,
                         'note':'Distant stereo-supported rock candidates bias the near route. No unobserved continuation is asserted.'},
            'limitations': 'Research local corridor only (near stereo error <=0.10 m). Camera obstacle/self-hardware perception has limited labelled-scene validation. Unmeasured cells are not assumed to be obstacles; the measured share of the corridor is reported. Distant candidates never establish free ground. Global map is unchanged.'}
