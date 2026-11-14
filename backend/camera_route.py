"""Draw a planned map route inside an archived camera image using measured stereo XYZ.

Every valid camera pixel has a measured SITE-frame XYZ point. Through the registered
SITE->global transform each pixel gets a ground position on the planning map. A route
sample is drawn at the image position of the nearest camera pixel on the ground, only
if that pixel lies within a small tolerance of the route. Ground hidden behind rocks
has no pixel and therefore produces a gap. No synthetic camera model or perspective is
invented.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates

from .data import DATA_DIR, regions
from .camera_model import site_points_to_pixels

R = 3396190.0


def _xy(ned: np.ndarray, meta: dict) -> np.ndarray:
    lat = np.degrees(ned[:, 0] / R); lon = np.degrees(ned[:, 1] / R)
    lat_ts = math.radians(meta.get("standard_parallel_deg", 0))
    x = meta["radius_m"] * math.cos(lat_ts) * np.radians((lon - meta.get("central_lon_deg", 0) + 180) % 360 - 180)
    y = meta["radius_m"] * np.radians(lat - meta.get("central_lat_deg", 0))
    return np.column_stack([x, y])


def ground_pixels(frame: dict, meta: dict):
    xyz = np.load(DATA_DIR / frame["xyz_path"]).astype(np.float64)
    valid = np.isfinite(xyz).all(-1) & (xyz != 0).any(-1)
    p = frame["places"]
    rot = np.array(p["archive_site_to_global_ned_rotation"]); tr = np.array(p["archive_site_to_global_ned_translation_m"])
    rows, cols = np.nonzero(valid)
    ned = xyz[rows, cols] @ rot.T + tr
    return _xy(ned, meta), rows, cols, xyz.shape[:2], ned


def project_route(frame: dict, meta: dict, route_cells: list, tolerance_m: float) -> dict:
    ground, rows, cols, shape, ned = ground_pixels(frame, meta)
    tree = cKDTree(ground)
    cell = meta["cell_m"]
    centres = np.array([[meta["west_m"] + (c + .5) * cell, meta["north_m"] - (r + .5) * cell] for r, c in route_cells], dtype=np.float64)
    # 10 cm samples along each leg (vectorised; same values as the former per-sample loop).
    if len(centres) > 1:
        delta = centres[1:] - centres[:-1]
        seg = np.hypot(delta[:, 0], delta[:, 1])
        counts = np.maximum(1, (seg / 0.1).astype(int))
        starts = np.r_[0., np.cumsum(seg)[:-1]]
        leg = np.repeat(np.arange(len(seg)), counts)
        t = (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)) / counts[leg]
        samples = centres[:-1][leg] + delta[leg] * t[:, None]
        along = starts[leg] + seg[leg] * t
        samples = np.vstack([samples, centres[-1:]]); total = float(starts[-1] + seg[-1]); along = np.r_[along, total]
    else:
        samples = centres.copy(); along = np.zeros(len(centres)); total = 0.0
    dist, idx = tree.query(samples, k=1)
    hit = dist <= tolerance_m
    model = frame.get('camera_model')
    projected = None
    if model and model.get('validation',{}).get('accepted'):
        # Keep vertical support in the measured stereo datum (the orbital DEM
        # has a documented vertical offset), but retain the planned horizontal
        # coordinates. No guessed perspective or horizontal route is drawn.
        world = ned[idx].copy()
        world[:, 0] = samples[:, 1]
        world[:, 1] = samples[:, 0]
        dem_region=regions.get(meta.get('id'))
        if dem_region is not None:
            def heights(xy):
                rc=np.array([(meta['north_m']-xy[:,1])/cell-.5,(xy[:,0]-meta['west_m'])/cell-.5])
                return map_coordinates(np.asarray(dem_region.elevation),rc,order=1,mode='constant',cval=np.nan)
            # The labelled stereo and areoid DEM have a measured vertical datum
            # difference. A robust vertical-only alignment does not change any
            # horizontal camera or map placement.
            ground_h=heights(ground)
            vertical_delta=-ned[:,2]-ground_h
            finite=np.isfinite(vertical_delta)&(np.abs(vertical_delta)<5)
            datum=float(np.median(vertical_delta[finite])) if finite.any() else 0.
            dem_h=heights(samples)+datum
            supported=hit.copy()
            world[~supported,2]=-dem_h[~supported]
            # Beyond stereo support, a calibrated DEM projection is a display
            # estimate, never a new obstacle observation or occupancy assertion.
            hit=np.isfinite(dem_h)
        else:
            supported=hit.copy()
        p = frame['places']
        archive = (world-np.asarray(p['archive_site_to_global_ned_translation_m'])) @ np.asarray(p['archive_site_to_global_ned_rotation'])
        projected = site_points_to_pixels(archive, model)
        height, width = model['native_shape']
        hit &= np.isfinite(projected).all(-1) & (projected[:,0]>=0) & (projected[:,0]<width) & (projected[:,1]>=0) & (projected[:,1]<height)
        # A nearer measured surface at the projected pixel occludes the route.
        uv = np.nan_to_num(projected)
        rr = np.clip((uv[:,1]/height*shape[0]).astype(int),0,shape[0]-1)
        cc = np.clip((uv[:,0]/width*shape[1]).astype(int),0,shape[1]-1)
        native_xyz = np.load(DATA_DIR/frame['xyz_path'])
        observed = native_xyz[rr,cc]
        camera_in_site = np.asarray(model['C']) @ np.asarray(model['rover_to_site_rotation']).T + np.asarray(model['rover_origin_in_site_m'])
        observed_range = np.linalg.norm(observed-camera_in_site,axis=1)
        route_range = np.linalg.norm(archive-camera_in_site,axis=1)
        has_depth = (observed!=0).any(-1) & np.isfinite(observed).all(-1)
        hit &= ~has_depth | (route_range <= observed_range + max(.3,tolerance_m))
        if dem_region is not None:
            camera_global=camera_in_site @ np.asarray(p['archive_site_to_global_ned_rotation']).T + np.asarray(p['archive_site_to_global_ned_translation_m'])
            # DEM line-of-sight test. This does not resolve sub-metre rocks.
            for fraction in np.linspace(.05,.95,28):
                ray=camera_global[None,:]+fraction*(world-camera_global)
                terrain_h=heights(np.column_stack((ray[:,1],ray[:,0])))+datum
                hit &= ~np.isfinite(terrain_h) | (-ray[:,2]>=terrain_h-.15)
    segments, current = [], []
    for i in range(len(samples)):
        if hit[i]:
            j = idx[i]
            point = ([float(projected[i,0])/width, float(projected[i,1])/height] if projected is not None else
                     [(cols[j]+.5)/shape[1],(rows[j]+.5)/shape[0]])
            if current and np.hypot(point[0]-current[-1][0],point[1]-current[-1][1])>.1:
                segments.append(current);current=[]
            current.append([round(point[0], 4),round(point[1], 4),round(float(along[i]), 2)])
        elif current:
            segments.append(current); current = []
    if current:
        segments.append(current)
    segments = [s for s in segments if len(s) >= 3]
    rover = np.array(frame["places"]["corrected_rover_origin_global_ned_m"], dtype=float)
    visible = sorted({round(p[2], 1) for s in segments for p in s})
    return {"frame_id": frame["id"], "segments": segments, "tolerance_m": tolerance_m,
            "visible_route_m": round(sum(s[-1][2] - s[0][2] for s in segments), 2),
            "visible_along_range_m": [visible[0], visible[-1]] if visible else None,
            "route_length_m": round(total, 2),
            "camera_ground_range_m": [round(float(v), 1) for v in np.percentile(np.hypot(*(ned[:, :2] - rover[:2]).T), [5, 95])],
            "method": ("PDS CAHVORE projection; stereo-supported heights nearby, vertically aligned native DEM beyond stereo; depth and DEM occlusion" if projected is not None else
                       "nearest measured stereo ground pixel within tolerance; gaps where the route is out of view or occluded")}


def view_mission(frame: dict, meta: dict) -> dict:
    """Start at the rover; goal at a measured ground point near the image centre column, far range."""
    ground, rows, cols, shape, ned = ground_pixels(frame, meta)
    rover = np.array(frame["places"]["corrected_rover_origin_global_ned_m"], dtype=float)
    rng = np.hypot(*(ned[:, :2] - rover[:2]).T)
    centre = np.abs(cols - shape[1] / 2) < shape[1] * .12
    far = np.percentile(rng[centre], 80) if centre.any() else np.percentile(rng, 80)
    pick = np.argmin(np.where(centre, np.abs(rng - far), np.inf))
    to_ll = lambda n, e: {"lat": math.degrees(n / R), "lon": (math.degrees(e / R) + 180) % 360 - 180}
    return {"start": to_ll(rover[0], rover[1]), "goal": to_ll(ned[pick, 0], ned[pick, 1]), "goal_range_m": round(float(rng[pick]), 1),
            "note": "Start is the PLACES-localized rover position; goal is a measured stereo ground point near the image centre."}
