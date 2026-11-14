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

from .data import DATA_DIR

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
    samples, along = [], []
    total = 0.0
    for a, b in zip(centres, centres[1:]):
        seg = float(np.hypot(*(b - a))); n = max(1, int(seg / 0.1))
        for t in np.arange(n) / n:
            samples.append(a + (b - a) * t); along.append(total + seg * t)
        total += seg
    if len(centres):
        samples.append(centres[-1]); along.append(total)
    samples = np.array(samples); along = np.array(along)
    dist, idx = tree.query(samples, k=1)
    hit = dist <= tolerance_m
    segments, current = [], []
    for i in range(len(samples)):
        if hit[i]:
            j = idx[i]
            current.append([round((cols[j] + .5) / shape[1], 4), round((rows[j] + .5) / shape[0], 4), round(float(along[i]), 2)])
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
            "method": "nearest measured stereo ground pixel within tolerance; gaps where the route is out of view or occluded"}


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
